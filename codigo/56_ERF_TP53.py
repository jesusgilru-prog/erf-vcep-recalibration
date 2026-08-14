"""
E-RF para TP53: mismo procedimiento que codigo/52_ (MSH2), pero con los
umbrales REALES de la especificacion oficial ClinGen TP53 VCEP (GN009 v2.4)
para BayesDel_addAF: BP4_Supporting si -0.008<BayesDel<0.16, BP4_Moderate si
BayesDel<=-0.008, PP3 (fuerza minima garantizada Supporting, ver
codigo/55_ sobre la simplificacion de aGVGD) si BayesDel>=0.16.
"""
import importlib
import sys

import numpy as np
from sklearn.mixture import GaussianMixture

sys.path.insert(0, "/home/jesus/paper_msh6/codigo")
c55 = importlib.import_module("55_analisis_completo_TP53")
c52 = importlib.import_module("52_recalibracion_LR_funcional_erf")

SEMILLA_GMM = 20260814
RNG_SEED_BOOT = 20260814
N_BOOT = 10000
UMBRAL_POSTERIOR = 0.9
CORTE_SUPPORTING = 2.08
CORTE_MODERATE = 4.3


def main():
    real = c55.cargar_dms()
    keys = sorted(real.keys())
    y = np.array([real[k] for k in keys])
    gmm = GaussianMixture(n_components=2, random_state=SEMILLA_GMM, n_init=10)
    gmm.fit(y.reshape(-1, 1))
    post = gmm.predict_proba(y.reshape(-1, 1))
    medias = gmm.means_.ravel()
    idx_patho = int(np.argmax(medias))
    idx_ben = int(np.argmin(medias))
    print(f"GMM TP53: medias={medias.tolist()}, pesos={gmm.weights_.tolist()}")
    post_patho = {k: float(post[i, idx_patho]) for i, k in enumerate(keys)}
    post_ben = {k: float(post[i, idx_ben]) for i, k in enumerate(keys)}

    keys_inequivoca = [k for k in keys if post_patho[k] > UMBRAL_POSTERIOR or post_ben[k] > UMBRAL_POSTERIOR]
    keys_ambigua = [k for k in keys if k not in set(keys_inequivoca)]
    y_hard = [1 if post_patho[k] > UMBRAL_POSTERIOR else 0 for k in keys_inequivoca]
    print(f"Region inequivoca: n={len(keys_inequivoca)} ({sum(y_hard)} patogenica-dura, "
          f"{len(y_hard)-sum(y_hard)} benigna-dura)")
    print(f"Region ambigua (titular): n={len(keys_ambigua)}")

    oficial = c55.cargar_bayesdel()
    cov_inequivoca = [k for k in keys_inequivoca if k in oficial]
    cov_ambigua = [k for k in keys_ambigua if k in oficial]
    y_hard_cov = [1 if post_patho[k] > UMBRAL_POSTERIOR else 0 for k in cov_inequivoca]
    print(f"Cobertura BayesDel: {len(cov_inequivoca)}/{len(keys_inequivoca)} inequivoca, "
          f"{len(cov_ambigua)}/{len(keys_ambigua)} ambigua")

    tramos = {
        "BP4_Supporting": ("benigno", -0.008, CORTE_SUPPORTING),
        "BP4_Moderate": ("benigno", None, CORTE_MODERATE),  # umbral <=-0.008, ver nota
        "PP3_Supporting_o_mas": ("patho", 0.16, CORTE_SUPPORTING),
    }
    # BP4_Moderate usa el mismo corte que BP4_Supporting (-0.008) pero
    # comprueba si alcanza Moderate (4.3) en vez de Supporting (2.08) --
    # mismo umbral de score, dos preguntas de fuerza distintas.
    resultado = {"gmm_medias": medias.tolist(), "gmm_pesos": gmm.weights_.tolist(),
                 "n_total": len(keys), "n_inequivoca": len(keys_inequivoca),
                 "n_ambigua": len(keys_ambigua), "tramos": {}}

    for etiqueta, (direccion, umbral, nominal) in [
        ("BP4_Supporting/Moderate (umbral -0.008, comprobar Supporting)", ("benigno", -0.008, CORTE_SUPPORTING)),
        ("BP4 mismo umbral, exigir Moderate", ("benigno", -0.008, CORTE_MODERATE)),
        ("PP3 (>=Supporting, umbral 0.16)", ("patho", 0.16, CORTE_SUPPORTING)),
    ]:
        ld = c52.lr_duro_con_ic(oficial, cov_inequivoca, y_hard_cov, umbral, direccion,
                                 seed=RNG_SEED_BOOT + hash(etiqueta) % 10000)
        lb = c52.lr_blando_con_ic(oficial, cov_ambigua, post_patho, post_ben, umbral, direccion,
                                   seed=RNG_SEED_BOOT + 1 + hash(etiqueta) % 10000)
        if ld is not None:
            ld["nominal_esperado"] = nominal
            ld["alcanza_nominal_duro"] = bool(ld["lr_puntual"] >= nominal)
        lb["nominal_esperado"] = nominal
        lb["alcanza_nominal_blando"] = bool(lb["lr_puntual"] >= nominal)
        lb["ic_incluye_1"] = bool(lb["ic95"][0] is not None and lb["ic95"][0] <= 1.0 <= lb["ic95"][1])
        resultado["tramos"][etiqueta] = {"umbral_score": umbral, "duro": ld, "blando_ambigua": lb}
        print(f"{etiqueta}: umbral={umbral}, nominal_LR>={nominal} | "
              f"duro LR={ld['lr_puntual']:.3g} IC95={ld['ic95']} | "
              f"blando(titular) LR={lb['lr_puntual']:.3g} IC95={lb['ic95']}")

    import json
    with open("/home/jesus/paper_msh6/datos/resultado_TP53_ERF.json", "w") as f:
        json.dump(resultado, f, indent=2)
    print("\nGuardado: datos/resultado_TP53_ERF.json")


if __name__ == "__main__":
    main()
