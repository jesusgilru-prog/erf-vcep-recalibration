"""
E-RF para BRCA1, tercer gen ancla del proyecto, con el umbral REAL de la
especificacion oficial ClinGen ENIGMA BRCA1/BRCA2 VCEP (GN092, v1.2,
verificado contra cspec.genome.network): BayesDel_noAF >=0.28 (PP3), <=0.15
(BP4), aplicable SOLO a variantes dentro de los dominios funcionales
clinicamente relevantes (RING aa2-101, coiled-coil aa1391-1424, BRCT
aa1650-1857) -- fuera de esos dominios la regla no aplica, asi que se
restringe el analisis a esas posiciones, igual que hace la propia
especificacion (no es una eleccion arbitraria del analisis).

Verdad funcional: Findlay et al. 2018 (saturation genome editing, cell
survival), ya en datos/dataset_BRCA1_SGE.json (n=2086, 1954 dentro de los 3
dominios).
"""
import importlib
import json
import sys

import numpy as np
from sklearn.mixture import GaussianMixture

sys.path.insert(0, "/home/jesus/paper_msh6/codigo")
c52 = importlib.import_module("52_recalibracion_LR_funcional_erf")

SEMILLA_GMM = 20260814
RNG_SEED_BOOT = 20260815
UMBRAL_POSTERIOR = 0.9
CORTE_SUPPORTING = 2.08
CORTE_MODERATE = 4.3

DOMINIOS = [(2, 101), (1391, 1424), (1650, 1857)]


def en_dominio(pos):
    return any(lo <= pos <= hi for lo, hi in DOMINIOS)


def cargar_dms_brca1():
    with open("/home/jesus/paper_msh6/datos/dataset_BRCA1_SGE.json") as f:
        d = json.load(f)
    return {(x["posicion"], x["mut_aa"]): x["score_danino"] for x in d}


def cargar_bayesdel_brca1():
    with open("/home/jesus/paper_msh6/datos/brca1_extension/BRCA1_bayesdel_noaf.json") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        p, a = k.rsplit("_", 1)
        out[(int(p), a)] = v
    return out


def main():
    real_todo = cargar_dms_brca1()
    real = {k: v for k, v in real_todo.items() if en_dominio(k[0])}
    print(f"BRCA1 SGE: n_total={len(real_todo)}, dentro de dominios funcionales "
          f"(RING/CC/BRCT, donde aplica la regla PP3/BP4 de GN092)={len(real)}")

    oficial = cargar_bayesdel_brca1()
    keys = sorted(real.keys())
    y = np.array([real[k] for k in keys])
    print(f"score_danino: media={y.mean():.3f}, mediana={np.median(y):.3f}, "
          f"min={y.min():.3f}, max={y.max():.3f}")

    gmm = GaussianMixture(n_components=2, random_state=SEMILLA_GMM, n_init=10)
    gmm.fit(y.reshape(-1, 1))
    post = gmm.predict_proba(y.reshape(-1, 1))
    medias = gmm.means_.ravel()
    idx_patho = int(np.argmax(medias))
    idx_ben = int(np.argmin(medias))
    print(f"GMM BRCA1 (solo dominios): medias={medias.tolist()}, pesos={gmm.weights_.tolist()}")
    post_patho = {k: float(post[i, idx_patho]) for i, k in enumerate(keys)}
    post_ben = {k: float(post[i, idx_ben]) for i, k in enumerate(keys)}

    keys_inequivoca = [k for k in keys if post_patho[k] > UMBRAL_POSTERIOR or post_ben[k] > UMBRAL_POSTERIOR]
    keys_ambigua = [k for k in keys if k not in set(keys_inequivoca)]
    y_hard = [1 if post_patho[k] > UMBRAL_POSTERIOR else 0 for k in keys_inequivoca]
    print(f"Region inequivoca: n={len(keys_inequivoca)} ({sum(y_hard)} patogenica-dura, "
          f"{len(y_hard)-sum(y_hard)} benigna-dura)")
    print(f"Region ambigua (titular): n={len(keys_ambigua)}")

    cov_inequivoca = [k for k in keys_inequivoca if k in oficial]
    cov_ambigua = [k for k in keys_ambigua if k in oficial]
    y_hard_cov = [1 if post_patho[k] > UMBRAL_POSTERIOR else 0 for k in cov_inequivoca]
    print(f"Cobertura BayesDel: {len(cov_inequivoca)}/{len(keys_inequivoca)} inequivoca, "
          f"{len(cov_ambigua)}/{len(keys_ambigua)} ambigua")

    resultado = {"gmm_medias": medias.tolist(), "gmm_pesos": gmm.weights_.tolist(),
                 "n_total_dominios": len(real), "n_inequivoca": len(keys_inequivoca),
                 "n_ambigua": len(keys_ambigua), "tramos": {}}

    tramos = [
        ("BP4_Supporting (umbral<=0.15)", "benigno", 0.15, CORTE_SUPPORTING),
        ("PP3 (umbral>=0.28)", "patho", 0.28, CORTE_SUPPORTING),
    ]
    for etiqueta, direccion, umbral, nominal in tramos:
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
        print(f"{etiqueta}: nominal_LR>={nominal} | "
              f"duro LR={ld['lr_puntual']:.3g} IC95={ld['ic95']} | "
              f"blando(titular) LR={lb['lr_puntual']:.3g} IC95={lb['ic95']}")

    with open("/home/jesus/paper_msh6/datos/resultado_BRCA1_ERF.json", "w") as f:
        json.dump(resultado, f, indent=2)
    print("\nGuardado: datos/resultado_BRCA1_ERF.json")


if __name__ == "__main__":
    main()
