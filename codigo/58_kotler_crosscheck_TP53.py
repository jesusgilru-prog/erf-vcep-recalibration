"""
Robustez: repite E-RF de TP53 usando Kotler et al. 2018 (Mol Cell,
urn:mavedb:00000059-a-1, ~10.000 variantes del dominio de union a ADN medidas
por crecimiento celular) en vez de Giacomelli et al. 2018, para comprobar que
el hallazgo (LR~1 del umbral oficial BayesDel_noAF en la region ambigua) no
depende de un unico ensayo funcional. Ensayo distinto (crecimiento vs
transactivacion combinada), celulas distintas, solo el dominio DBD (no la
proteina completa).
"""
import csv
import importlib
import json
import re
import statistics
import sys

import numpy as np
from sklearn.mixture import GaussianMixture

sys.path.insert(0, "/home/jesus/paper_msh6/codigo")
c52 = importlib.import_module("52_recalibracion_LR_funcional_erf")

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}
SEMILLA_GMM = 20260814
RNG_SEED_BOOT = 20260816
UMBRAL_POSTERIOR = 0.9
CORTE_SUPPORTING = 2.08
CORTE_MODERATE = 4.3


def cargar_kotler():
    scores = {}
    with open("/home/jesus/paper_msh6/datos/tp53_extension/tp53_kotler_growth_scores.csv") as f:
        for row in csv.DictReader(f):
            hgvs = row["hgvs_pro"]
            m = re.match(r"^p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})$", hgvs)
            if not m:
                continue
            aa1_3, pos, aa2_3 = m.groups()
            wt_aa, mut_aa = AA3_TO_1.get(aa1_3.capitalize()), AA3_TO_1.get(aa2_3.capitalize())
            if wt_aa is None or mut_aa is None or wt_aa == mut_aa:
                continue
            if row["score"] in ("NA", ""):
                continue
            scores.setdefault((int(pos), mut_aa), []).append(float(row["score"]))
    return {k: statistics.median(v) for k, v in scores.items()}


def main():
    real = cargar_kotler()
    print(f"Kotler: {len(real)} sustituciones missense unicas con score")

    hotspots = [(175, "H"), (248, "Q"), (273, "H"), (282, "W")]
    for pos, aa in hotspots:
        print(f"  hotspot {pos}{aa}: score={real.get((pos, aa))}")

    with open("/home/jesus/paper_msh6/datos/tp53_extension/TP53_bayesdel_noaf.json") as f:
        raw = json.load(f)
    oficial = {}
    for k, v in raw.items():
        p, a = k.rsplit("_", 1)
        oficial[(int(p), a)] = v

    keys = sorted(real.keys())
    y = np.array([real[k] for k in keys])
    gmm = GaussianMixture(n_components=2, random_state=SEMILLA_GMM, n_init=10)
    gmm.fit(y.reshape(-1, 1))
    post = gmm.predict_proba(y.reshape(-1, 1))
    medias = gmm.means_.ravel()
    idx_patho = int(np.argmax(medias))
    idx_ben = int(np.argmin(medias))
    print(f"GMM Kotler: medias={medias.tolist()}, pesos={gmm.weights_.tolist()}")
    post_patho = {k: float(post[i, idx_patho]) for i, k in enumerate(keys)}
    post_ben = {k: float(post[i, idx_ben]) for i, k in enumerate(keys)}

    keys_inequivoca = [k for k in keys if post_patho[k] > UMBRAL_POSTERIOR or post_ben[k] > UMBRAL_POSTERIOR]
    keys_ambigua = [k for k in keys if k not in set(keys_inequivoca)]
    y_hard = [1 if post_patho[k] > UMBRAL_POSTERIOR else 0 for k in keys_inequivoca]
    print(f"Region inequivoca: n={len(keys_inequivoca)} ({sum(y_hard)} patogenica-dura, "
          f"{len(y_hard)-sum(y_hard)} benigna-dura)")
    print(f"Region ambigua: n={len(keys_ambigua)}")

    cov_inequivoca = [k for k in keys_inequivoca if k in oficial]
    cov_ambigua = [k for k in keys_ambigua if k in oficial]
    y_hard_cov = [1 if post_patho[k] > UMBRAL_POSTERIOR else 0 for k in cov_inequivoca]
    print(f"Cobertura BayesDel: {len(cov_inequivoca)}/{len(keys_inequivoca)} inequivoca, "
          f"{len(cov_ambigua)}/{len(keys_ambigua)} ambigua")

    resultado = {"gmm_medias": medias.tolist(), "gmm_pesos": gmm.weights_.tolist(),
                 "n_total": len(keys), "n_inequivoca": len(keys_inequivoca),
                 "n_ambigua": len(keys_ambigua), "tramos": {}}
    tramos = [
        ("BP4 (umbral<=-0.008)", "benigno", -0.008, CORTE_SUPPORTING),
        ("PP3 (umbral>=0.16)", "patho", 0.16, CORTE_SUPPORTING),
    ]
    for etiqueta, direccion, umbral, nominal in tramos:
        ld = c52.lr_duro_con_ic(oficial, cov_inequivoca, y_hard_cov, umbral, direccion,
                                 seed=RNG_SEED_BOOT + hash(etiqueta) % 10000)
        lb = c52.lr_blando_con_ic(oficial, cov_ambigua, post_patho, post_ben, umbral, direccion,
                                   seed=RNG_SEED_BOOT + 1 + hash(etiqueta) % 10000)
        if ld is not None:
            ld["nominal_esperado"] = nominal
        lb["nominal_esperado"] = nominal
        lb["ic_incluye_1"] = bool(lb["ic95"][0] is not None and lb["ic95"][0] <= 1.0 <= lb["ic95"][1])
        resultado["tramos"][etiqueta] = {"duro": ld, "blando_ambigua": lb}
        print(f"{etiqueta}: nominal_LR>={nominal} | "
              f"duro LR={ld['lr_puntual']:.3g} IC95={ld['ic95']} | "
              f"blando(titular) LR={lb['lr_puntual']:.3g} IC95={lb['ic95']}")

    with open("/home/jesus/paper_msh6/datos/resultado_TP53_kotler_crosscheck.json", "w") as f:
        json.dump(resultado, f, indent=2)
    print("\nGuardado: datos/resultado_TP53_kotler_crosscheck.json")


if __name__ == "__main__":
    main()
