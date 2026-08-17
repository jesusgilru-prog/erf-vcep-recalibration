"""
Corrige bloqueante 4 (parte 2) de la ronda de revision multi-IA 2026-08-17:
Conclusions afirmaba AUC "good in all three cases examined" pero nunca se
habia calculado la AUC de BRCA1 contra ClinVar (solo se calculo para MSH2 y
TP53). Se calcula aqui, con la MISMA metodologia que codigo/52_/56_ para
TP53 (ClinVar alta confianza, DeLong no aplica porque solo hay un predictor
BayesDel aqui, se reporta AUC simple con IC bootstrap), restringida a los
n=1723 sustituciones dentro de los 3 dominios funcionales (RING/CC/BRCT)
donde aplica la regla PP3/BP4 de GN092 -- misma poblacion que el resto de
los resultados de BRCA1 en el paper, para que sea comparable.
"""
import importlib
import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/home/jesus/paper_msh6/codigo")
c52 = importlib.import_module("52_recalibracion_LR_funcional_erf")
c57 = importlib.import_module("57_ERF_BRCA1")

RNG_SEED_BOOT = 20260817
N_BOOT = 10000


def main():
    pat, ben = c52.construir_calibracion_clinvar("BRCA1")
    bayesdel = c57.cargar_bayesdel_brca1()

    def en_dominio(pos):
        return c57.en_dominio(pos)

    pat_dom = sorted(set(k for k in pat if en_dominio(k[0]) and k in bayesdel))
    ben_dom = sorted(set(k for k in ben if en_dominio(k[0]) and k in bayesdel))
    print(f"BRCA1 ClinVar alta confianza, dentro de dominios, con score BayesDel: "
          f"{len(pat_dom)} patogenicas, {len(ben_dom)} benignas")

    y = np.array([1] * len(pat_dom) + [0] * len(ben_dom))
    s = np.array([bayesdel[k] for k in pat_dom] + [bayesdel[k] for k in ben_dom])
    auc = roc_auc_score(y, s)
    print(f"AUC BRCA1 (dentro de dominios) = {auc:.4f}, n_pat={len(pat_dom)}, n_ben={len(ben_dom)}")

    rng = np.random.default_rng(RNG_SEED_BOOT)
    n = len(y)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yb, sb = y[idx], s[idx]
        if len(set(yb.tolist())) < 2:
            continue
        boots.append(roc_auc_score(yb, sb))
    ci = np.percentile(boots, [2.5, 97.5]).tolist()
    print(f"IC95 bootstrap ({len(boots)} validos): [{ci[0]:.4f}, {ci[1]:.4f}]")

    # tambien sin restriccion de dominio, para contexto (no es la poblacion
    # que usa el resto del paper, pero es informativo declararlo)
    pat_all = sorted(set(k for k in pat if k in bayesdel))
    ben_all = sorted(set(k for k in ben if k in bayesdel))
    y_all = np.array([1] * len(pat_all) + [0] * len(ben_all))
    s_all = np.array([bayesdel[k] for k in pat_all] + [bayesdel[k] for k in ben_all])
    auc_all = roc_auc_score(y_all, s_all)
    print(f"AUC BRCA1 (sin restriccion de dominio, contexto) = {auc_all:.4f}, "
          f"n_pat={len(pat_all)}, n_ben={len(ben_all)}")

    out = {
        "auc_dominio": float(auc), "ic95_dominio": ci,
        "n_pat_dominio": len(pat_dom), "n_ben_dominio": len(ben_dom),
        "auc_todo_gen_contexto": float(auc_all),
        "n_pat_todo_gen": len(pat_all), "n_ben_todo_gen": len(ben_all),
    }
    with open("/home/jesus/paper_msh6/datos/resultado_62_auc_brca1_clinvar.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
