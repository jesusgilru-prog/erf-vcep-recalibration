"""
Corrige el ultimo bloqueante de formato senalado por los tres revisores en
ronda 2: la Tabla 1 mezclaba denominadores (concordancia en n=5.460 para
AlphaMissense/ESM-1v, acierto en n=5.193). Aqui se recalcula la concordancia
de los 3 pares (MAPP/PP2 vs ESM-2/AlphaMissense/ESM-1v) TODOS sobre el mismo
n=5.193 de referencia, con IC bootstrap pareado, y se calcula el gap
concordancia-menos-acierto para los 3 pares (no solo el par que maximiza el
gap, MAPP-ESM2), con su propio IC.
"""
import importlib
import json
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, "/home/jesus/paper_msh6/codigo")
c40 = importlib.import_module("40_comparar_alphamissense")
c41 = importlib.import_module("41_comparar_esm1v")

SEMILLA = 20260812
N_BOOT = 2000


def main():
    oficial = c40.cargar_oficial("MSH2_priors")
    esm2 = c40.cargar_esm("MSH2")
    am = c40.cargar_alphamissense("MSH2")
    esm1v = c41.cargar_esm1v("MSH2")
    with open("/home/jesus/paper_msh6/datos/dataset_H0_MSH2.json") as f:
        dms = json.load(f)
    real = {(d["posicion"], d["mut_aa"]): d["score_danino"] for d in dms}

    comun = sorted(set(oficial) & set(esm2) & set(am) & set(esm1v) & set(real))
    print(f"n={len(comun)} (denominador unico para concordancia Y acierto)")
    y_of = np.array([oficial[k] for k in comun])
    y_real = np.array([real[k] for k in comun])
    rng = np.random.default_rng(SEMILLA)
    idx = np.arange(len(comun))

    resultado = {"n": len(comun)}
    gaps = []
    for nombre, modelo in [("ESM-2", esm2), ("AlphaMissense", am), ("ESM-1v", esm1v)]:
        y_m = np.array([modelo[k] for k in comun])
        rho_conc, _ = spearmanr(y_of, y_m)
        rho_acc, _ = spearmanr(y_m, y_real)
        conc_boots, acc_boots, gap_boots = [], [], []
        for _ in range(N_BOOT):
            bi = rng.choice(idx, len(idx), replace=True)
            rc, _ = spearmanr(y_of[bi], y_m[bi])
            ra, _ = spearmanr(y_m[bi], y_real[bi])
            if not (np.isnan(rc) or np.isnan(ra)):
                conc_boots.append(rc)
                acc_boots.append(ra)
                gap_boots.append(rc - ra)
        ci_conc = np.percentile(conc_boots, [2.5, 97.5]).tolist()
        ci_acc = np.percentile(acc_boots, [2.5, 97.5]).tolist()
        ci_gap = np.percentile(gap_boots, [2.5, 97.5]).tolist()
        gap = rho_conc - rho_acc
        print(f"  {nombre}: concordancia={rho_conc:.4f} {ci_conc}, acierto={rho_acc:.4f} {ci_acc}, "
              f"gap={gap:.4f} {ci_gap}")
        resultado[nombre] = {
            "rho_concordancia": float(rho_conc), "ci95_concordancia": ci_conc,
            "rho_acierto": float(rho_acc), "ci95_acierto": ci_acc,
            "gap": float(gap), "ci95_gap": ci_gap,
        }
        gaps.append(gap)

    gap_medio = float(np.mean(gaps))
    print(f"\n  Gap medio de los 3 pares: {gap_medio:.4f} (rango {min(gaps):.4f}-{max(gaps):.4f})")
    resultado["gap_medio_3_pares"] = gap_medio
    resultado["gap_rango_3_pares"] = [float(min(gaps)), float(max(gaps))]

    with open("/home/jesus/paper_msh6/datos/resultado_tabla1_denominador_unico.json", "w") as f:
        json.dump(resultado, f, indent=2)
    print("\nGuardado: datos/resultado_tabla1_denominador_unico.json")


if __name__ == "__main__":
    main()
