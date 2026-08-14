"""
ESM-1v (Meier et al. 2021, ensemble de 5, disenado especificamente para efecto
de variante) contra: oficial, ESM-2, AlphaMissense, y para MSH2 la verdad
funcional real. Completa la tabla de 4 modelos independientes.
"""
import json

import numpy as np
from scipy.stats import spearmanr

import importlib
c40 = importlib.import_module("40_comparar_alphamissense")


def cargar_esm1v(gene):
    with open(f"/home/jesus/paper_msh6/datos/esm1v_zeroshot/{gene}_esm1v_ensemble5_zeroshot.json") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        pos_str, aa = k.rsplit("_", 1)
        out[(int(pos_str), aa)] = -v
    return out


def main():
    resultado = {}
    for gene, select_db in [("MSH2", "MSH2_priors"), ("MSH6", "MSH6_priors"), ("PMS2", "PMS2_priors")]:
        print(f"\n{'='*70}\n{gene}: ESM-1v vs oficial / ESM-2 / AlphaMissense\n{'='*70}")
        oficial = c40.cargar_oficial(select_db)
        esm2 = c40.cargar_esm(gene)
        am = c40.cargar_alphamissense(gene)
        esm1v = cargar_esm1v(gene)
        print(f"  ESM-1v: {len(esm1v)} variantes")

        for nombre, otro in [("oficial", oficial), ("ESM-2", esm2), ("AlphaMissense", am)]:
            comunes = sorted(set(esm1v) & set(otro))
            y1 = np.array([esm1v[k] for k in comunes])
            y2 = np.array([otro[k] for k in comunes])
            rho, p = spearmanr(y1, y2)
            print(f"  ESM-1v vs {nombre}: rho={rho:.4f} (n={len(comunes)}, p={p:.3g})")
            resultado.setdefault(gene, {})[f"esm1v_vs_{nombre}"] = {"rho": float(rho), "n": len(comunes)}

    print(f"\n{'='*70}\nMSH2: ESM-1v vs verdad funcional REAL (DMS HAP1)\n{'='*70}")
    esm1v_msh2 = cargar_esm1v("MSH2")
    with open("/home/jesus/paper_msh6/datos/dataset_H0_MSH2.json") as f:
        dms = json.load(f)
    real = {(d["posicion"], d["mut_aa"]): d["score_danino"] for d in dms}
    comunes = sorted(set(esm1v_msh2) & set(real))
    y1 = np.array([esm1v_msh2[k] for k in comunes])
    y2 = np.array([real[k] for k in comunes])
    rho, p = spearmanr(y1, y2)
    print(f"  ESM-1v vs verdad funcional real: rho={rho:.4f} (n={len(comunes)}, p={p:.3g})")
    resultado["MSH2_vs_verdad_real"] = {"rho": float(rho), "n": len(comunes)}

    print("\n=== RESUMEN: acierto de los 4 modelos frente a la verdad funcional real de MSH2 ===")
    print("  Oficial (MAPP/PP2):  rho=0.2957")
    print("  ESM-2 (650M):        rho=0.2796")
    print("  AlphaMissense:       rho=0.3514")
    print(f"  ESM-1v (ensemble5):  rho={rho:.4f}")

    with open("/home/jesus/paper_msh6/datos/resultado_comparacion_esm1v.json", "w") as f:
        json.dump(resultado, f, indent=2)
    print("\nGuardado: datos/resultado_comparacion_esm1v.json")


if __name__ == "__main__":
    main()
