"""
Compara AlphaMissense (DeepMind, Cheng et al. 2023, Science -- modelo de IA
aplicado a medicina, publicado y en uso clinico real) contra: (1) el prior
oficial MAPP/PP2 de MSH2/MSH6/PMS2, (2) ESM-2 650M, y (3) para MSH2, la verdad
funcional real (DMS HAP1) -- misma logica que 38_, ahora con un tercer modelo.
Cobertura parcial declarada (la base de AlphaMissense usada, ~71M filas, no es
saturacion completa del proteoma; ver datos/alphamissense/README).
"""
import json
import re

import numpy as np
from scipy.stats import spearmanr

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}


def parse_protein_change(pc):
    m = re.match(r"^p\.([A-Za-z]{1,3})(\d+)([A-Za-z]{1,3})$", pc)
    if not m:
        return None
    aa1, pos, aa2 = m.groups()
    if len(aa1) == 3:
        aa1 = AA3_TO_1.get(aa1.capitalize())
    if len(aa2) == 3:
        aa2 = AA3_TO_1.get(aa2.capitalize())
    if aa1 is None or aa2 is None:
        return None
    return int(pos), aa1, aa2


def cargar_oficial(select_db):
    with open(f"/home/jesus/paper_msh6/datos/{select_db}_hci_lovd.json") as f:
        raw = json.load(f)
    out = {}
    for r in raw:
        parsed = parse_protein_change(r["protein_change"])
        if parsed is None:
            continue
        pos, wt_aa, mut_aa = parsed
        out[(pos, mut_aa)] = r["prior_p"]
    return out


def cargar_esm(gene):
    with open(f"/home/jesus/paper_msh6/datos/esm2_zeroshot/{gene}_esm2_650M_zeroshot.json") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        pos_str, aa = k.rsplit("_", 1)
        out[(int(pos_str), aa)] = -v
    return out


def cargar_alphamissense(gene):
    with open(f"/home/jesus/paper_msh6/datos/alphamissense/{gene}_alphamissense.json") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        pos_str, aa = k.rsplit("_", 1)
        out[(int(pos_str), aa)] = v["score"]
    return out


def comparar(gene, select_db):
    print(f"\n{'='*70}\n{gene}: AlphaMissense vs oficial vs ESM-2\n{'='*70}")
    oficial = cargar_oficial(select_db)
    esm = cargar_esm(gene)
    am = cargar_alphamissense(gene)
    print(f"  AlphaMissense: {len(am)} variantes. Oficial: {len(oficial)}. ESM-2: {len(esm)}.")

    comunes_of = sorted(set(am) & set(oficial))
    y_am, y_of = np.array([am[k] for k in comunes_of]), np.array([oficial[k] for k in comunes_of])
    rho_am_of, p1 = spearmanr(y_am, y_of)
    print(f"  AlphaMissense vs oficial: rho={rho_am_of:.4f} (n={len(comunes_of)}, p={p1:.3g})")

    comunes_esm = sorted(set(am) & set(esm))
    y_am2, y_esm = np.array([am[k] for k in comunes_esm]), np.array([esm[k] for k in comunes_esm])
    rho_am_esm, p2 = spearmanr(y_am2, y_esm)
    print(f"  AlphaMissense vs ESM-2:   rho={rho_am_esm:.4f} (n={len(comunes_esm)}, p={p2:.3g})")

    return {"gene": gene, "n_am": len(am),
            "rho_am_vs_oficial": float(rho_am_of), "n_am_of": len(comunes_of),
            "rho_am_vs_esm2": float(rho_am_esm), "n_am_esm": len(comunes_esm)}


def comparar_msh2_con_verdad_real():
    print(f"\n{'='*70}\nMSH2: AlphaMissense vs verdad funcional REAL (DMS HAP1)\n{'='*70}")
    am = cargar_alphamissense("MSH2")
    with open("/home/jesus/paper_msh6/datos/dataset_H0_MSH2.json") as f:
        dms = json.load(f)
    real = {(d["posicion"], d["mut_aa"]): d["score_danino"] for d in dms}
    comunes = sorted(set(am) & set(real))
    y_am, y_real = np.array([am[k] for k in comunes]), np.array([real[k] for k in comunes])
    rho, p = spearmanr(y_am, y_real)
    print(f"  AlphaMissense vs verdad funcional real: rho={rho:.4f} (n={len(comunes)}, p={p:.3g})")
    print(f"  Comparar: ESM-2 vs verdad funcional real (38_) = rho=0.2796; "
          f"oficial vs verdad funcional real (38_) = rho=0.2957")
    return {"rho_am_vs_real": float(rho), "n": len(comunes)}


def main():
    resultado = {}
    for gene, select_db in [("MSH2", "MSH2_priors"), ("MSH6", "MSH6_priors"), ("PMS2", "PMS2_priors")]:
        resultado[gene] = comparar(gene, select_db)
    resultado["MSH2_vs_verdad_real"] = comparar_msh2_con_verdad_real()

    with open("/home/jesus/paper_msh6/datos/resultado_comparacion_alphamissense.json", "w") as f:
        json.dump(resultado, f, indent=2)
    print("\nGuardado: datos/resultado_comparacion_alphamissense.json")


if __name__ == "__main__":
    main()
