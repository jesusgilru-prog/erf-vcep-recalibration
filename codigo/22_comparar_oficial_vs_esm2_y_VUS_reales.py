"""
El experimento decisivo: comparar ESM-2 650M zero-shot contra el prior OFICIAL
(MAPP/PP2 Prior P, hci-lovd.hci.utah.edu, curado por Bryony Thompson, citado
por la especificacion vigente de ClinGen/InSiGHT GN138/GN139) sobre miles de
variantes -- no las 18/51 del conjunto congelado, sino la base completa.

Y, mas importante para el proyecto: aplicar los umbrales OFICIALES (0.11/0.68/
0.81) al prior OFICIAL directamente sobre las VUS reales de ClinVar, sin pasar
por ESM-2 en absoluto. Esa es la cifra que de verdad se puede defender como
evidencia PP3/BP4 segun la guia vigente -- todo lo anterior con ESM-2 era una
aproximacion honesta pero declarada como tal.
"""
import csv
import json
import re
import sys

import numpy as np
from scipy.stats import spearmanr

csv.field_size_limit(sys.maxsize)

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}


def parse_protein_change(pc):
    """'p.S2T' -> (2, 'S', 'T'); tolera codigos de 3 letras tambien."""
    m = re.match(r"^p\.([A-Za-z]{1,3})(\d+)([A-Za-z]{1,3})$", pc)
    if not m:
        return None
    aa1, pos, aa2 = m.groups()
    if len(aa1) == 3:
        aa1 = AA3_TO_1.get(aa1.capitalize())
    if len(aa2) == 3:
        aa2 = AA3_TO_1.get(aa2.capitalize())
    if aa1 is None or aa2 is None or len(aa1) != 1 or len(aa2) != 1:
        return None
    return int(pos), aa1, aa2


def cargar_esm(gene):
    path = f"/home/jesus/paper_msh6/datos/esm2_zeroshot/{gene}_esm2_650M_zeroshot.json"
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        pos_str, aa = k.rsplit("_", 1)
        out[(int(pos_str), aa)] = -v
    return out


def cargar_oficial(gene, select_db):
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


def evidencia_oficial(prior_p):
    if prior_p > 0.81:
        return "PP3_Moderate"
    elif prior_p > 0.68:
        return "PP3_Supporting"
    elif prior_p < 0.11:
        return "BP4_Supporting"
    else:
        return "sin_evidencia"


def extraer_missense(name):
    m = re.search(r"\(p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})\)", name)
    if not m:
        return None
    aa1_3, pos, aa2_3 = m.groups()
    aa1, aa2 = AA3_TO_1.get(aa1_3.capitalize()), AA3_TO_1.get(aa2_3.capitalize())
    if aa1 is None or aa2 is None or aa1 == aa2:
        return None
    return int(pos), aa1, aa2


def cargar_filas_gen(gene):
    filas = []
    with open("/home/jesus/paper_msh6/datos/variant_summary.txt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["Assembly"] != "GRCh38":
                continue
            if gene not in set(row["GeneSymbol"].split("|")):
                continue
            filas.append(row)
    return filas


def procesar_gen(gene, select_db):
    print(f"\n{'='*70}\n{gene}: ESM-2 vs prior oficial (MAPP/PP2, hci-lovd)\n{'='*70}")
    esm = cargar_esm(gene)
    oficial = cargar_oficial(gene, select_db)
    print(f"ESM-2: {len(esm)} sustituciones puntuadas. Oficial: {len(oficial)} variantes con prior.")

    # --- comparacion directa ESM-2 vs oficial, sobre TODAS las variantes compartidas ---
    comunes = set(esm.keys()) & set(oficial.keys())
    print(f"Variantes en comun (comparacion directa, n grande): {len(comunes)}")
    esm_vals = [esm[k] for k in comunes]
    of_vals = [oficial[k] for k in comunes]
    rho, p = spearmanr(esm_vals, of_vals)
    print(f"Correlacion ESM-2 vs prior oficial (Spearman): rho={rho:.4f} (p={p:.3g}, n={len(comunes)})")

    # --- evidencia oficial aplicada directamente a TODAS las VUS reales de ClinVar ---
    filas_gen = cargar_filas_gen(gene)
    vus_unicas = {}
    for row in filas_gen:
        if row["ClinicalSignificance"] != "Uncertain significance":
            continue
        parsed = extraer_missense(row["Name"])
        if parsed is None:
            continue
        pos, wt_aa, mut_aa = parsed
        vus_unicas[(pos, wt_aa, mut_aa)] = row
    print(f"VUS missense unicas de {gene} en ClinVar hoy: {len(vus_unicas)}")

    con_prior_oficial = 0
    conteo = {}
    detalle_vus = []
    for (pos, wt_aa, mut_aa), row in vus_unicas.items():
        prior = oficial.get((pos, mut_aa))
        if prior is None:
            continue
        con_prior_oficial += 1
        ev = evidencia_oficial(prior)
        conteo[ev] = conteo.get(ev, 0) + 1
        detalle_vus.append({"variante": f"{wt_aa}{pos}{mut_aa}", "prior_p_oficial": prior,
                             "evidencia_oficial": ev, "variation_id": row["VariationID"]})

    print(f"VUS con prior OFICIAL disponible (cobertura real de la base de datos): "
          f"{con_prior_oficial}/{len(vus_unicas)} ({100*con_prior_oficial/len(vus_unicas):.1f}%)")
    print("Distribucion de evidencia OFICIAL (sin pasar por ESM-2 en absoluto):")
    for ev, n in sorted(conteo.items(), key=lambda x: -x[1]):
        print(f"  {ev}: {n} ({100*n/con_prior_oficial:.1f}% de las que tienen prior oficial)")

    pp3 = sum(v for k, v in conteo.items() if k.startswith("PP3"))
    bp4 = sum(v for k, v in conteo.items() if k.startswith("BP4"))
    print(f"\nCon alguna evidencia PP3 o BP4 (oficial, directa): {pp3+bp4}/{con_prior_oficial} "
          f"({100*(pp3+bp4)/con_prior_oficial:.1f}%)")

    return {
        "gene": gene, "n_esm": len(esm), "n_oficial": len(oficial), "n_comunes": len(comunes),
        "correlacion_esm_vs_oficial": {"rho": float(rho), "p": float(p), "n": len(comunes)},
        "n_vus_total": len(vus_unicas), "n_vus_con_prior_oficial": con_prior_oficial,
        "cobertura_pct": 100 * con_prior_oficial / len(vus_unicas),
        "distribucion_evidencia_oficial": conteo,
        "pp3_mas_bp4": pp3 + bp4,
        "pct_con_evidencia_oficial": 100 * (pp3 + bp4) / con_prior_oficial if con_prior_oficial else None,
        "detalle_vus": detalle_vus,
    }


def main():
    resultados = {}
    resultados["MSH6"] = procesar_gen("MSH6", "MSH6_priors")
    resultados["PMS2"] = procesar_gen("PMS2", "PMS2_priors")

    with open("/home/jesus/paper_msh6/datos/resultado_oficial_vs_esm2.json", "w") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print("\n\nGuardado: datos/resultado_oficial_vs_esm2.json")


if __name__ == "__main__":
    main()
