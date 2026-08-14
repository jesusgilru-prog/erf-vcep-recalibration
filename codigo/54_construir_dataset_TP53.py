"""
Construye el dataset de verdad funcional real para TP53 (segundo gen ancla del
proyecto, Li-Fraumeni, fuera de la familia MMR), mismo formato que
dataset_H0_MSH2.json, a partir del score combinado de Giacomelli et al. 2018
(Nature Genetics, 3 ensayos: nutlin-3 x2 + etoposido, urn:mavedb:00000068-0-1),
usado como verdad funcional PRIMARIA -- mismo papel que Jia et al. 2021 (HAP1)
para MSH2: cobertura casi-genomica del gen completo (no solo un dominio), y
citado explicitamente por la especificacion oficial ClinGen TP53 VCEP (GN009)
para PS3/BS3 (evidencia funcional "loss of function (LOF) by the majority of
eligible assays").

Signo verificado empiricamente (no asumido): los 4 hotspots patogenicos mas
conocidos de TP53 (R175H, R248Q, R273H, R282W) tienen score MUY positivo
(0.81-1.22), la mediana global es -0.135 -- score ALTO = mas danino, misma
convencion que score_danino en MSH2 (no hace falta invertir signo).
"""
import csv
import json
import re

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}


def main():
    dataset = []
    n_no_score = 0
    n_no_parse = 0
    with open("/home/jesus/paper_msh6/datos/tp53_extension/tp53_giacomelli_combined_scores.csv") as f:
        for row in csv.DictReader(f):
            hgvs = row["hgvs_pro"]
            m = re.match(r"^p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})$", hgvs)
            if not m:
                n_no_parse += 1
                continue
            aa1_3, pos, aa2_3 = m.groups()
            wt_aa, mut_aa = AA3_TO_1.get(aa1_3.capitalize()), AA3_TO_1.get(aa2_3.capitalize())
            if wt_aa is None or mut_aa is None or wt_aa == mut_aa:
                continue
            if row["score"] in ("NA", ""):
                n_no_score += 1
                continue
            score = float(row["score"])
            dataset.append({
                "posicion": int(pos), "wt_aa": wt_aa, "mut_aa": mut_aa,
                "score_dms": score, "gen": "TP53", "score_danino": score,
            })

    print(f"TP53: {len(dataset)} sustituciones missense con score real "
          f"({n_no_score} sin score, {n_no_parse} filas no-missense/sin parsear)")

    out_path = "/home/jesus/paper_msh6/datos/dataset_H0_TP53.json"
    with open(out_path, "w") as f:
        json.dump(dataset, f)
    print(f"Guardado: {out_path}")


if __name__ == "__main__":
    main()
