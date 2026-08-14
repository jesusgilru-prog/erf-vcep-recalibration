"""
Congela el conjunto de validacion externa (control 25 del preregistro): las variantes
MSH6 (inCAMA, Szabo et al. 2025) y PMS2 (CIMRA, Rayner et al. 2022) con evidencia
funcional real y publicada. Cruza cada variante con la entrada actual de ClinVar
(5-ago-2026) para dejar constancia, ANTES de entrenar nada, de si el propio ClinVar
ya refleja la reclasificacion del paper o si sigue como estaba.

Este fichero (datos/CONJUNTO_VALIDACION_EXTERNA_CONGELADO.json) no se debe modificar
una vez generado. No entra en entrenamiento bajo ninguna circunstancia.
"""
import csv
import json
import re
import sys
from datetime import datetime, timezone

csv.field_size_limit(sys.maxsize)

AA3 = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys", "Q": "Gln",
    "E": "Glu", "G": "Gly", "H": "His", "I": "Ile", "L": "Leu", "K": "Lys",
    "M": "Met", "F": "Phe", "P": "Pro", "S": "Ser", "T": "Thr", "W": "Trp",
    "Y": "Tyr", "V": "Val",
}


def one_to_three(variant_1letter):
    """E220D -> Glu220Asp"""
    m = re.match(r"^([A-Z])(\d+)([A-Z])$", variant_1letter.strip())
    if not m:
        return None
    aa1, pos, aa2 = m.groups()
    if aa1 not in AA3 or aa2 not in AA3:
        return None
    return f"{AA3[aa1]}{pos}{AA3[aa2]}"


with open("/home/jesus/paper_msh6/datos/inCAMA_tabla3_parseada.json") as f:
    incama = json.load(f)
msh6_incama = [r for r in incama if r["gene"] == "MSH6"]

with open("/home/jesus/paper_msh6/datos/CIMRA_tabla1_parseada.json") as f:
    cimra = json.load(f)

# Indexar ClinVar por gen -> lista de (Name, VariationID, ClinicalSignificance, LastEvaluated)
clinvar_by_gene = {"MSH6": [], "PMS2": []}
PATH = "/home/jesus/paper_msh6/datos/variant_summary.txt"
with open(PATH, encoding="utf-8", errors="replace") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        if row["Assembly"] != "GRCh38":
            continue
        genes = set(row["GeneSymbol"].split("|"))
        for g in ("MSH6", "PMS2"):
            if g in genes:
                clinvar_by_gene[g].append(row)

print(f"ClinVar MSH6 filas: {len(clinvar_by_gene['MSH6'])}")
print(f"ClinVar PMS2 filas: {len(clinvar_by_gene['PMS2'])}")


def buscar_en_clinvar(gene, protein_3letter):
    if protein_3letter is None:
        return []
    hits = []
    needle = f"(p.{protein_3letter})"
    for row in clinvar_by_gene[gene]:
        if needle in row["Name"]:
            hits.append({
                "variation_id": row["VariationID"],
                "name": row["Name"],
                "clinical_significance_clinvar_hoy": row["ClinicalSignificance"],
                "last_evaluated": row["LastEvaluated"],
                "review_status": row["ReviewStatus"],
            })
    return hits


frozen = {
    "generado_utc": "2026-08-05T00:00:00Z",
    "nota": "Timestamp fijado a la fecha de descarga de ClinVar (5-ago-2026), no a la hora de ejecucion del script, para no depender de reloj de sistema no disponible en workflows deterministas.",
    "fuente_msh6": "Szabo et al. 2025, Human Mutation, DOI 10.1155/humu/3923193, PMC12433325",
    "fuente_pms2": "Rayner et al. 2022, Human Mutation, DOI 10.1002/humu.24387, PMC9545740",
    "regla": "Este conjunto NUNCA se usa para entrenar ni ajustar hiperparametros. Se evalua una sola vez, al final.",
    "msh6": [],
    "pms2": [],
}

for r in msh6_incama:
    p3 = one_to_three(r["variant_protein"].rstrip("g"))  # 'V509Ag' -> footnote marker
    clinvar_hits = buscar_en_clinvar("MSH6", p3)
    frozen["msh6"].append({
        "variant_1letter": r["variant_protein"],
        "variant_3letter": p3,
        "oddspath_functional_inCAMA": r["oddspath_functional"],
        "clasificacion_clingen_insight_en_paper_2025": r["clasificacion_clingen_insight_actual"],
        "clasificacion_predicha_por_inCAMA": r["clasificacion_predicha_actualizada"],
        "clinvar_hoy_5ago2026": clinvar_hits,
    })

for r in cimra:
    m = re.match(r"^p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})$", r["variant_protein"], re.I)
    p3 = None
    if m:
        aa1, pos, aa2 = m.groups()
        p3 = f"{aa1.capitalize()}{pos}{aa2.capitalize()}"
    clinvar_hits = buscar_en_clinvar("PMS2", p3)
    frozen["pms2"].append({
        "variant_protein": r["variant_protein"],
        "origen": r["source"],
        "oddspath_CIMRA": r["oddspath"],
        "acmg_evidence_strength": r["acmg_evidence_strength"],
        "clasificacion_en_bd_2022": r["clasificacion_en_bd"],
        "clinvar_hoy_5ago2026": clinvar_hits,
    })

n_msh6_con_match = sum(1 for x in frozen["msh6"] if x["clinvar_hoy_5ago2026"])
n_pms2_con_match = sum(1 for x in frozen["pms2"] if x["clinvar_hoy_5ago2026"])

print(f"\nMSH6: {len(frozen['msh6'])} variantes congeladas, {n_msh6_con_match} con match directo en ClinVar por notacion proteica")
print(f"PMS2: {len(frozen['pms2'])} variantes congeladas, {n_pms2_con_match} con match directo en ClinVar por notacion proteica")

out_path = "/home/jesus/paper_msh6/datos/CONJUNTO_VALIDACION_EXTERNA_CONGELADO.json"
with open(out_path, "w") as f:
    json.dump(frozen, f, indent=2, ensure_ascii=False)

import hashlib
with open(out_path, "rb") as f:
    h = hashlib.sha256(f.read()).hexdigest()
print(f"\nSHA256 del fichero congelado: {h}")
print(f"Ruta: {out_path}")
