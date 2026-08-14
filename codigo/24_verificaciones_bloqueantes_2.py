"""
Tres verificaciones baratas pedidas por la revision externa (Claude, bloqueante 2):

(2.2) Si una fraccion no trivial de las VUS con prior oficial disponible ya fue
      revisada por el panel de expertos InSiGHT (ReviewStatus == "reviewed by
      expert panel") y se dejo en VUS, la premisa "evidencia no integrada" esta
      refutada para esas variantes -- hay que contarlas, no asumir.
(2.3) Si la cobertura oficial de PMS2 (54.6%) esta estructurada por posicion
      (concentrada fuera de la region C-terminal homologa a PMS2CL, exones
      11-15, aprox. residuos 620-862 segun UniProt P54278) en vez de ser un
      hueco aleatorio.
(integridad) Cuantos accessions del campo Name de ClinVar corresponden al
      transcrito de referencia esperado (NM_000179 MSH6 / NM_000535 PMS2) --
      exposicion al riesgo de mezclar variantes de otro transcrito.
"""
import csv
import json
import re
import sys
from collections import Counter

csv.field_size_limit(sys.maxsize)

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}
NM_ESPERADO = {"MSH6": "NM_000179", "PMS2": "NM_000535"}
# PMS2CL (pseudogen) es homologo a los exones 11-15 de PMS2, region C-terminal.
# Limites de residuo aproximados desde UniProt P54278 / literatura (De Vos et al. 2004,
# Vaughn et al. 2010): exon 11 empieza ~codon 620, el gen tiene 862 aa.
PMS2CL_INICIO_RESIDUO = 620


def extraer_missense(name):
    m = re.search(r"\(p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})\)", name)
    if not m:
        return None
    aa1_3, pos, aa2_3 = m.groups()
    aa1, aa2 = AA3_TO_1.get(aa1_3.capitalize()), AA3_TO_1.get(aa2_3.capitalize())
    if aa1 is None or aa2 is None or aa1 == aa2:
        return None
    return int(pos), aa1, aa2


def parse_protein_change(pc):
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


def verificar_gen(gene, select_db):
    print(f"\n{'='*70}\n{gene}\n{'='*70}")
    oficial = cargar_oficial(gene, select_db)
    filas = cargar_filas_gen(gene)

    # --- (integridad) transcrito ---
    nm_esperado = NM_ESPERADO[gene]
    conteo_nm = Counter()
    for row in filas:
        m = re.match(r"^(NM_\d+)", row["Name"])
        conteo_nm[m.group(1) if m else "SIN_NM"] += 1
    print(f"Distribucion de accession en campo Name (top 5): {conteo_nm.most_common(5)}")
    otros = sum(v for k, v in conteo_nm.items() if k != nm_esperado)
    print(f"Filas con accession != {nm_esperado}: {otros}/{len(filas)} ({100*otros/len(filas):.2f}%)")

    # --- (2.2) VUS con prior oficial que YA fueron revisadas por panel de expertos ---
    vus_unicas = {}
    for row in filas:
        if row["ClinicalSignificance"] != "Uncertain significance":
            continue
        parsed = extraer_missense(row["Name"])
        if parsed is None:
            continue
        pos, wt_aa, mut_aa = parsed
        vus_unicas[(pos, wt_aa, mut_aa)] = row

    con_prior = []
    for (pos, wt_aa, mut_aa), row in vus_unicas.items():
        if (pos, mut_aa) in oficial:
            con_prior.append((pos, wt_aa, mut_aa, row))

    conteo_review = Counter(row["ReviewStatus"] for (_, _, _, row) in con_prior)
    print(f"\nVUS missense con prior oficial disponible: {len(con_prior)}")
    print("Desglose por ReviewStatus (¿ya revisadas por panel de expertos?):")
    for status, n in conteo_review.most_common():
        print(f"  {status}: {n} ({100*n/len(con_prior):.1f}%)")
    n_panel = conteo_review.get("reviewed by expert panel", 0)
    print(f"-> 'reviewed by expert panel' entre las VUS con prior oficial: {n_panel}/{len(con_prior)} "
          f"({100*n_panel/len(con_prior):.1f}%)")
    if n_panel > 0:
        print("   Estas VUS fueron evaluadas por InSiGHT y quedaron en VUS pese a la revision experta.")
        print("   Para ellas, 'evidencia no integrada' es una afirmacion mas debil: puede que el panel")
        print("   ya haya considerado evidencia computacional y no haya bastado para reclasificar sola,")
        print("   consistente con el techo Moderate/Supporting del VCEP (no mueve una VUS por si sola).")

    # --- (2.3) estructura posicional de la cobertura (solo relevante para PMS2 / PMS2CL) ---
    resultado_posicion = None
    if gene == "PMS2":
        posiciones_vus = sorted(set(pos for pos, _, _ in vus_unicas.keys()))
        con_prior_pos = set(pos for pos, _, _, _ in con_prior)
        c_term = [p for p in posiciones_vus if p >= PMS2CL_INICIO_RESIDUO]
        n_term = [p for p in posiciones_vus if p < PMS2CL_INICIO_RESIDUO]
        cobertura_cterm = sum(1 for p in c_term if p in con_prior_pos) / len(c_term) if c_term else None
        cobertura_nterm = sum(1 for p in n_term if p in con_prior_pos) / len(n_term) if n_term else None
        print(f"\nEstructura posicional de cobertura PMS2 (frontera PMS2CL en residuo {PMS2CL_INICIO_RESIDUO}):")
        print(f"  Posiciones VUS N-terminal (< {PMS2CL_INICIO_RESIDUO}, fuera de homologia PMS2CL): {len(n_term)}, "
              f"cobertura de prior oficial: {cobertura_nterm}")
        print(f"  Posiciones VUS C-terminal (>= {PMS2CL_INICIO_RESIDUO}, homologa a PMS2CL): {len(c_term)}, "
              f"cobertura de prior oficial: {cobertura_cterm}")
        resultado_posicion = {
            "frontera_pms2cl": PMS2CL_INICIO_RESIDUO,
            "n_posiciones_nterm": len(n_term), "cobertura_nterm": cobertura_nterm,
            "n_posiciones_cterm": len(c_term), "cobertura_cterm": cobertura_cterm,
        }

    return {
        "gene": gene,
        "transcrito_esperado": nm_esperado,
        "distribucion_accession": dict(conteo_nm),
        "n_vus_con_prior": len(con_prior),
        "desglose_review_status_vus_con_prior": dict(conteo_review),
        "n_reviewed_by_expert_panel": n_panel,
        "estructura_posicional_pms2": resultado_posicion,
    }


def main():
    resultados = {"MSH6": verificar_gen("MSH6", "MSH6_priors"),
                  "PMS2": verificar_gen("PMS2", "PMS2_priors")}
    with open("/home/jesus/paper_msh6/datos/resultado_verificaciones_bloqueantes_2.json", "w") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print("\n\nGuardado: datos/resultado_verificaciones_bloqueantes_2.json")


if __name__ == "__main__":
    main()
