"""
Cobertura de ClinVar para MSH2, MLH1, MSH6, PMS2 y farmacogenes (RUNX1, CYP2D6,
SLCO1B1, DPYD, UGT1A1). Verifica las cifras de la Propuesta 7 contra el fichero
primario descargado el 5 de agosto de 2026.
"""
import csv
import sys
from collections import Counter, defaultdict

csv.field_size_limit(sys.maxsize)

PATH = "/home/jesus/paper_msh6/datos/variant_summary.txt"
GENES = ["MSH2", "MLH1", "MSH6", "PMS2", "RUNX1", "CYP2D6", "SLCO1B1", "DPYD", "UGT1A1"]

by_gene_total = Counter()
by_gene_class = defaultdict(Counter)
total_rows = 0
total_grch38 = 0
global_class = Counter()

with open(PATH, encoding="utf-8", errors="replace") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        total_rows += 1
        if row["Assembly"] != "GRCh38":
            continue
        total_grch38 += 1
        global_class[row["ClinicalSignificance"]] += 1
        gs = row["GeneSymbol"]
        # GeneSymbol puede venir como "MSH2" o compuesto "MSH2|OTHER" en loci solapados
        genes_in_row = set(gs.split("|"))
        for g in GENES:
            if g in genes_in_row:
                by_gene_total[g] += 1
                by_gene_class[g][row["ClinicalSignificance"]] += 1

print(f"Filas totales en el fichero: {total_rows}")
print(f"Filas GRCh38: {total_grch38}")
print()
print("=== Recuento por gen (GRCh38, variantes únicas de ClinVar) ===")
for g in GENES:
    print(f"{g}: {by_gene_total[g]}")

print()
print("=== Detalle de clasificación por gen ===")
for g in GENES:
    print(f"\n--- {g} (n={by_gene_total[g]}) ---")
    for cls, n in by_gene_class[g].most_common(15):
        print(f"  {cls}: {n}")
