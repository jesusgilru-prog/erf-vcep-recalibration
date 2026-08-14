"""
Extrae la Tabla 3 de Szabo et al. 2025 (inCAMA, PMC12433325) fila a fila, respetando
las secciones "MSH6 variants" / "MSH2 variants" del propio XML, para no atribuir mal
una variante a un gen (error que se cometio en una primera lectura del texto aplanado).
"""
import re
import json

with open("/tmp/tab3.xml", encoding="utf-8") as f:
    xml = f.read()

rows = re.findall(r"<tr>(.*?)</tr>", xml, re.S)

current_gene = None
current_class = None
records = []

for row in rows:
    cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
    cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
    cells = [re.sub(r"\s+", " ", c) for c in cells]

    if len(cells) == 1 and cells[0] == "":
        continue
    if len(cells) == 1 and "variants" in cells[0]:
        current_gene = cells[0].split()[0]
        continue
    if len(cells) == 1 and ("benign" in cells[0].lower() or "pathogenic" in cells[0].lower()):
        current_class = cells[0].strip()
        continue
    if cells and cells[0].strip() == "Variant":
        continue
    if len(cells) >= 7:
        variant, ddr, repair, oddspath, current_clinvar, acmg, predicted = cells[:7]
        records.append({
            "gene": current_gene,
            "variant_protein": variant.strip(),
            "clase_original_paper": current_class,
            "dna_damage_response": ddr,
            "dna_repair": repair,
            "oddspath_functional": oddspath,
            "clasificacion_clingen_insight_actual": current_clinvar,
            "acmg_evidence_code": acmg,
            "clasificacion_predicha_actualizada": predicted,
        })

print(f"Total filas extraidas: {len(records)}")
by_gene = {}
for r in records:
    by_gene.setdefault(r["gene"], []).append(r)
for g, rs in by_gene.items():
    print(f"  {g}: {len(rs)} variantes")

with open("/home/jesus/paper_msh6/datos/inCAMA_tabla3_parseada.json", "w") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

print("\n=== MSH6 (las que importan para este proyecto) ===")
for r in by_gene.get("MSH6", []):
    print(r["variant_protein"], "|", r["oddspath_functional"], "|", r["clasificacion_clingen_insight_actual"], "->", r["clasificacion_predicha_actualizada"])

print("\n=== MSH2 (usadas por inCAMA para calibrar/probar su propio ensayo, NO son el hueco de este proyecto) ===")
for r in by_gene.get("MSH2", []):
    print(r["variant_protein"], "|", r["oddspath_functional"], "|", r["clasificacion_clingen_insight_actual"], "->", r["clasificacion_predicha_actualizada"])
