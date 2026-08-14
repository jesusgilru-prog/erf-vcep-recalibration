"""
Extrae la Tabla 1 de Rayner et al. 2022 (CIMRA, PMC9545740): variantes PMS2 con OddsPath
y fuerza de evidencia ACMG/AMP, fila a fila desde el XML.
"""
import re
import json

with open("/tmp/cimra_tab1.xml", encoding="utf-8") as f:
    xml = f.read()

rows = re.findall(r"<tr[^>]*>(.*?)</tr>", xml, re.S)

records = []
for row in rows:
    cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)
    cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
    cells = [re.sub(r"\s+", " ", c) for c in cells]
    if not cells or cells[0] in ("Source", ""):
        continue
    if len(cells) < 6:
        continue
    source, variant, prior_p, activity, oddspath, acmg = cells[:6]
    if not re.match(r"^p\.", variant, re.I):
        continue
    classification = cells[6] if len(cells) > 6 else ""
    records.append({
        "source": source,
        "variant_protein": variant,
        "prior_p": prior_p,
        "cimra_assay_activity": activity,
        "oddspath": oddspath,
        "acmg_evidence_strength": acmg,
        "clasificacion_en_bd": classification,
    })

print(f"Total filas extraidas: {len(records)}")
sources = {}
for r in records:
    sources.setdefault(r["source"], 0)
    sources[r["source"]] += 1
print("Por origen:", sources)

with open("/home/jesus/paper_msh6/datos/CIMRA_tabla1_parseada.json", "w") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

for r in records[:15]:
    print(r["variant_protein"], "|", r["oddspath"], "|", r["acmg_evidence_strength"], "|", r["clasificacion_en_bd"])
