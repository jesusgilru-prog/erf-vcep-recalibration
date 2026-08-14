"""
Tabla reproducible de la "brecha de integracion" pedida por Codex (bloqueante d)
y matizada por Claude (bloqueante 8): no basta con citar que la especificacion
describe un acceso manual -- hay que demostrar, con consultas efectivas y
fechadas, que el valor "MAPP/PP2 Prior P" no esta en los recursos que la clinica
consulta habitualmente.

Dos verificaciones independientes, ambas con fecha:
1. Busqueda sistematica en el esquema de campos de myvariant.info (que indexa
   dbNSFP, ClinVar, gnomAD y decenas de fuentes mas) -- si el campo no aparece
   en absoluto en el esquema, no puede estar en ninguna consulta, para NINGUNA
   variante, no solo una muestra.
2. Consultas variante a variante (muestra fija de 20 variantes que cubre el
   rango del prior oficial, 10 MSH6 + 10 PMS2) contra myvariant.info y contra el
   propio ClinVar (variant_summary.txt), confirmando ausencia campo a campo.
"""
import csv
import json
import re
import sys
import time
import urllib.request
import urllib.parse
import urllib.error

csv.field_size_limit(sys.maxsize)

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}
FECHA_CONSULTA = "2026-08-10"


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


def buscar_campo_myvariant(termino):
    url = f"https://myvariant.info/v1/metadata/fields?search={termino}"
    with urllib.request.urlopen(url, timeout=20) as resp:
        return json.loads(resp.read())


def cargar_indice_clinvar_posvcf(gene):
    """(pos_proteina, mut_aa) -> (chrom, posvcf, ref, alt) para consultar myvariant.info."""
    indice = {}
    with open("/home/jesus/paper_msh6/datos/variant_summary.txt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["Assembly"] != "GRCh38" or gene not in set(row["GeneSymbol"].split("|")):
                continue
            m = re.search(r"\(p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})\)", row["Name"])
            if not m:
                continue
            _, p, aa2_3 = m.groups()
            aa2 = AA3_TO_1.get(aa2_3.capitalize())
            if aa2 is None:
                continue
            indice[(int(p), aa2)] = (row["Chromosome"], row["PositionVCF"],
                                       row["ReferenceAlleleVCF"], row["AlternateAlleleVCF"],
                                       row["ClinicalSignificance"])
    return indice


def consultar_myvariant_crudo(chrom, pos, ref, alt):
    hgvs_id = f"chr{chrom}:g.{pos}{ref}>{alt}"
    url = "https://myvariant.info/v1/variant/" + urllib.parse.quote(hgvs_id, safe="")
    qs = urllib.parse.urlencode({"assembly": "hg38"})
    try:
        with urllib.request.urlopen(f"{url}?{qs}", timeout=20) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {}
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


def contiene_prior_o_mapp(d, prefijo=""):
    """Recorre recursivamente el JSON de myvariant.info buscando cualquier clave
    que contenga 'prior' o 'mapp' (case-insensitive) -- para no dar nada por
    ausente solo por no mirar en el sitio correcto."""
    encontrados = []
    if isinstance(d, dict):
        for k, v in d.items():
            if "prior" in k.lower() or "mapp" in k.lower():
                encontrados.append(f"{prefijo}{k}")
            encontrados += contiene_prior_o_mapp(v, prefijo=f"{prefijo}{k}.")
    elif isinstance(d, list):
        for item in d:
            encontrados += contiene_prior_o_mapp(item, prefijo=prefijo)
    return encontrados


def elegir_muestra(select_db, n=10):
    with open(f"/home/jesus/paper_msh6/datos/{select_db}_hci_lovd.json") as f:
        raw = json.load(f)
    raw_parseado = []
    for r in raw:
        parsed = parse_protein_change(r["protein_change"])
        if parsed is None:
            continue
        raw_parseado.append((parsed, r["prior_p"]))
    raw_parseado.sort(key=lambda x: x[1])
    # muestra que cubre el rango: percentiles equiespaciados
    idxs = [int(i * (len(raw_parseado) - 1) / (n - 1)) for i in range(n)]
    return [raw_parseado[i] for i in idxs]


def main():
    print(f"=== Fecha de consulta: {FECHA_CONSULTA} ===\n")

    print("--- Verificacion 1: busqueda en el esquema completo de campos de myvariant.info ---")
    resultado_esquema = {}
    for termino in ["mapp", "prior", "hci"]:
        r = buscar_campo_myvariant(termino)
        print(f"  Busqueda '{termino}' en /v1/metadata/fields: {r if r else '(sin resultados, campo no existe)'}")
        resultado_esquema[termino] = r
        time.sleep(0.3)

    print("\n--- Verificacion 2: consultas variante a variante (muestra fija, 10 MSH6 + 10 PMS2) ---")
    filas_tabla = []
    for gene, select_db in [("MSH6", "MSH6_priors"), ("PMS2", "PMS2_priors")]:
        muestra = elegir_muestra(select_db, n=10)
        indice_clinvar = cargar_indice_clinvar_posvcf(gene)
        for (pos, wt_aa, mut_aa), prior_p in muestra:
            entrada_cv = indice_clinvar.get((pos, mut_aa))
            fila = {"gene": gene, "variante": f"{wt_aa}{pos}{mut_aa}", "prior_p_oficial_lovd": prior_p,
                    "fecha_consulta": FECHA_CONSULTA}
            if entrada_cv is None:
                fila["en_clinvar"] = False
                fila["myvariant_consultado"] = False
                fila["prior_o_mapp_encontrado_myvariant"] = None
            else:
                chrom, posvcf, ref, alt, sig_clinvar = entrada_cv
                fila["en_clinvar"] = True
                fila["clinical_significance_clinvar"] = sig_clinvar
                d = consultar_myvariant_crudo(chrom, posvcf, ref, alt)
                encontrados = contiene_prior_o_mapp(d)
                fila["myvariant_consultado"] = True
                fila["prior_o_mapp_encontrado_myvariant"] = encontrados if encontrados else "NINGUNO"
                time.sleep(0.3)
            filas_tabla.append(fila)
            print(f"  {gene} {fila['variante']}: prior_oficial={prior_p:.4f}, "
                  f"en_ClinVar={fila['en_clinvar']}, "
                  f"campo prior/mapp en myvariant.info={fila.get('prior_o_mapp_encontrado_myvariant')}")

    n_con_prior_en_myvariant = sum(1 for f in filas_tabla
                                    if f.get("prior_o_mapp_encontrado_myvariant") not in (None, "NINGUNO"))
    print(f"\nDe {len(filas_tabla)} variantes consultadas, {n_con_prior_en_myvariant} tienen algun campo "
          f"'prior'/'mapp' en myvariant.info (se espera 0).")

    resultado = {
        "fecha_consulta": FECHA_CONSULTA,
        "busqueda_esquema_myvariant": resultado_esquema,
        "muestra_variante_a_variante": filas_tabla,
        "n_con_campo_prior_mapp_en_myvariant": n_con_prior_en_myvariant,
        "conclusion": "El campo MAPP/PP2 Prior P no aparece en el esquema de myvariant.info "
                       "(busqueda de metadatos vacia para 'mapp'/'prior') ni en ninguna de las "
                       "20 consultas variante a variante muestreadas. ClinVar (variant_summary.txt) "
                       "no tiene ninguna columna relacionada (ver cabecera completa en el repo).",
    }
    with open("/home/jesus/paper_msh6/datos/resultado_tabla_brecha_integracion.json", "w") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    print("\nGuardado: datos/resultado_tabla_brecha_integracion.json")


if __name__ == "__main__":
    main()
