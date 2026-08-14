"""
Sonda de 1 dia (reencuadre de Claude, debate del 11-ago-2026): en vez de medir
"tiene API" (un hecho de la web que caduca), medir para cada especificacion
VCEP liberada de ClinGen que clasificacion ACMG es alcanzable usando SOLO
evidencia libremente computable (sin laboratorio, sin dato de paciente, sin
pagar nada), bajo las reglas de combinacion categoricas de Richards et al. 2015
(las que citan literalmente las especificaciones ya verificadas, GN138/GN139).

Alcance de "computable" declarado explicitamente (conservador a proposito,
sesga el techo hacia abajo, no hacia arriba):
  - PM2, BS1, BS2, BA1: frecuencia poblacional (gnomAD, publico y gratuito).
  - PP3 / BP4: evidencia computacional, SOLO si la herramienta que cita la
    especificacion es un predictor conocido servido en dbNSFP/myvariant.info
    (ya verificado en este proyecto: AlphaMissense, REVEL, CADD, EVE,
    PrimateAI, ClinPred, VARITY, BayesDel, MetaRNN, SpliceAI, MutationTaster,
    PolyPhen, SIFT, MaxEntScan, Align-GVGD, dbscSNV, M-CAP, MutPred, PROVEAN,
    FATHMM) -- o un portal publico ya verificado por este proyecto (MAPP/PP2
    via HCI-LOVD). Si cita una herramienta/portal no reconocido, se clasifica
    "manual_o_no_verificado" y NO se cuenta como computable (conservador).
  - Todo lo demas (PVS1, PS1-4, PM1/3/4/5/6, PP1/2/4/5, BP1/2/3/5/6/7, BS3/4)
    requiere segregacion familiar, ensayo funcional, fenotipo del paciente o
    juicio experto especifico del gen -- se excluye del techo computable por
    diseno, aunque a veces sea derivable de datos publicos en casos concretos
    (ej. PM1 con estructura). Esto hace el techo una cota INFERIOR conservadora,
    no una medida exacta -- declarado asi, no una limitacion escondida.
"""
import glob
import json
import re

DIR_SPECS = "/home/jesus/paper_msh6/datos/fuentes_primarias/cspec_all"
FECHA = "20260811"

CODIGOS_FRECUENCIA = {"PM2", "BS1", "BS2", "BA1"}
CODIGOS_COMPUTACIONALES = {"PP3", "BP4"}

HERRAMIENTAS_DBNSFP_CONOCIDAS = [
    "AlphaMissense", "REVEL", "CADD", "EVE", "PrimateAI", "ClinPred", "VARITY",
    "BayesDel", "MetaRNN", "SpliceAI", "MutationTaster", "PolyPhen", "SIFT",
    "MaxEntScan", "Align-GVGD", "AlignGVGD", "dbscSNV", "M-CAP", "MCAP",
    "MutPred", "PROVEAN", "FATHMM", "MutationAssessor", "VEST4", "LRT",
    "PhyloP", "PhastCons", "GERP",
]
HERRAMIENTAS_PORTAL_VERIFICADO = ["MAPP/PP2", "MAPP", "HCI-PRIOR", "hci-lovd", "hci-priors"]

ORDEN_FUERZA_PATOGENICA = ["Supporting", "Moderate", "Strong", "Very Strong"]
ORDEN_FUERZA_BENIGNA = ["Supporting", "Strong", "Stand Alone"]


def cargar_specs():
    paths = sorted(glob.glob(f"{DIR_SPECS}/GN*_{FECHA}.json"))
    specs = []
    for p in paths:
        with open(p) as f:
            specs.append(json.load(f))
    return specs


def genes_de_spec(d):
    genes = set()
    for rs in d.get("ruleSets", []):
        for g in rs.get("genes", []):
            if g.get("label"):
                genes.add(g["label"])
    return sorted(genes)


def es_aplicable(valor):
    """Bug corregido 11-ago-2026 (encontrado por Codex en revision):
    ClinGen usa 'Applicable', pero tambien 'Applicable with VCEP specification'
    y variantes -- comparar por igualdad exacta con 'applicable' perdia estas
    ultimas (ej. GN005/GN007/GN010, que asi salian sin PP3 pese a tenerlo)."""
    return (valor or "").strip().lower().startswith("applicable")


def extraer_codigo(d, label_buscado):
    """Devuelve (fuerzas_aplicables, descripcion_texto) UNIENDO todas las
    apariciones del label en TODOS los ruleSets (bug corregido 11-ago-2026:
    la version anterior devolvia en el primer match, perdiendo apariciones en
    ruleSets posteriores si un gen tiene varios)."""
    fuerzas_aplicables = []
    descripciones = []
    desc_general = ""
    encontrado = False
    for rs in d.get("ruleSets", []):
        for cc in rs.get("criteriaCodes", []):
            if cc.get("label") != label_buscado:
                continue
            encontrado = True
            if not desc_general:
                desc_general = cc.get("description", "")
            for es in cc.get("evidenceStrengths", []):
                if es_aplicable(es.get("applicability")):
                    fuerzas_aplicables.append(es.get("label"))
                    if es.get("description"):
                        descripciones.append(es["description"])
    if not encontrado:
        return [], ""
    return fuerzas_aplicables, (desc_general + " " + " ".join(descripciones))


def clasificar_herramienta_pp3_bp4(texto):
    if not texto.strip():
        return "sin_codigo"
    for h in HERRAMIENTAS_PORTAL_VERIFICADO:
        if h.lower() in texto.lower():
            return "portal_verificado_por_este_proyecto"
    for h in HERRAMIENTAS_DBNSFP_CONOCIDAS:
        if re.search(re.escape(h), texto, re.I):
            return "conocido_dbnsfp_myvariant"
    return "manual_o_no_verificado"


def mejor_fuerza(fuerzas, orden):
    disponibles = [f for f in fuerzas if f in orden]
    if not disponibles:
        return None
    return max(disponibles, key=lambda f: orden.index(f))


def techo_patogenico(pm2_fuerza, pp3_fuerza):
    """Richards 2015, solo con hasta 2 lineas de evidencia patogenica computable."""
    niveles = []
    if pm2_fuerza:
        niveles.append(pm2_fuerza)
    if pp3_fuerza:
        niveles.append(pp3_fuerza)
    n_vs = niveles.count("Very Strong")
    n_s = niveles.count("Strong")
    n_m = niveles.count("Moderate")
    n_sup = niveles.count("Supporting")

    # Pathogenic
    if n_vs >= 1 and (n_s >= 1 or n_m >= 2 or (n_m >= 1 and n_sup >= 1) or n_sup >= 2):
        return "Pathogenic"
    if n_s >= 2:
        return "Pathogenic"
    if n_s >= 1 and (n_m >= 3 or (n_m >= 2 and n_sup >= 2) or (n_m >= 1 and n_sup >= 4)):
        return "Pathogenic"
    # Likely Pathogenic
    if n_vs >= 1 and n_m >= 1:
        return "Likely Pathogenic"
    if n_s >= 1 and n_m >= 1:
        return "Likely Pathogenic"
    if n_s >= 1 and n_sup >= 2:
        return "Likely Pathogenic"
    if n_m >= 3:
        return "Likely Pathogenic"
    if n_m >= 2 and n_sup >= 2:
        return "Likely Pathogenic"
    if n_m >= 1 and n_sup >= 4:
        return "Likely Pathogenic"
    if niveles:
        return "VUS (evidencia insuficiente para mover la categoria)"
    return "sin evidencia patogenica computable disponible"


def techo_benigno(ba1_aplica, bs1_fuerza, bs2_fuerza, bp4_fuerza):
    niveles = []
    if bs1_fuerza:
        niveles.append(bs1_fuerza)
    if bs2_fuerza:
        niveles.append(bs2_fuerza)
    if bp4_fuerza:
        niveles.append(bp4_fuerza)
    n_sa = 1 if ba1_aplica else 0
    n_s = niveles.count("Strong")
    n_sup = niveles.count("Supporting")

    if n_sa >= 1:
        return "Benign (via BA1 si la variante cumple el umbral de frecuencia)"
    if n_s >= 2:
        return "Benign"
    if (n_s >= 1 and n_sup >= 1) or n_sup >= 2:
        return "Likely Benign"
    if niveles:
        return "VUS (evidencia insuficiente para mover la categoria)"
    return "sin evidencia benigna computable disponible"


def procesar_spec(d):
    gn_id = d.get("@id", "").rsplit("/", 1)[-1]
    afiliacion = (d.get("affiliation") or {}).get("label", "")
    genes = genes_de_spec(d)

    resultado_codigos = {}
    for label in CODIGOS_FRECUENCIA | CODIGOS_COMPUTACIONALES:
        fuerzas, texto = extraer_codigo(d, label)
        resultado_codigos[label] = {"fuerzas_aplicables": fuerzas}
        if label in CODIGOS_COMPUTACIONALES:
            resultado_codigos[label]["clasificacion_herramienta"] = clasificar_herramienta_pp3_bp4(texto)
            resultado_codigos[label]["texto_extracto"] = texto[:300]

    pm2_f = mejor_fuerza(resultado_codigos["PM2"]["fuerzas_aplicables"], ORDEN_FUERZA_PATOGENICA)
    pp3_computable = resultado_codigos["PP3"]["clasificacion_herramienta"] in (
        "conocido_dbnsfp_myvariant", "portal_verificado_por_este_proyecto")
    pp3_f = mejor_fuerza(resultado_codigos["PP3"]["fuerzas_aplicables"], ORDEN_FUERZA_PATOGENICA) if pp3_computable else None

    ba1_aplica = bool(resultado_codigos["BA1"]["fuerzas_aplicables"])
    bs1_f = mejor_fuerza(resultado_codigos["BS1"]["fuerzas_aplicables"], ORDEN_FUERZA_BENIGNA)
    bs2_f = mejor_fuerza(resultado_codigos["BS2"]["fuerzas_aplicables"], ORDEN_FUERZA_BENIGNA)
    bp4_computable = resultado_codigos["BP4"]["clasificacion_herramienta"] in (
        "conocido_dbnsfp_myvariant", "portal_verificado_por_este_proyecto")
    bp4_f = mejor_fuerza(resultado_codigos["BP4"]["fuerzas_aplicables"], ORDEN_FUERZA_BENIGNA) if bp4_computable else None

    techo_pat = techo_patogenico(pm2_f, pp3_f)
    techo_ben = techo_benigno(ba1_aplica, bs1_f, bs2_f, bp4_f)

    return {
        "gn_id": gn_id, "afiliacion": afiliacion, "genes": genes,
        "pp3_clasificacion_herramienta": resultado_codigos["PP3"]["clasificacion_herramienta"],
        "bp4_clasificacion_herramienta": resultado_codigos["BP4"]["clasificacion_herramienta"],
        "pp3_texto_extracto": resultado_codigos["PP3"]["texto_extracto"],
        "techo_patogenico_computable": techo_pat,
        "techo_benigno_computable": techo_ben,
        "detalle_codigos": resultado_codigos,
    }


def main():
    specs = cargar_specs()
    print(f"Analizando {len(specs)} especificaciones VCEP liberadas...\n")

    resultados = [procesar_spec(d) for d in specs]

    conteo_herramienta_pp3 = {}
    conteo_techo_pat = {}
    conteo_techo_ben = {}
    for r in resultados:
        k = r["pp3_clasificacion_herramienta"]
        conteo_herramienta_pp3[k] = conteo_herramienta_pp3.get(k, 0) + 1
        conteo_techo_pat[r["techo_patogenico_computable"]] = conteo_techo_pat.get(r["techo_patogenico_computable"], 0) + 1
        conteo_techo_ben[r["techo_benigno_computable"]] = conteo_techo_ben.get(r["techo_benigno_computable"], 0) + 1

    print("=== Clasificacion de la herramienta citada para PP3, las 122 especificaciones ===")
    for k, v in sorted(conteo_herramienta_pp3.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} ({100*v/len(resultados):.1f}%)")

    print("\n=== Techo PATOGENICO alcanzable con evidencia 100% computable (PM2+PP3 solo) ===")
    for k, v in sorted(conteo_techo_pat.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} ({100*v/len(resultados):.1f}%)")

    print("\n=== Techo BENIGNO alcanzable con evidencia 100% computable (BA1/BS1/BS2/BP4) ===")
    for k, v in sorted(conteo_techo_ben.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} ({100*v/len(resultados):.1f}%)")

    # caso MSH6/PMS2 explicito, para contrastar con el resto
    for r in resultados:
        if "MSH6" in r["genes"] or "PMS2" in r["genes"]:
            print(f"\n  [{r['gn_id']}] {r['genes']}: techo patogenico = {r['techo_patogenico_computable']}, "
                  f"techo benigno = {r['techo_benigno_computable']}, herramienta PP3 = {r['pp3_clasificacion_herramienta']}")

    manual_o_ambiguo = [r for r in resultados if r["pp3_clasificacion_herramienta"] == "manual_o_no_verificado"]
    print(f"\n=== Especificaciones con PP3 citando herramienta NO reconocida "
          f"(candidatas a 'brecha real', requieren revision manual una a una) ===")
    for r in manual_o_ambiguo:
        print(f"  [{r['gn_id']}] {r['afiliacion']} | genes={r['genes']} | "
              f"extracto: {r['pp3_texto_extracto'][:150]}")

    with open("/home/jesus/paper_msh6/datos/resultado_techo_automatizacion_acmg.json", "w") as f:
        json.dump({
            "n_specs": len(resultados),
            "resumen_herramienta_pp3": conteo_herramienta_pp3,
            "resumen_techo_patogenico": conteo_techo_pat,
            "resumen_techo_benigno": conteo_techo_ben,
            "resultados_por_spec": resultados,
        }, f, indent=2, ensure_ascii=False)
    print("\n\nGuardado: datos/resultado_techo_automatizacion_acmg.json")


if __name__ == "__main__":
    main()
