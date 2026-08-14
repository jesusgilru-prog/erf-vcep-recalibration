"""
Corrige el fallo real que encontro Claude en la auditoria del 25_: comparar solo
la INTERSECCION de filas que ambos parsers ya coincidian en ver es ciego, por
construccion, a las filas que un parser pierde. Aqui se re-descarga la base
COMPLETA (todas las paginas, ambos genes) con el parser independiente (HTML
estructural), se afirma programaticamente la fila de cabecera de la tabla
contra el orden de columnas esperado (Exon, DNA change, RNA change, Protein,
Custom PP2.1 score, MAPP score, MAPP/PP2 Prior P, Reference, Template,
Technique, DB-ID -- verificado leyendo <TH> el 10-ago-2026, ver
datos/fuentes_primarias/), y se compara contra el JSON original por UNION de
claves, no interseccion -- para poder ver exactamente que se gano o se perdio.
"""
import json
import re
import time
import urllib.request

BASE = "http://hci-lovd.hci.utah.edu/variants.php"
HEADER_ESPERADO = [
    "Exon", "DNA&nbsp;change", "RNA&nbsp;change", "Protein", "Custom&nbsp;PP2.1&nbsp;score",
    "MAPP&nbsp;score", "MAPP/PP2&nbsp;Prior&nbsp;P", "Reference", "Template", "Technique", "DB-ID",
]


def fetch(select_db, page, limit=100, max_reintentos=3):
    url = f"{BASE}?action=search_unique&select_db={select_db}&limit={limit}&page={page}"
    for intento in range(max_reintentos):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("iso-8859-1", errors="replace")
        except Exception:
            if intento == max_reintentos - 1:
                raise
            time.sleep(2)


def verificar_cabecera(texto_html):
    """Falla ruidosamente si el orden de columnas de la pagina no es el esperado
    (verificado el 10-ago-2026) -- exactamente el chequeo que Claude senalo que faltaba."""
    headers = re.findall(r"<TH[^>]*>(.*?)</TH>", texto_html, re.S)
    headers_limpios = [re.sub(r"<[^>]+>", "", h).strip() for h in headers]
    headers_limpios = [h for h in headers_limpios if h and h not in ("", "&nbsp;")]
    # los headers relevantes son un subconjunto contiguo; buscamos el orden exacto
    encontrado = False
    for i in range(len(headers_limpios) - len(HEADER_ESPERADO) + 1):
        if headers_limpios[i:i + len(HEADER_ESPERADO)] == HEADER_ESPERADO:
            encontrado = True
            break
    if not encontrado:
        raise RuntimeError(
            f"CABECERA DE TABLA INESPERADA. Se esperaba {HEADER_ESPERADO} como subsecuencia "
            f"contigua, no encontrada en: {headers_limpios}. El orden de columnas pudo cambiar "
            f"en el portal -- NO usar el resultado sin revisar manualmente."
        )
    return True


def parse_independiente(texto_html):
    filas_html = re.findall(r'<TR valign="top"[^>]*onclick="window\.location.*?</TR>', texto_html, re.S)
    resultado = []
    for fila in filas_html:
        celdas = re.findall(r"<TD[^>]*>(.*?)</TD>", fila, re.S)
        if len(celdas) < 7:
            continue
        prot_raw = re.sub(r"<[^>]+>", "", celdas[3]).strip()
        if not re.match(r"^p\.[A-Za-z]{1,3}\d+[A-Za-z]{1,3}(\*)?$", prot_raw):
            continue
        try:
            pp2 = float(re.sub(r"<[^>]+>", "", celdas[4]).strip())
            mapp = float(re.sub(r"<[^>]+>", "", celdas[5]).strip())
            prior = float(re.sub(r"<[^>]+>", "", celdas[6]).strip())
        except ValueError:
            continue
        dna_raw = re.sub(r"<[^>]+>", "", celdas[1]).strip()
        resultado.append({"dna_change": dna_raw, "protein_change": prot_raw,
                           "pp2_score": pp2, "mapp_score": mapp, "prior_p": prior})
    return resultado


def descargar_completo(select_db, n_paginas_estimadas, limit=100):
    todas = {}
    cabecera_verificada = False
    pagina = 1
    paginas_vacias_seguidas = 0
    while True:
        html_pagina = fetch(select_db, pagina, limit)
        if not cabecera_verificada:
            verificar_cabecera(html_pagina)
            cabecera_verificada = True
            print(f"  Cabecera de columnas verificada OK contra el orden esperado.")
        filas = parse_independiente(html_pagina)
        nuevas = 0
        for f in filas:
            key = (f["dna_change"], f["protein_change"])
            if key not in todas:
                nuevas += 1
            todas[key] = f
        print(f"  {select_db} pagina {pagina}: {len(filas)} filas, {nuevas} nuevas "
              f"(acumulado {len(todas)})", flush=True)
        if len(filas) == 0:
            paginas_vacias_seguidas += 1
            if paginas_vacias_seguidas >= 2:
                print(f"  {select_db}: 2 paginas vacias seguidas, fin de paginacion")
                break
        else:
            paginas_vacias_seguidas = 0
        pagina += 1
        if pagina > n_paginas_estimadas + 5:
            print(f"  {select_db}: limite de seguridad de paginas alcanzado, parando")
            break
        time.sleep(0.3)
    return todas


def comparar_con_original(nuevo, path_original):
    with open(path_original) as f:
        original_lista = json.load(f)
    original = {(r["dna_change"], r["protein_change"]): r for r in original_lista}

    todas_claves = set(nuevo) | set(original)
    solo_nuevo = set(nuevo) - set(original)
    solo_original = set(original) - set(nuevo)
    comunes = set(nuevo) & set(original)
    prior_distinto = [k for k in comunes if abs(nuevo[k]["prior_p"] - original[k]["prior_p"]) > 1e-9]

    print(f"  Union de claves: {len(todas_claves)}")
    print(f"  Solo en el re-scrapeo (nuevas, no estaban en el JSON original): {len(solo_nuevo)}")
    print(f"  Solo en el JSON original (perdidas en el re-scrapeo): {len(solo_original)}")
    print(f"  Comunes: {len(comunes)}, con prior_p distinto: {len(prior_distinto)}")
    if solo_nuevo:
        print(f"    Ejemplos solo-nuevo: {list(solo_nuevo)[:5]}")
    if solo_original:
        print(f"    Ejemplos solo-original: {list(solo_original)[:5]}")
    if prior_distinto:
        print(f"    Ejemplos prior distinto: {[(k, original[k]['prior_p'], nuevo[k]['prior_p']) for k in prior_distinto[:5]]}")

    return {
        "n_union": len(todas_claves), "n_solo_nuevo": len(solo_nuevo),
        "n_solo_original": len(solo_original), "n_comunes": len(comunes),
        "n_prior_distinto": len(prior_distinto),
        "claves_solo_nuevo": [list(k) for k in solo_nuevo],
        "claves_solo_original": [list(k) for k in solo_original],
    }


def main():
    resultado = {}
    for select_db, n_paginas, path_original in [
        ("PMS2_priors", 200, "/home/jesus/paper_msh6/datos/PMS2_priors_hci_lovd.json"),
    ]:
        print(f"\n{'='*70}\nRe-scrapeo completo: {select_db}\n{'='*70}")
        nuevo = descargar_completo(select_db, n_paginas)
        print(f"\nTotal descargado (parser independiente, cabecera verificada): {len(nuevo)}")
        print(f"\nComparacion contra {path_original} (union de claves, no interseccion):")
        comp = comparar_con_original(nuevo, path_original)
        resultado[select_db] = {"n_nuevo": len(nuevo), "comparacion": comp}

        # Guardar el dataset re-scrapeado como version verificada (con backup del original)
        path_nuevo = path_original.replace(".json", "_v2_verificado.json")
        with open(path_nuevo, "w") as f:
            json.dump(list(nuevo.values()), f, indent=2)
        print(f"Guardado: {path_nuevo}")

    with open("/home/jesus/paper_msh6/datos/resultado_rescrapeo_verificado.json", "w") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    print("\n\nGuardado: datos/resultado_rescrapeo_verificado.json")


if __name__ == "__main__":
    main()
