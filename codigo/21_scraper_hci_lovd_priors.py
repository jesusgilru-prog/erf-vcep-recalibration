"""
Descarga completa de las bases de datos oficiales "MAPP/PP2 Prior P" de
hci-lovd.hci.utah.edu (LOVD v2.0, curador Bryony Thompson) para MSH6 y PMS2 --
la herramienta que la especificacion vigente de ClinGen/InSiGHT (GN138, GN139,
marzo 2026) cita como la fuente oficial de evidencia PP3/BP4 computacional para
estos genes, verificada el 10-ago-2026 (ver PREREGISTRO.md).

hci-priors.hci.utah.edu (dominio separado citado en la especificacion para
MSH6) esta caido ("no route to host"), pero LOVD lo sirve en este mismo
portal bajo select_db=MSH6_priors -- mismos datos, dominio distinto.

Columnas reales (verificadas contra la leyenda de la pagina):
  Custom PP2.1 score | MAPP score | MAPP/PP2 Prior P
La tercera es el valor citado en la especificacion oficial con los umbrales
0.11 / 0.68 / 0.81.
"""
import re
import time
import html
import json
import urllib.request

BASE = "http://hci-lovd.hci.utah.edu/variants.php"


def fetch(select_db, page, limit=100, max_reintentos=3):
    url = f"{BASE}?action=search_unique&select_db={select_db}&limit={limit}&page={page}"
    for intento in range(max_reintentos):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            if intento == max_reintentos - 1:
                raise
            time.sleep(2)


def parse_pagina(texto_html):
    """Extrae (dna_change, protein_change, pp2_score, mapp_score, prior_p) por fila.
    Estructura verificada: cada variante tiene un DNA change (c.NNN...) seguido,
    tras varios campos, del cambio de proteina (p.XNNNY) y despues los 3 scores
    numericos en orden PP2.1, MAPP, Prior_P."""
    text = re.sub(r"<[^>]+>", "|", texto_html)
    text = re.sub(r"\|+", "|", text)
    text = html.unescape(text)
    tokens = [t.strip() for t in text.split("|") if t.strip()]

    filas = []
    i = 0
    while i < len(tokens):
        if re.match(r"^c\.[0-9_+\-*]+[ACGT>a-z0-9]*$", tokens[i]) or re.match(r"^c\.\d", tokens[i]):
            dna = tokens[i]
            # buscar el siguiente p.XNNNY dentro de una ventana corta
            ventana = tokens[i:i+8]
            prot = None
            for t in ventana:
                if re.match(r"^p\.[A-Za-z]{1,3}\d+[A-Za-z]{1,3}(\*)?$", t):
                    prot = t
                    break
            if prot is not None:
                idx_prot = tokens.index(prot, i)
                nums = []
                j = idx_prot + 1
                while len(nums) < 3 and j < len(tokens) and j < idx_prot + 6:
                    if re.match(r"^-?\d+\.?\d*$", tokens[j]):
                        nums.append(float(tokens[j]))
                    j += 1
                if len(nums) == 3:
                    filas.append({
                        "dna_change": dna, "protein_change": prot,
                        "pp2_score": nums[0], "mapp_score": nums[1], "prior_p": nums[2],
                    })
        i += 1
    return filas


def descargar_gen(select_db, n_paginas_estimadas, limit=100):
    todas = []
    vistos = set()
    for page in range(1, n_paginas_estimadas + 2):
        html_pagina = fetch(select_db, page, limit)
        filas = parse_pagina(html_pagina)
        nuevas = 0
        for f in filas:
            key = (f["dna_change"], f["protein_change"])
            if key in vistos:
                continue
            vistos.add(key)
            todas.append(f)
            nuevas += 1
        print(f"  {select_db} pagina {page}: {len(filas)} filas parseadas, {nuevas} nuevas "
              f"(total acumulado {len(todas)})", flush=True)
        if nuevas == 0 and page > 1:
            print(f"  {select_db}: sin filas nuevas, se asume fin de la paginacion")
            break
        time.sleep(0.3)  # cortesia con el servidor
    return todas


def main():
    for select_db, n_paginas in [("MSH6_priors", 92), ("PMS2_priors", 30)]:
        print(f"\n=== Descargando {select_db} ===")
        datos = descargar_gen(select_db, n_paginas)
        print(f"{select_db}: {len(datos)} variantes unicas descargadas")
        out_path = f"/home/jesus/paper_msh6/datos/{select_db}_hci_lovd.json"
        with open(out_path, "w") as f:
            json.dump(datos, f, indent=2)
        print(f"Guardado: {out_path}")


if __name__ == "__main__":
    main()
