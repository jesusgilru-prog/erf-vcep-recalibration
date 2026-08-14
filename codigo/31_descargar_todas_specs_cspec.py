"""
Descarga las especificaciones completas de las 122 VCEP "Released" de
ClinGen/cspec.genome.network (listado guardado en
datos/fuentes_primarias/cspec_all/listado_svis_20260811.json, SHA256
0e4bd499...). Un JSON por especificacion, con fecha en el nombre -- fuente
primaria citable, no un resumen de WebFetch (leccion del debate del 10-ago).
"""
import json
import time
import urllib.request

FECHA = "20260811"
DIR_OUT = "/home/jesus/paper_msh6/datos/fuentes_primarias/cspec_all"


def fetch(gn_id):
    url = f"https://cspec.genome.network/cspec/api/SequenceVariantInterpretation/id/{gn_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def main():
    with open(f"{DIR_OUT}/listado_svis_{FECHA}.json") as f:
        listado = json.load(f)
    liberadas = [x for x in listado["data"] if x.get("status") == "Released"]
    ids = [x["@id"].rsplit("/", 1)[-1] for x in liberadas]
    print(f"{len(ids)} especificaciones Released a descargar")

    ok, fallos = 0, []
    for i, gn_id in enumerate(ids):
        out_path = f"{DIR_OUT}/{gn_id}_{FECHA}.json"
        try:
            raw = fetch(gn_id)
            with open(out_path, "wb") as f:
                f.write(raw)
            ok += 1
        except Exception as e:
            fallos.append((gn_id, str(e)))
            print(f"  FALLO {gn_id}: {e}")
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(ids)} descargadas...")
        time.sleep(0.2)

    print(f"\nDescargadas OK: {ok}/{len(ids)}. Fallos: {len(fallos)}")
    if fallos:
        print(fallos)


if __name__ == "__main__":
    main()
