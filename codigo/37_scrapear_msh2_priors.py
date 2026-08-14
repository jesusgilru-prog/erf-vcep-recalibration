"""
MSH2 tiene su propia especificacion VCEP (GN137) que cita la MISMA herramienta
oficial que MSH6/PMS2 (HCI-prior / MAPP-PP2, mismos umbrales 0.11/0.68/0.81) --
verificado leyendo el JSON fuente. Se descarga MSH2_priors del mismo portal
LOVD (select_db=MSH2_priors, confirmado que existe en el selector de bases de
datos del portal) para poder hacer, dentro del mismo gen y sin el cruce de gen
que invalidaba la comparacion con BRCA1 (revision 11-ago-2026), la comparacion
de tres vias: prior oficial vs ESM-2 vs verdad funcional real (el propio DMS de
MSH2 en HAP1, ya en datos/dataset_H0_MSH2.json).
"""
import importlib
import json

import sys
sys.path.insert(0, "/home/jesus/paper_msh6/codigo")
scraper = importlib.import_module("27_rescrapeo_verificado_lovd")


def main():
    select_db = "MSH2_priors"
    todas = {}
    cabecera_verificada = False
    pagina = 1
    vacias_seguidas = 0
    while True:
        html_pagina = scraper.fetch(select_db, pagina)
        if not cabecera_verificada:
            scraper.verificar_cabecera(html_pagina)
            cabecera_verificada = True
            print("Cabecera verificada OK")
        filas = scraper.parse_independiente(html_pagina)
        nuevas = 0
        for f in filas:
            key = (f["dna_change"], f["protein_change"])
            if key not in todas:
                nuevas += 1
            todas[key] = f
        print(f"  pagina {pagina}: {len(filas)} filas, {nuevas} nuevas (acumulado {len(todas)})", flush=True)
        if len(filas) == 0:
            vacias_seguidas += 1
            if vacias_seguidas >= 2:
                break
        else:
            vacias_seguidas = 0
        pagina += 1
        if pagina > 250:
            print("limite de seguridad alcanzado")
            break
        import time
        time.sleep(0.3)

    print(f"\nMSH2_priors: {len(todas)} variantes descargadas")
    with open("/home/jesus/paper_msh6/datos/MSH2_priors_hci_lovd.json", "w") as f:
        json.dump(list(todas.values()), f, indent=2)
    print("Guardado: datos/MSH2_priors_hci_lovd.json")


if __name__ == "__main__":
    main()
