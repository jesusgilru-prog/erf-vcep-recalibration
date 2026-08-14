"""
Ultimo intento preespecificado (E6 del panel de revision, 6-ago-2026) de rescatar
el modelo de transferencia: ¿ordena mejor que ESM-2 solo DENTRO de cada clase
(entre las patogenicas, entre las benignas), donde el zero-shot puede no tener
matiz fino? Preespecificado antes de mirar: si no gana en NINGUNA de las 4
combinaciones (gen x clase), se acepta que la transferencia no aporta nada aqui.
"""
import json

import numpy as np
from scipy.stats import spearmanr

with open("/home/jesus/paper_msh6/datos/resultado_H1_H2.json") as f:
    h1h2 = json.load(f)
with open("/home/jesus/paper_msh6/revision_debate/control_baseline_esm_zeroshot.json") as f:
    esm_solo = json.load(f)

# reconstruyo esm-2 650M solo por variante, en el mismo orden que h1h2 (reordeno por y para emparejar)
import sys
sys.path.insert(0, "/home/jesus/paper_msh6/revision_debate")
import re
AA3 = {"Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E","Gly":"G",
       "His":"H","Ile":"I","Leu":"L","Lys":"K","Met":"M","Phe":"F","Pro":"P","Ser":"S",
       "Thr":"T","Trp":"W","Tyr":"Y","Val":"V"}

def f_uni(s): return float(s.replace("−", "-").strip())

def esm_dict(gene):
    p = f"/home/jesus/paper_msh6/datos/esm2_zeroshot/{gene}_esm2_650M_zeroshot.json"
    raw = json.load(open(p))
    return {(int(k.rsplit("_",1)[0]), k.rsplit("_",1)[1]): v for k,v in raw.items()}

frozen = json.load(open("/home/jesus/paper_msh6/datos/CONJUNTO_VALIDACION_EXTERNA_CONGELADO.json"))

resultado = {}
for nombre, clave, key in [("MSH6", "H1_msh2_a_msh6", "msh6"), ("PMS2", "H2_mlh1_a_pms2", "pms2")]:
    r = h1h2[clave]
    y = np.array(r["y"])
    pred_transfer = np.array(r["pred"])

    d = esm_dict(nombre)
    esm_vals = []
    for e in frozen[key]:
        if key == "msh6":
            m = re.match(r"^([A-Z])(\d+)([A-Z])$", e["variant_1letter"].rstrip("g"))
            pos, mut = int(m.group(2)), m.group(3)
        else:
            m = re.match(r"^[Pp]\.\s*([A-Za-z]{3})(\d+)([A-Za-z]{3})$", e["variant_protein"].strip())
            pos, mut = int(m.group(2)), AA3[m.group(3).capitalize()]
        esm_vals.append(-d[(pos, mut)])
    esm_vals = np.array(esm_vals[:len(y)]) if len(esm_vals) >= len(y) else None
    # nota: el orden de 'frozen[key]' puede no coincidir 1:1 con 'y' si H1 omitio
    # variantes sin estructura (2 en MSH6) -- se recalcula por separado si hace falta.

    print(f"\n=== {nombre} (n={len(y)}) ===")
    fila = {}
    for signo, etiqueta in [(1, "patogenicas (y>0)"), (-1, "benignas (y<0)")]:
        mask = (y * signo) > 0
        n = int(mask.sum())
        if n < 4:
            print(f"  {etiqueta}: n={n}, insuficiente para correlacion")
            continue
        rho_t, p_t = spearmanr(y[mask], pred_transfer[mask])
        print(f"  {etiqueta}: n={n}, rho_transfer={rho_t:+.4f} (p={p_t:.3f})")
        fila[etiqueta] = {"n": n, "rho_transfer": float(rho_t), "p": float(p_t)}
    resultado[nombre] = fila

with open("/home/jesus/paper_msh6/datos/resultado_rescate_intraclase.json", "w") as f:
    json.dump(resultado, f, indent=2, ensure_ascii=False)
print("\nGuardado: datos/resultado_rescate_intraclase.json")
print("\nNOTA: el rho intra-clase de ESM-2-solo ya lo calculo el panel en ronda 3 "
      "(ver ACTA.md punto 5): PMS2 pat=+0.207 (p=0.27) ben=+0.079 (p=0.74); "
      "MSH6 pat=+0.588 (p=0.07) ben=+0.383 (p=0.31). Se compara contra esos numeros.")
