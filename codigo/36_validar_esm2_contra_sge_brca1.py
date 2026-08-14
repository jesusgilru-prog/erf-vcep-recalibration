"""
Comparacion decisiva pedida por Claude en espiritu (arbitro independiente, no
concordancia entre metodos afines): ESM-2 zero-shot vs verdad funcional REAL
(SGE, Findlay 2018) para BRCA1, a escala completa (n~2.086 missense). A
diferencia de todo lo hecho hasta ahora con MAPP/PP2 (que es el mismo tipo de
metodo que ESM-2), el SGE mide el efecto funcional directo -- es la primera
comprobacion de todo el proyecto de ESM-2 contra una verdad no basada en
alineamiento/evolucion.
"""
import json
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import numpy as np


def cargar_esm():
    with open("/home/jesus/paper_msh6/datos/esm2_zeroshot/BRCA1_esm2_650M_zeroshot.json") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        pos_str, aa = k.rsplit("_", 1)
        out[(int(pos_str), aa)] = -v  # alto = mas danino, misma convencion del proyecto
    return out


def main():
    with open("/home/jesus/paper_msh6/datos/dataset_BRCA1_SGE.json") as f:
        sge = json.load(f)
    esm = cargar_esm()

    y_sge, y_esm = [], []
    for d in sge:
        key = (d["posicion"], d["mut_aa"])
        if key not in esm:
            continue
        y_sge.append(d["score_danino"])  # ya orientado alto=danino en 33_
        y_esm.append(esm[key])
    y_sge, y_esm = np.array(y_sge), np.array(y_esm)
    print(f"Variantes BRCA1 con SGE real Y ESM-2: {len(y_sge)}")

    rho, p = spearmanr(y_sge, y_esm)
    print(f"Spearman ESM-2 vs SGE funcional real: rho={rho:.4f} (p={p:.3g})")

    # AUC: usando terciles de SGE como proxy patogenico/benigno (SGE continuo, sin
    # umbral clinico oficial publico) -- variantes en el tercil mas danino vs menos danino
    p33, p67 = np.percentile(y_sge, [33, 67])
    extremos_mask = (y_sge <= p33) | (y_sge >= p67)
    y_bin = (y_sge[extremos_mask] >= p67).astype(int)
    y_esm_ext = y_esm[extremos_mask]
    auc = roc_auc_score(y_bin, y_esm_ext)
    print(f"AUC ESM-2 discriminando tercil mas danino vs menos danino de SGE: {auc:.4f} "
          f"(n={len(y_bin)})")

    print(f"\nComparar con el resto del proyecto: rho ESM-2 vs prior oficial MAPP/PP2 "
          f"(mismo tipo de metodo) MSH6=0,803 PMS2=0,739. Este rho (SGE, verdad funcional "
          f"real e independiente) = {rho:.4f}.")

    with open("/home/jesus/paper_msh6/datos/resultado_validacion_esm2_sge_brca1.json", "w") as f:
        json.dump({"n": len(y_sge), "rho": float(rho), "p": float(p),
                    "auc_terciles": float(auc), "n_auc": int(len(y_bin))}, f, indent=2)
    print("\nGuardado: datos/resultado_validacion_esm2_sge_brca1.json")


if __name__ == "__main__":
    main()
