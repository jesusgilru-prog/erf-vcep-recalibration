"""
Version corregida del hallazgo 2 (revision 11-ago-2026: comparar oficial-vs-ESM2
en un gen contra ESM2-vs-SGE en OTRO gen -- BRCA1 -- era un cruce de gen que
invalidaba la conclusion). Aqui las TRES piezas son del MISMO gen, MSH2:
  - prior oficial MAPP/PP2 (mismo portal y umbrales que MSH6/PMS2/GN137,
    scrapeado en 37_)
  - ESM-2 650M zero-shot (ya calculado, 06_)
  - verdad funcional REAL e independiente: el propio DMS de MSH2 (HAP1,
    complementacion, MaveDB urn:mavedb:00000050-a-1, ya en
    datos/dataset_H0_MSH2.json) -- ni alineamiento ni evolutivo, mide fitness
    celular real.
"""
import json
import re

import numpy as np
from scipy.stats import spearmanr

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}


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


def main():
    with open("/home/jesus/paper_msh6/datos/MSH2_priors_hci_lovd.json") as f:
        raw_oficial = json.load(f)
    oficial = {}
    for r in raw_oficial:
        parsed = parse_protein_change(r["protein_change"])
        if parsed is None:
            continue
        pos, wt_aa, mut_aa = parsed
        oficial[(pos, mut_aa)] = r["prior_p"]

    with open("/home/jesus/paper_msh6/datos/esm2_zeroshot/MSH2_esm2_650M_zeroshot.json") as f:
        raw_esm = json.load(f)
    esm = {}
    for k, v in raw_esm.items():
        pos_str, aa = k.rsplit("_", 1)
        esm[(int(pos_str), aa)] = -v  # alto = mas danino

    with open("/home/jesus/paper_msh6/datos/dataset_H0_MSH2.json") as f:
        dms = json.load(f)
    real = {(d["posicion"], d["mut_aa"]): d["score_danino"] for d in dms}

    comunes = sorted(set(oficial) & set(esm) & set(real))
    print(f"Variantes MSH2 con prior oficial + ESM-2 + verdad funcional real (DMS HAP1) simultaneos: "
          f"{len(comunes)}")

    y_of = np.array([oficial[k] for k in comunes])
    y_esm = np.array([esm[k] for k in comunes])
    y_real = np.array([real[k] for k in comunes])

    rho_of_esm, p1 = spearmanr(y_of, y_esm)
    rho_of_real, p2 = spearmanr(y_of, y_real)
    rho_esm_real, p3 = spearmanr(y_esm, y_real)

    print(f"\nDentro de MSH2 (mismo gen, las 3 piezas):")
    print(f"  Oficial vs ESM-2 (concordancia entre 2 metodos afines):        rho={rho_of_esm:.4f} (p={p1:.3g})")
    print(f"  Oficial vs verdad funcional real (acierto real del oficial):  rho={rho_of_real:.4f} (p={p2:.3g})")
    print(f"  ESM-2 vs verdad funcional real (acierto real de ESM-2):       rho={rho_esm_real:.4f} (p={p3:.3g})")

    print(f"\nComparar con MSH6/PMS2 (oficial vs ESM-2, mismo gen, ya calculado): rho=0,803/0,739")
    print(f"Comparar con BRCA1 (ESM-2 vs SGE real, gen DISTINTO, invalidado por cruce de gen): rho=0,4993")
    print(f"\nEsta es la comparacion limpia, sin cruce de gen: ¿el patron de MSH6/PMS2 "
          f"(concordancia alta oficial-ESM2) se sostiene tambien frente a la verdad funcional real "
          f"del MISMO gen (MSH2)?")

    # bootstrap CI de la diferencia rho_of_esm - rho_esm_real (inflacion por concordancia entre afines)
    rng = np.random.default_rng(20260811)
    n = len(comunes)
    diffs = []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        r1, _ = spearmanr(y_of[idx], y_esm[idx])
        r2, _ = spearmanr(y_esm[idx], y_real[idx])
        if not (np.isnan(r1) or np.isnan(r2)):
            diffs.append(r1 - r2)
    ci = np.percentile(diffs, [2.5, 97.5])
    print(f"\nIC bootstrap 95% de (rho_oficial_vs_ESM2 - rho_ESM2_vs_real): [{ci[0]:.4f}, {ci[1]:.4f}]")

    resultado = {
        "n": len(comunes),
        "rho_oficial_vs_esm2": float(rho_of_esm), "p_oficial_vs_esm2": float(p1),
        "rho_oficial_vs_real": float(rho_of_real), "p_oficial_vs_real": float(p2),
        "rho_esm2_vs_real": float(rho_esm_real), "p_esm2_vs_real": float(p3),
        "ic95_diferencia_concordancia_menos_acierto": [float(ci[0]), float(ci[1])],
    }
    with open("/home/jesus/paper_msh6/datos/resultado_tres_vias_msh2.json", "w") as f:
        json.dump(resultado, f, indent=2)
    print("\nGuardado: datos/resultado_tres_vias_msh2.json")


if __name__ == "__main__":
    main()
