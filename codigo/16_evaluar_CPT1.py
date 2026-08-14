"""
Evalua CPT-1 (Jagota, Ye et al. 2023, Genome Biology, 10.1186/s13059-023-03024-6)
-- un metodo de transferencia cruzada entre proteinas ya publicado, validado y
mantenido (repo actualizado jun-2026) -- contra el mismo conjunto de validacion
externa congelado (inCAMA/CIMRA) que se uso para todo lo demas en este proyecto.

No es nuestro metodo: es la pregunta de si ya existe algo mejor que lo que hemos
construido, antes de seguir intentando construirlo desde cero. CPT-1 combina EVE +
ESM-1v + alineamientos de vertebrados + estructura AlphaFold (incluyendo
ProteinMPNN) en un framework disenado especificamente para generalizar a proteinas
sin datos DMS -- justo nuestro problema.
"""
import gzip
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


def f_uni(s):
    return float(s.replace("−", "-").strip())


def cargar_cpt(gene):
    path = f"/home/jesus/paper_msh6/datos/cpt/{gene}_HUMAN.csv.gz"
    out = {}
    with gzip.open(path, "rt") as f:
        next(f)
        for line in f:
            mutant, score = line.strip().split(",")
            wt, pos, mut = mutant[0], int(mutant[1:-1]), mutant[-1]
            out[(pos, mut)] = float(score)
    return out


def boot_ci(y, s, n=2000, seed=20260805):
    rng = np.random.default_rng(seed)
    n_ = len(y)
    out = []
    for _ in range(n):
        idx = rng.integers(0, n_, n_)
        r, _ = spearmanr(np.array(y)[idx], np.array(s)[idx])
        if not np.isnan(r):
            out.append(r)
    return np.percentile(out, [2.5, 97.5])


def auroc(y_bin, score):
    from itertools import product
    pos = [s for s, b in zip(score, y_bin) if b]
    neg = [s for s, b in zip(score, y_bin) if not b]
    if not pos or not neg:
        return None
    w = sum((p > n) + 0.5 * (p == n) for p, n in product(pos, neg))
    return w / (len(pos) * len(neg))


def main():
    with open("/home/jesus/paper_msh6/datos/CONJUNTO_VALIDACION_EXTERNA_CONGELADO.json") as f:
        frozen = json.load(f)

    resultado = {}
    for gene, key, oddspath_field in [("MSH6", "msh6", "oddspath_functional_inCAMA"),
                                        ("PMS2", "pms2", "oddspath_CIMRA")]:
        cpt = cargar_cpt(gene)
        ys, ss = [], []
        omitidas = 0
        for e in frozen[key]:
            if key == "msh6":
                m = re.match(r"^([A-Z])(\d+)([A-Z])$", e["variant_1letter"].rstrip("g"))
                pos, mut = int(m.group(2)), m.group(3)
            else:
                m = re.match(r"^[Pp]\.\s*([A-Za-z]{3})(\d+)([A-Za-z]{3})$", e["variant_protein"].strip())
                pos, mut = int(m.group(2)), AA3_TO_1[m.group(3).capitalize()]
            score = cpt.get((pos, mut))
            if score is None:
                omitidas += 1
                continue
            y = np.log10(f_uni(e[oddspath_field]))
            ys.append(y)
            ss.append(score)  # CPT1_score: mayor = mas patogenico (convencion del paper)

        ys, ss = np.array(ys), np.array(ss)
        rho, p = spearmanr(ys, ss)
        ci = boot_ci(ys, ss)
        a = auroc([y > 0 for y in ys], ss)
        print(f"{gene}: n={len(ys)} (omitidas {omitidas}), rho={rho:+.4f} (p={p:.4g}), "
              f"IC95=[{ci[0]:+.4f},{ci[1]:+.4f}], AUROC={a:.4f}")
        resultado[gene] = {"n": len(ys), "omitidas": omitidas, "rho": float(rho), "p": float(p),
                            "ci95": [float(ci[0]), float(ci[1])], "auroc": float(a)}

    with open("/home/jesus/paper_msh6/datos/resultado_CPT1.json", "w") as f:
        json.dump(resultado, f, indent=2)
    print("\nGuardado: datos/resultado_CPT1.json")
    print("\nComparar: ESM-2 650M solo -> MSH6 rho=0.770 AUROC=0.911; PMS2 rho=0.767 AUROC=1.000")
    print("          ESM-2 3B solo    -> MSH6 rho=0.560 AUROC=0.878; PMS2 rho=0.820 AUROC=0.998")


if __name__ == "__main__":
    main()
