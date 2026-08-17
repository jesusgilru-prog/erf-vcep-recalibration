"""
Replica en TP53 (Li-Fraumeni, fuera de la familia MMR) el analisis central del
proyecto: concordancia entre predictores vs acierto real contra DMS, AUC contra
extremos clinicos ClinVar, y recalibracion LR en la region ambigua (E-RF),
usando la herramienta oficial VCEP-citada real para TP53 (BayesDel_noAF,
especificacion GN009 v2.4, verificada via cspec.genome.network) en vez de
MAPP/PP2. Requiere que ESM-2/ESM-1v de TP53 ya esten calculados (codigo/53_).

Simplificacion declarada: la especificacion GN009 exige BayesDel Y aGVGD para
fijar la fuerza PP3 (Moderate si aGVGD C65, Supporting si C25-C55); no se
calcula aGVGD aqui (requiere alineamiento multiple adicional no disponible),
asi que se trata cualquier BayesDel>=0.16 como PP3_Supporting (la fuerza minima
garantizada sin conocer la clase aGVGD) -- una simplificacion conservadora, no
una lectura literal completa de la regla combinada.
"""
import csv
import json
import re
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

csv.field_size_limit(sys.maxsize)

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}
SEMILLA_GMM = 20260813
RNG_SEED_BOOT = 20260813
N_BOOT = 10000
CORTE_SUPPORTING = 2.08
CORTE_MODERATE = 4.3


def cargar_dms():
    with open("/home/jesus/paper_msh6/datos/dataset_H0_TP53.json") as f:
        dms = json.load(f)
    return {(d["posicion"], d["mut_aa"]): d["score_danino"] for d in dms}


def cargar_esm2():
    with open("/home/jesus/paper_msh6/datos/esm2_zeroshot/TP53_esm2_650M_zeroshot.json") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        p, a = k.rsplit("_", 1)
        out[(int(p), a)] = -v
    return out


def cargar_esm1v():
    with open("/home/jesus/paper_msh6/datos/esm1v_zeroshot/TP53_esm1v_ensemble5_zeroshot.json") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        p, a = k.rsplit("_", 1)
        out[(int(p), a)] = -v
    return out


def cargar_alphamissense():
    with open("/home/jesus/paper_msh6/datos/alphamissense/TP53_alphamissense.json") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        p, a = k.rsplit("_", 1)
        out[(int(p), a)] = v["score"]
    return out


def cargar_bayesdel():
    # BayesDel_noAF, no _addAF: verificado contra la fuente primaria de la
    # calibracion TP53-especifica (Fortuno et al., PMC6043381) que el umbral
    # 0.16 de la especificacion GN009 es para BayesDel SIN frecuencia
    # alelica -- error real detectado y corregido 14-ago-2026 (se habia
    # extraido add_af por defecto), ver PREREGISTRO.md.
    with open("/home/jesus/paper_msh6/datos/tp53_extension/TP53_bayesdel_noaf.json") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        p, a = k.rsplit("_", 1)
        out[(int(p), a)] = v
    return out


def cargar_clinvar_gen(gene):
    filas = []
    with open("/home/jesus/paper_msh6/datos/variant_summary.txt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["Assembly"] != "GRCh38":
                continue
            if gene not in set(row["GeneSymbol"].split("|")):
                continue
            filas.append(row)
    return filas


ALTA_CONFIANZA = {"reviewed by expert panel", "practice guideline",
                   "criteria provided, multiple submitters, no conflicts"}


def etiquetas_clinvar_extremos(gene):
    pat, ben = {}, {}
    for row in cargar_clinvar_gen(gene):
        if row["ReviewStatus"] not in ALTA_CONFIANZA:
            continue
        m = re.search(r"\(p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})\)", row["Name"])
        if not m:
            continue
        aa1_3, pos, aa2_3 = m.groups()
        aa1, aa2 = AA3_TO_1.get(aa1_3.capitalize()), AA3_TO_1.get(aa2_3.capitalize())
        if aa1 is None or aa2 is None or aa1 == aa2:
            continue
        cs = row["ClinicalSignificance"]
        key = (int(pos), aa2)
        if cs in ("Pathogenic", "Pathogenic/Likely pathogenic", "Likely pathogenic"):
            pat[key] = 1
        elif cs in ("Benign", "Benign/Likely benign", "Likely benign"):
            ben[key] = 0
    return pat, ben


def bootstrap_ci_rho(y1, y2, seed, n_boot=2000):
    rng = np.random.default_rng(seed)
    n = len(y1)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r, _ = spearmanr(y1[idx], y2[idx])
        if not np.isnan(r):
            boots.append(r)
    return np.percentile(boots, [2.5, 97.5]).tolist()


def main():
    real = cargar_dms()
    oficial = cargar_bayesdel()  # herramienta VCEP-citada real: BayesDel_noAF
    esm2 = cargar_esm2()
    esm1v = cargar_esm1v()
    am = cargar_alphamissense()
    print(f"TP53 -- cobertura: DMS={len(real)}, BayesDel={len(oficial)}, "
          f"ESM-2={len(esm2)}, ESM-1v={len(esm1v)}, AlphaMissense={len(am)}")

    predictores = {"oficial_bayesdel": oficial, "esm2": esm2, "esm1v": esm1v, "alphamissense": am}

    # --- 1. Concordancia y acierto en el subconjunto de 4 vias + DMS ---
    comunes4 = sorted(set(real) & set(oficial) & set(esm2) & set(esm1v) & set(am))
    print(f"\nSubconjunto de 4 predictores + DMS: n={len(comunes4)}")
    resultado = {"n_total_dms": len(real), "n_matched_4way": len(comunes4), "concordancia": {}, "acierto": {}}
    y_real = np.array([real[k] for k in comunes4])
    nombres = list(predictores.keys())
    for i, n1 in enumerate(nombres):
        s1 = np.array([predictores[n1][k] for k in comunes4])
        rho_acc, p_acc = spearmanr(s1, y_real)
        ci_acc = bootstrap_ci_rho(s1, y_real, seed=RNG_SEED_BOOT + i)
        resultado["acierto"][n1] = {"rho": float(rho_acc), "p": float(p_acc), "ic95": ci_acc, "n": len(comunes4)}
        print(f"  acierto {n1} vs DMS: rho={rho_acc:.3f} {ci_acc}")
        for j, n2 in enumerate(nombres):
            if j <= i:
                continue
            s2 = np.array([predictores[n2][k] for k in comunes4])
            rho_c, p_c = spearmanr(s1, s2)
            ci_c = bootstrap_ci_rho(s1, s2, seed=RNG_SEED_BOOT + 100 + i * 10 + j)
            resultado["concordancia"][f"{n1}_vs_{n2}"] = {"rho": float(rho_c), "p": float(p_c), "ic95": ci_c}
            print(f"  concordancia {n1} vs {n2}: rho={rho_c:.3f} {ci_c}")

    # acierto en cobertura completa propia (referencia, sin bootstrap)
    resultado["acierto_full_coverage"] = {}
    for nombre, scores in predictores.items():
        comunes_full = sorted(set(real) & set(scores))
        s = np.array([scores[k] for k in comunes_full])
        y = np.array([real[k] for k in comunes_full])
        rho, p = spearmanr(s, y)
        resultado["acierto_full_coverage"][nombre] = {"rho": float(rho), "n": len(comunes_full)}
        print(f"  acierto full-coverage {nombre}: rho={rho:.3f} (n={len(comunes_full)})")

    # --- 2. AUC contra extremos ClinVar reales ---
    pat, ben = etiquetas_clinvar_extremos("TP53")
    etiq = {**pat, **ben}
    print(f"\nClinVar extremos alta confianza: {len(pat)} patogenicas, {len(ben)} benignas")
    resultado["auc_clinvar"] = {}
    for nombre, scores in predictores.items():
        comunes = [k for k in etiq if k in scores]
        y = np.array([etiq[k] for k in comunes])
        s = np.array([scores[k] for k in comunes])
        if len(set(y.tolist())) < 2:
            continue
        auc = roc_auc_score(y, s)
        resultado["auc_clinvar"][nombre] = {"auc": float(auc), "n": len(comunes),
                                             "n_pat": int(y.sum()), "n_ben": int((1 - y).sum())}
        print(f"  AUC {nombre}: {auc:.3f} (n={len(comunes)}, pat={int(y.sum())}, ben={int((1-y).sum())})")

    with open("/home/jesus/paper_msh6/datos/resultado_TP53_concordancia_acierto_auc.json", "w") as f:
        json.dump(resultado, f, indent=2)
    print("\nGuardado: datos/resultado_TP53_concordancia_acierto_auc.json")


if __name__ == "__main__":
    main()
