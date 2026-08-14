"""
Responde al bloqueante mas grave señalado por la revision externa (Claude, ronda
veredicto final Q1): rho=0.80/0.73 entre ESM-2 y el prior oficial es CONCORDANCIA
entre dos modelos, no ACIERTO frente a un desenlace real. Dos predictores
correlacionados pueden equivocarse en direcciones distintas justo donde importa
(la discordancia decide reclasificaciones).

Este script compara, de forma PAREADA (mismas variantes, mismo n, misma verdad),
el AUC de ESM-2 zero-shot vs el AUC del prior oficial MAPP/PP2 (hci-lovd) frente
a las etiquetas de ClinVar de alta confianza (Pathogenic/Likely pathogenic vs
Benign/Likely benign; se excluye VUS y clasificaciones conflictivas). Incluye:
  - AUC de cada modelo sobre el MISMO conjunto pareado.
  - Test de DeLong pareado (correlacionado) para la diferencia de AUC.
  - IC bootstrap (2000 iter, resampleo de variantes) de la diferencia de AUC.
  - Analisis de discordancia: variantes donde ambos modelos, usando su propio
    umbral de Youden calculado EN ESTE MISMO conjunto (exploratorio, no un
    corte validado externamente -- se declara asi), caen en lados opuestos.

Limitacion declarada (circularidad): una fraccion de las etiquetas P/LP y B/LB
de ClinVar para MSH6/PMS2 proviene de las clasificaciones del panel de expertos
InSiGHT, cuyo modelo multifactorial usa como uno de sus componentes el propio
prior MAPP/PP2. En la comparacion cabeza a cabeza, el prior oficial puede jugar
parcialmente con etiquetas que el mismo ayudo a generar; ESM-2 no tiene ese
sesgo. Se reporta por separado el subconjunto NO revisado por panel de expertos
(review_status distinto de "reviewed by expert panel") como sensibilidad.
"""
import csv
import json
import re
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

csv.field_size_limit(sys.maxsize)

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}

PATOGENICAS = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
BENIGNAS = {"Benign", "Likely benign", "Benign/Likely benign"}

N_BOOTSTRAP = 2000
SEMILLA = 20260805


def extraer_missense(name):
    m = re.search(r"\(p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})\)", name)
    if not m:
        return None
    aa1_3, pos, aa2_3 = m.groups()
    aa1, aa2 = AA3_TO_1.get(aa1_3.capitalize()), AA3_TO_1.get(aa2_3.capitalize())
    if aa1 is None or aa2 is None or aa1 == aa2:
        return None
    return int(pos), aa1, aa2


def parse_protein_change(pc):
    m = re.match(r"^p\.([A-Za-z]{1,3})(\d+)([A-Za-z]{1,3})$", pc)
    if not m:
        return None
    aa1, pos, aa2 = m.groups()
    if len(aa1) == 3:
        aa1 = AA3_TO_1.get(aa1.capitalize())
    if len(aa2) == 3:
        aa2 = AA3_TO_1.get(aa2.capitalize())
    if aa1 is None or aa2 is None or len(aa1) != 1 or len(aa2) != 1:
        return None
    return int(pos), aa1, aa2


def cargar_esm(gene):
    path = f"/home/jesus/paper_msh6/datos/esm2_zeroshot/{gene}_esm2_650M_zeroshot.json"
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        pos_str, aa = k.rsplit("_", 1)
        out[(int(pos_str), aa)] = -v
    return out


def cargar_oficial(gene, select_db):
    with open(f"/home/jesus/paper_msh6/datos/{select_db}_hci_lovd.json") as f:
        raw = json.load(f)
    out = {}
    for r in raw:
        parsed = parse_protein_change(r["protein_change"])
        if parsed is None:
            continue
        pos, wt_aa, mut_aa = parsed
        out[(pos, mut_aa)] = r["prior_p"]
    return out


def cargar_filas_gen(gene):
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


def delong_test_pareado(y_true, score_a, score_b):
    """Test de DeLong pareado (correlacionado) para diferencia de AUC.
    Implementacion 'fast DeLong' (Sun & Xu 2014) via structural components."""
    y_true = np.asarray(y_true)
    order = np.argsort(-y_true, kind="stable")
    y_true = y_true[order]
    scores = np.vstack([score_a[order], score_b[order]])
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    def midrank(x):
        j = np.argsort(x, kind="stable")
        z = x[j]
        n = len(x)
        t = np.zeros(n, dtype=float)
        i = 0
        while i < n:
            k = i
            while k < n and z[k] == z[i]:
                k += 1
            t[i:k] = 0.5 * (i + k - 1) + 1
            i = k
        r = np.empty(n, dtype=float)
        r[j] = t
        return r

    k = scores.shape[0]
    tx = np.empty([k, n_pos])
    ty = np.empty([k, n_neg])
    tz = np.empty([k, n_pos + n_neg])
    for r in range(k):
        tx[r, :] = midrank(scores[r, :n_pos])
        ty[r, :] = midrank(scores[r, n_pos:])
        tz[r, :] = midrank(scores[r, :])
    aucs = tz[:, :n_pos].sum(axis=1) / (n_pos * n_neg) - (n_pos + 1.0) / (2.0 * n_neg)

    v01 = (tz[:, :n_pos] - tx) / n_neg
    v10 = 1.0 - (tz[:, n_pos:] - ty) / n_pos
    sx = np.cov(v01)
    sy = np.cov(v10)
    s = sx / n_pos + sy / n_neg
    var_diff = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    diff = aucs[0] - aucs[1]
    if var_diff <= 0:
        return float(aucs[0]), float(aucs[1]), float(diff), None, None
    z = diff / np.sqrt(var_diff)
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(diff), float(z), float(p)


def procesar_gen(gene, select_db, rng):
    print(f"\n{'='*70}\n{gene}: AUC PAREADO ESM-2 vs prior oficial (verdad = ClinVar P/LP vs B/LB)\n{'='*70}")
    esm = cargar_esm(gene)
    oficial = cargar_oficial(gene, select_db)
    filas = cargar_filas_gen(gene)

    etiquetados = {}
    revisado_panel = {}
    for row in filas:
        sig = row["ClinicalSignificance"]
        if sig in PATOGENICAS:
            label = 1
        elif sig in BENIGNAS:
            label = 0
        else:
            continue
        parsed = extraer_missense(row["Name"])
        if parsed is None:
            continue
        pos, wt_aa, mut_aa = parsed
        etiquetados[(pos, mut_aa)] = label
        revisado_panel[(pos, mut_aa)] = (row["ReviewStatus"] == "reviewed by expert panel")

    comunes = [k for k in etiquetados if k in esm and k in oficial]
    print(f"Variantes P/LP + B/LB en ClinVar con score ESM-2 Y prior oficial simultaneos: {len(comunes)}")
    y = np.array([etiquetados[k] for k in comunes])
    s_esm = np.array([esm[k] for k in comunes])
    s_of = np.array([oficial[k] for k in comunes])
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    print(f"  Patogenicas/Prob.patogenicas: {n_pos} | Benignas/Prob.benignas: {n_neg}")

    if n_pos < 5 or n_neg < 5:
        print("  INSUFICIENTE para AUC fiable (menos de 5 por clase). Se omite.")
        return {"gene": gene, "n": len(comunes), "n_pos": n_pos, "n_neg": n_neg, "insuficiente": True}

    auc_esm = roc_auc_score(y, s_esm)
    auc_of = roc_auc_score(y, s_of)
    print(f"  AUC ESM-2 zero-shot:      {auc_esm:.4f}")
    print(f"  AUC prior oficial MAPP/PP2: {auc_of:.4f}")

    auc_of_dl, auc_esm_dl, diff, z, p = delong_test_pareado(y, s_of, s_esm)
    print(f"  Diferencia (oficial - ESM-2): {diff:+.4f}  (DeLong z={z}, p={p})")

    diffs_boot = []
    idx = np.arange(len(comunes))
    for _ in range(N_BOOTSTRAP):
        bi = rng.choice(idx, size=len(idx), replace=True)
        yb = y[bi]
        if yb.sum() < 2 or (1 - yb).sum() < 2:
            continue
        try:
            a_of = roc_auc_score(yb, s_of[bi])
            a_esm = roc_auc_score(yb, s_esm[bi])
            diffs_boot.append(a_of - a_esm)
        except ValueError:
            continue
    diffs_boot = np.array(diffs_boot)
    ci_lo, ci_hi = np.percentile(diffs_boot, [2.5, 97.5])
    print(f"  IC bootstrap 95% de la diferencia (oficial - ESM-2): [{ci_lo:+.4f}, {ci_hi:+.4f}] (n_boot={len(diffs_boot)})")

    # --- discordancia: umbral de Youden de cada modelo EN ESTE MISMO conjunto (exploratorio) ---
    fpr_of, tpr_of, thr_of = roc_curve(y, s_of)
    thr_youden_of = thr_of[np.argmax(tpr_of - fpr_of)]
    fpr_e, tpr_e, thr_e = roc_curve(y, s_esm)
    thr_youden_esm = thr_e[np.argmax(tpr_e - fpr_e)]
    call_of = (s_of >= thr_youden_of).astype(int)
    call_esm = (s_esm >= thr_youden_esm).astype(int)
    discordantes = call_of != call_esm
    n_disc = int(discordantes.sum())
    print(f"  Discordancia (umbral de Youden de cada modelo, exploratorio): {n_disc}/{len(comunes)} ({100*n_disc/len(comunes):.1f}%)")
    acierto_of_en_disc = (call_of[discordantes] == y[discordantes]).mean() if n_disc else None
    acierto_esm_en_disc = (call_esm[discordantes] == y[discordantes]).mean() if n_disc else None
    print(f"    De las discordantes, acierta oficial: {acierto_of_en_disc}, acierta ESM-2: {acierto_esm_en_disc}")

    # --- sensibilidad: excluyendo lo ya revisado por panel de expertos (mitiga circularidad InSiGHT/MAPP-PP2) ---
    no_panel = np.array([not revisado_panel[k] for k in comunes])
    resultado_no_panel = None
    if no_panel.sum() >= 10 and y[no_panel].sum() >= 5 and (1 - y[no_panel]).sum() >= 5:
        auc_esm_np = roc_auc_score(y[no_panel], s_esm[no_panel])
        auc_of_np = roc_auc_score(y[no_panel], s_of[no_panel])
        print(f"  [Sensibilidad, excluyendo 'reviewed by expert panel', n={int(no_panel.sum())}]: "
              f"AUC ESM-2={auc_esm_np:.4f}, AUC oficial={auc_of_np:.4f}")
        resultado_no_panel = {"n": int(no_panel.sum()), "auc_esm": float(auc_esm_np), "auc_oficial": float(auc_of_np)}
    else:
        print("  [Sensibilidad sin panel de expertos: insuficiente n]")

    return {
        "gene": gene, "n": len(comunes), "n_pos": n_pos, "n_neg": n_neg,
        "auc_esm": float(auc_esm), "auc_oficial": float(auc_of),
        "diferencia_oficial_menos_esm": float(diff),
        "delong_z": z, "delong_p": p,
        "bootstrap_ci95_diferencia": [float(ci_lo), float(ci_hi)], "n_bootstrap_validos": len(diffs_boot),
        "discordancia": {
            "n_discordantes": n_disc, "pct_discordantes": 100 * n_disc / len(comunes),
            "umbral_youden_oficial": float(thr_youden_of), "umbral_youden_esm": float(thr_youden_esm),
            "acierto_oficial_en_discordantes": acierto_of_en_disc,
            "acierto_esm_en_discordantes": acierto_esm_en_disc,
        },
        "sensibilidad_excluyendo_panel_expertos": resultado_no_panel,
    }


def main():
    rng = np.random.default_rng(SEMILLA)
    resultados = {}
    resultados["MSH6"] = procesar_gen("MSH6", "MSH6_priors", rng)
    resultados["PMS2"] = procesar_gen("PMS2", "PMS2_priors", rng)
    with open("/home/jesus/paper_msh6/datos/resultado_auc_pareado_esm_vs_oficial.json", "w") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print("\n\nGuardado: datos/resultado_auc_pareado_esm_vs_oficial.json")


if __name__ == "__main__":
    main()
