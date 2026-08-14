"""
Calibracion ACMG con ESM-2 650M solo -- version corregida tras la revision del
10-ago-2026 (revision_debate_2/ACTA.md), que encontro tres errores reales:

(i) CATEGORIA: un modelo de lenguaje de proteinas es evidencia COMPUTACIONAL.
    Va en PP3/BP4 (Pejaver et al. 2022, subgrupo computacional del SVI de
    ClinGen), no en PS3/BS3 (que es para ensayos funcionales in vitro/in vivo,
    Brnich et al. 2019, PMC6938631). El marco de OddsPath y sus umbrales
    (Tavtigian et al. 2018) son los mismos; solo cambia la etiqueta de
    categoria.

(ii) PRIOR: verificado contra el propio Brnich et al. 2019 (texto completo,
    PMC6938631): "We treated the proportion of pathogenic variants in the
    overall modeled data as a prior probability (P1)". El prior NO es un valor
    fijo importado de otro estudio (0.10 generico, o el 0.0441 de Pejaver, que
    se calculo sobre OTRO conjunto modelado) -- es la proporcion de
    patogenicas en el propio set de calibracion. Se calcula aqui por gen.

(iii) FUERZA DECLARADA: con calibraciones de 59 (MSH6) y 27 (PMS2) variantes,
    declarar "VeryStrong" (OddsPath>=350) a partir de un unico punto estimado
    no esta sostenido. Se reporta el LIMITE INFERIOR (lado patogenico) o
    SUPERIOR (lado benigno) del IC95 bootstrap del OddsPath, no el punto.

(iv) DEGENERACION: si la calibracion nunca predice una direccion (todo PP3,
    cero BP4, o viceversa), se detecta y se declara explicitamente en vez de
    reportar una "tasa de acierto" que en realidad es la prevalencia del test.
"""
import csv
import json
import re
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm

csv.field_size_limit(sys.maxsize)

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}
ALTA_CONFIANZA = {"reviewed by expert panel", "practice guideline",
                   "criteria provided, multiple submitters, no conflicts"}
UMBRALES_PATOGENICO = [("VeryStrong", 350), ("Strong", 18.7), ("Moderate", 4.3), ("Supporting", 2.08)]
UMBRALES_BENIGNO = [("VeryStrong", 1/350), ("Strong", 1/18.7), ("Moderate", 1/4.3), ("Supporting", 1/2.08)]
UMBRAL_BS1_AF = 0.001
RNG_SEED = 20260810


def evidencia_acmg(oddspath):
    """Etiqueta PP3 (patogenico) / BP4 (benigno) -- evidencia computacional,
    no PS3/BS3 (funcional). Mismos umbrales de Tavtigian 2018."""
    if oddspath >= 1:
        for nombre, umbral in UMBRALES_PATOGENICO:
            if oddspath >= umbral:
                return f"PP3_{nombre}", "patogenico"
        return "sin_evidencia", "patogenico_debil"
    else:
        for nombre, umbral in UMBRALES_BENIGNO:
            if oddspath <= umbral:
                return f"BP4_{nombre}", "benigno"
        return "sin_evidencia", "benigno_debil"


def cargar_esm(gene):
    path = f"/home/jesus/paper_msh6/datos/esm2_zeroshot/{gene}_esm2_650M_zeroshot.json"
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        pos_str, aa = k.rsplit("_", 1)
        out[(int(pos_str), aa)] = -v  # invertido: alto = danino
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


def extraer_missense_pos(name):
    m = re.search(r"\(p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})\)", name)
    if not m:
        return None
    aa1_3, pos, aa2_3 = m.groups()
    aa1, aa2 = AA3_TO_1.get(aa1_3.capitalize()), AA3_TO_1.get(aa2_3.capitalize())
    if aa1 is None or aa2 is None or aa1 == aa2:
        return None
    return int(pos), aa1, aa2


def construir_calibracion(gene, excluir):
    pat, ben = [], []
    for row in cargar_clinvar_gen(gene):
        if row["ReviewStatus"] not in ALTA_CONFIANZA:
            continue
        parsed = extraer_missense_pos(row["Name"])
        if parsed is None:
            continue
        pos, wt_aa, mut_aa = parsed
        if (pos, mut_aa) in excluir:
            continue
        cs = row["ClinicalSignificance"]
        if cs in ("Pathogenic", "Pathogenic/Likely pathogenic", "Likely pathogenic"):
            pat.append((pos, mut_aa))
        elif cs in ("Benign", "Benign/Likely benign", "Likely benign"):
            ben.append((pos, mut_aa))
    return pat, ben


def construir_benignas_gnomad_bs1(gene, excluir):
    with open(f"/home/jesus/paper_msh6/datos/gnomad_{gene}.json") as f:
        variants = json.load(f)
    ben = []
    for v in variants:
        if v["consequence"] != "missense_variant":
            continue
        af = None
        if v.get("exome") and v["exome"].get("af") is not None:
            af = v["exome"]["af"]
        elif v.get("genome") and v["genome"].get("af") is not None:
            af = v["genome"]["af"]
        if af is None or af <= UMBRAL_BS1_AF:
            continue
        hgvsp = v.get("hgvsp")
        if not hgvsp:
            continue
        m = re.match(r"^p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})$", hgvsp)
        if not m:
            continue
        aa1_3, pos, aa2_3 = m.groups()
        wt_aa, mut_aa = AA3_TO_1.get(aa1_3.capitalize()), AA3_TO_1.get(aa2_3.capitalize())
        if wt_aa is None or mut_aa is None or (int(pos), mut_aa) in excluir:
            continue
        ben.append((int(pos), mut_aa))
    return ben


def f_uni(s):
    return float(s.replace("−", "-").strip())


def wilson_ci(aciertos, total, z=1.96):
    if total == 0:
        return (float("nan"), float("nan"))
    p = aciertos / total
    denom = 1 + z**2 / total
    centro = (p + z**2 / (2 * total)) / denom
    margen = (z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))) / denom
    return (max(0.0, centro - margen), min(1.0, centro + margen))


def ajustar_calibracion(X_cal, y_cal):
    """Devuelve (logreg, prior_odds) con el prior de Brnich 2019: proporcion
    de patogenicas en el propio set de calibracion."""
    logreg = LogisticRegression()
    logreg.fit(X_cal, y_cal)
    prior_p = float(np.mean(y_cal))  # proporcion de patogenicas en el modeled data
    prior_odds = prior_p / (1 - prior_p)
    return logreg, prior_p, prior_odds


def score_a_oddspath_bootstrap(logreg_orig, X_cal, y_cal, score, prior_odds_orig,
                                 n_boot=2000, seed=RNG_SEED):
    """OddsPath puntual (con el prior/logreg originales) + IC95 bootstrap
    reajustando la logistica y el prior en cada remuestreo del set de
    calibracion (para que el IC refleje la incertidumbre de calibrar con
    pocos puntos, no solo la del score en si)."""
    p = logreg_orig.predict_proba([[score]])[0, 1]
    p = min(max(p, 1e-6), 1 - 1e-6)
    oddspath_puntual = (p / (1 - p)) / prior_odds_orig

    rng = np.random.default_rng(seed)
    n = len(y_cal)
    oddspaths_boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        Xb, yb = X_cal[idx], y_cal[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            lr_b = LogisticRegression()
            lr_b.fit(Xb, yb)
            prior_p_b = float(np.mean(yb))
            if prior_p_b <= 0 or prior_p_b >= 1:
                continue
            prior_odds_b = prior_p_b / (1 - prior_p_b)
            pb = lr_b.predict_proba([[score]])[0, 1]
            pb = min(max(pb, 1e-6), 1 - 1e-6)
            oddspaths_boot.append((pb / (1 - pb)) / prior_odds_b)
        except Exception:
            continue
    if len(oddspaths_boot) < 100:
        return oddspath_puntual, None, None
    lo, hi = np.percentile(oddspaths_boot, [2.5, 97.5])
    return oddspath_puntual, float(lo), float(hi)


def evidencia_conservadora(oddspath_puntual, ci_lo, ci_hi):
    """Fuerza declarable = la que sostiene el EXTREMO MENOS FAVORABLE del IC95,
    no el punto. Lado patogenico: usar ci_lo (el mas bajo, mas conservador).
    Lado benigno: usar ci_hi (el mas alto, mas conservador). Si el IC no se
    pudo calcular, se declara 'sin_IC' explicitamente -- no se inventa fuerza."""
    if ci_lo is None or ci_hi is None:
        return evidencia_acmg(oddspath_puntual)[0] + "_sin_IC", (
            "patogenico" if oddspath_puntual >= 1 else "benigno")
    if oddspath_puntual >= 1:
        return evidencia_acmg(ci_lo)  # el limite inferior del IC decide la fuerza
    else:
        return evidencia_acmg(ci_hi)  # el limite superior del IC decide la fuerza


def main():
    with open("/home/jesus/paper_msh6/datos/CONJUNTO_VALIDACION_EXTERNA_CONGELADO.json") as f:
        frozen = json.load(f)

    resultados = {}
    for etiqueta, gene, clave_frozen, oddspath_field in [
        ("MSH6", "MSH6", "msh6", "oddspath_functional_inCAMA"),
        ("PMS2", "PMS2", "pms2", "oddspath_CIMRA"),
    ]:
        print(f"\n{'='*70}\n{etiqueta}: calibracion PP3/BP4 con ESM-2 650M solo "
              f"(prior de Brnich 2019, IC95 bootstrap)\n{'='*70}")

        esm = cargar_esm(gene)

        excluir = set()
        for entrada in frozen[clave_frozen]:
            if clave_frozen == "msh6":
                v = entrada["variant_1letter"].rstrip("g")
                m = re.match(r"^([A-Z])(\d+)([A-Z])$", v)
                pos, mut_aa = int(m.group(2)), m.group(3)
            else:
                m = re.match(r"^[Pp]\.\s*([A-Za-z]{3})(\d+)([A-Za-z]{3})$", entrada["variant_protein"].strip())
                pos, mut_aa = int(m.group(2)), AA3_TO_1[m.group(3).capitalize()]
            excluir.add((pos, mut_aa))

        pat, ben = construir_calibracion(gene, excluir)
        ben_gnomad = construir_benignas_gnomad_bs1(gene, excluir | set(ben))
        ben_total = ben + ben_gnomad
        print(f"Calibracion: {len(pat)} patogenicas, {len(ben)} benignas ClinVar + "
              f"{len(ben_gnomad)} benignas gnomAD BS1 = {len(ben_total)} benignas totales")

        X_cal, y_cal = [], []
        for pos, mut_aa in pat:
            s = esm.get((pos, mut_aa))
            if s is not None:
                X_cal.append(s)
                y_cal.append(1)
        for pos, mut_aa in ben_total:
            s = esm.get((pos, mut_aa))
            if s is not None:
                X_cal.append(s)
                y_cal.append(0)
        X_cal, y_cal = np.array(X_cal).reshape(-1, 1), np.array(y_cal)
        print(f"Calibracion con score ESM disponible: {len(y_cal)} variantes "
              f"({int(sum(y_cal))} pat, {int(len(y_cal)-sum(y_cal))} ben)")

        logreg, prior_p, prior_odds = ajustar_calibracion(X_cal, y_cal)
        print(f"Prior_P (Brnich 2019, proporcion de patogenicas en el set de calibracion): "
              f"{prior_p:.4f} (prior_odds={prior_odds:.4f})")

        aciertos, total, detalle = 0, 0, []
        for entrada in frozen[clave_frozen]:
            if clave_frozen == "msh6":
                v = entrada["variant_1letter"].rstrip("g")
                m = re.match(r"^([A-Z])(\d+)([A-Z])$", v)
                pos, mut_aa = int(m.group(2)), m.group(3)
                clasificacion_real = entrada["clasificacion_predicha_por_inCAMA"]
                direccion_real = "patogenico" if clasificacion_real in ("P", "LP") else (
                    "benigno" if clasificacion_real in ("B", "LB") else "VUS")
            else:
                m = re.match(r"^[Pp]\.\s*([A-Za-z]{3})(\d+)([A-Za-z]{3})$", entrada["variant_protein"].strip())
                pos, mut_aa = int(m.group(2)), AA3_TO_1[m.group(3).capitalize()]
                acmg_real = entrada["acmg_evidence_strength"]
                direccion_real = "patogenico" if acmg_real.startswith("PS3") else (
                    "benigno" if acmg_real.startswith("BS3") else "VUS")
            if direccion_real == "VUS":
                continue
            s = esm.get((pos, mut_aa))
            if s is None:
                continue
            oddspath_puntual, ci_lo, ci_hi = score_a_oddspath_bootstrap(
                logreg, X_cal, y_cal, s, prior_odds)
            evidencia_pred, direccion_pred = evidencia_conservadora(oddspath_puntual, ci_lo, ci_hi)
            direccion_pred_simple = "patogenico" if "patogenico" in direccion_pred else "benigno"
            acierta = direccion_pred_simple == direccion_real
            aciertos += int(acierta)
            total += 1
            detalle.append({
                "variante": f"{mut_aa}@{pos}", "direccion_real": direccion_real,
                "oddspath_puntual": float(oddspath_puntual),
                "oddspath_ci95": [ci_lo, ci_hi],
                "evidencia_predicha_conservadora": evidencia_pred,
                "acierta": acierta,
            })

        tasa = aciertos / total if total else float("nan")
        ci_tasa = wilson_ci(aciertos, total)
        print(f"\nCriterio (b), direccion puntual: {aciertos}/{total} = {100*tasa:.1f}% "
              f"(IC95 Wilson [{100*ci_tasa[0]:.1f}%, {100*ci_tasa[1]:.1f}%], umbral preregistrado 70%)")
        for d in detalle:
            marca = "OK" if d["acierta"] else "FALLO"
            ci = d["oddspath_ci95"]
            ci_str = f"[{ci[0]:.3g},{ci[1]:.3g}]" if ci[0] is not None else "sin_IC"
            print(f"  [{marca}] {d['variante']}: real={d['direccion_real']:10s} "
                  f"evidencia_conservadora={d['evidencia_predicha_conservadora']:22s} "
                  f"(OddsPath={d['oddspath_puntual']:.3g}, IC95={ci_str})")

        n_pp3 = sum(1 for d in detalle if d["evidencia_predicha_conservadora"].startswith("PP3"))
        n_bp4 = sum(1 for d in detalle if d["evidencia_predicha_conservadora"].startswith("BP4"))
        n_sin = total - n_pp3 - n_bp4
        degenerada = (n_pp3 == total) or (n_bp4 == total)
        print(f"Distribucion: {n_pp3} PP3-algo, {n_bp4} BP4-algo, {n_sin} sin evidencia/sin IC")
        if degenerada:
            print(f"*** CALIBRACION DEGENERADA: predice una sola direccion para el 100% de los "
                  f"casos. La 'tasa de acierto' de {100*tasa:.1f}% es la prevalencia del test "
                  f"({sum(1 for d in detalle if d['direccion_real']=='patogenico')}/{total} patogenicas), "
                  f"NO una medida de discriminacion. No se reporta como criterio (b) superado. ***")

        resultados[etiqueta] = {
            "n_calibracion_pat": len(pat), "n_calibracion_ben": len(ben_total),
            "prior_p_brnich2019": prior_p,
            "aciertos": aciertos, "total": total, "tasa_acierto": tasa,
            "tasa_acierto_ic95_wilson": list(ci_tasa),
            "cumple_criterio_b_70pct_puntual": bool(tasa >= 0.70) if total else None,
            "calibracion_degenerada": degenerada,
            "n_pp3": n_pp3, "n_bp4": n_bp4, "detalle": detalle,
        }

    with open("/home/jesus/paper_msh6/datos/resultado_calibracion_ESM_solo.json", "w") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print("\n\nGuardado: datos/resultado_calibracion_ESM_solo.json")


if __name__ == "__main__":
    main()
