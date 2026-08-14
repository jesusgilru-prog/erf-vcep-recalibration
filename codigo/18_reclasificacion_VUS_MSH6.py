"""
VERSION CORREGIDA (10-ago-2026, tras revision_debate_2/ACTA.md): aplica la
calibracion PP3/BP4 corregida (prior de Brnich 2019 = prevalencia del propio
set de calibracion, fuerza declarada = limite conservador del IC95 bootstrap
del OddsPath) a TODAS las VUS missense reales y actuales de MSH6 y PMS2 en
ClinVar. Reemplaza la version anterior, que usaba PRIOR_P=0.10 fijo y
etiquetaba PS3/BS3 (categoria equivocada para evidencia computacional) y
declaraba fuerza sin IC.

Nota: por coste computacional (miles de VUS x 2000 remuestreos bootstrap cada
una seria intratable), el bootstrap del IC se calcula sobre el CLASIFICADOR
(distribucion de logreg+prior por remuestreo de la calibracion), reutilizado
para todas las VUS: se generan N_BOOT logisticas alternativas UNA VEZ, y para
cada VUS se evalua su OddsPath bajo cada una, tomando percentiles. Es la misma
matematica que 17_calibracion_ACMG_con_ESM_solo.py, vectorizada.
"""
import csv
import json
import re
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

csv.field_size_limit(sys.maxsize)

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}
AA1_TO_3 = {v: k for k, v in AA3_TO_1.items()}
ALTA_CONFIANZA = {"reviewed by expert panel", "practice guideline",
                   "criteria provided, multiple submitters, no conflicts"}
# Topes REALES de fuerza, verificados el 10-ago-2026 contra la especificacion oficial
# vigente de ClinGen InSiGHT Hereditary CRC/Polyposis VCEP para MSH6 (GN138 v2.0) y PMS2
# (GN139 v2.0), ambas "Released", ultima actualizacion 2026-03-05
# (cspec.genome.network/cspec/api/SequenceVariantInterpretation/id/GN138 y GN139):
# PP3 (evidencia patogenica computacional) SOLO es aplicable hasta Moderate.
# BP4 (evidencia benigna computacional) SOLO es aplicable hasta Supporting.
# Strong y VeryStrong estan explicitamente "Not Applicable" para evidencia computacional
# en estos genes -- usar herramientas distintas (HCI-prior para MSH6, MAPP/PP2 Prior P
# para PMS2, ninguna es ESM-2) con umbrales propios (0.11/0.68/0.81), no los de
# Tavtigian 2018 genericos. Aqui se capa la fuerza MAXIMA declarable con ESM-2+Tavtigian
# a lo que el VCEP permite para evidencia computacional, aunque el umbral numerico siga
# siendo el de Tavtigian (no se tiene el score HCI-prior/MAPP-PP2 para recalibrar contra
# sus propios cortes 0.11/0.68/0.81) -- se declara esta limitacion explicitamente.
UMBRALES_PATOGENICO = [("Moderate", 4.3), ("Supporting", 2.08)]
UMBRALES_BENIGNO = [("Supporting", 1/2.08)]
UMBRAL_BS1_AF = 0.001
RNG_SEED = 20260810
N_BOOT = 500  # menor que en 17_ (2000) por coste: se aplica a miles de VUS, no a 18-51


def evidencia_acmg(oddspath):
    if oddspath >= 1:
        for nombre, umbral in UMBRALES_PATOGENICO:
            if oddspath >= umbral:
                return f"PP3_{nombre}"
        return "sin_evidencia"
    else:
        for nombre, umbral in UMBRALES_BENIGNO:
            if oddspath <= umbral:
                return f"BP4_{nombre}"
        return "sin_evidencia"


def cargar_esm(gene):
    path = f"/home/jesus/paper_msh6/datos/esm2_zeroshot/{gene}_esm2_650M_zeroshot.json"
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        pos_str, aa = k.rsplit("_", 1)
        out[(int(pos_str), aa)] = -v
    return out


def extraer_missense(name):
    m = re.search(r"\(p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})\)", name)
    if not m:
        return None
    aa1_3, pos, aa2_3 = m.groups()
    aa1, aa2 = AA3_TO_1.get(aa1_3.capitalize()), AA3_TO_1.get(aa2_3.capitalize())
    if aa1 is None or aa2 is None or aa1 == aa2:
        return None
    return int(pos), aa1, aa2


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


def procesar_gen(gene, frozen_key, frozen):
    print(f"\n{'='*70}\n{gene}\n{'='*70}")
    esm = cargar_esm(gene)
    filas_gen = cargar_filas_gen(gene)

    excluir = set()
    for entrada in frozen[frozen_key]:
        if frozen_key == "msh6":
            v = entrada["variant_1letter"].rstrip("g")
            m = re.match(r"^([A-Z])(\d+)([A-Z])$", v)
            excluir.add((int(m.group(2)), m.group(3)))
        else:
            m = re.match(r"^[Pp]\.\s*([A-Za-z]{3})(\d+)([A-Za-z]{3})$", entrada["variant_protein"].strip())
            excluir.add((int(m.group(2)), AA3_TO_1[m.group(3).capitalize()]))

    pat, ben = [], []
    for row in filas_gen:
        if row["ReviewStatus"] not in ALTA_CONFIANZA:
            continue
        parsed = extraer_missense(row["Name"])
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
    ben_gnomad = construir_benignas_gnomad_bs1(gene, excluir | set(ben))
    ben_total = ben + ben_gnomad

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
    print(f"Calibracion: {int(sum(y_cal))} pat + {int(len(y_cal)-sum(y_cal))} ben = {len(y_cal)}")

    logreg = LogisticRegression()
    logreg.fit(X_cal, y_cal)
    prior_p = float(np.mean(y_cal))
    prior_odds = prior_p / (1 - prior_p)
    print(f"Prior_P (Brnich 2019): {prior_p:.4f}")

    # Clasificadores bootstrap (una vez), reutilizados para todas las VUS.
    rng = np.random.default_rng(RNG_SEED)
    n = len(y_cal)
    modelos_boot = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        Xb, yb = X_cal[idx], y_cal[idx]
        if len(np.unique(yb)) < 2:
            continue
        pb = float(np.mean(yb))
        if pb <= 0 or pb >= 1:
            continue
        lr_b = LogisticRegression()
        lr_b.fit(Xb, yb)
        modelos_boot.append((lr_b, pb / (1 - pb)))
    print(f"Modelos bootstrap validos: {len(modelos_boot)}/{N_BOOT}")

    def score_a_evidencia_conservadora(score):
        p = logreg.predict_proba([[score]])[0, 1]
        p = min(max(p, 1e-6), 1 - 1e-6)
        oddspath_puntual = (p / (1 - p)) / prior_odds

        oddspaths_boot = []
        for lr_b, prior_odds_b in modelos_boot:
            pb = lr_b.predict_proba([[score]])[0, 1]
            pb = min(max(pb, 1e-6), 1 - 1e-6)
            oddspaths_boot.append((pb / (1 - pb)) / prior_odds_b)
        ci_lo, ci_hi = np.percentile(oddspaths_boot, [2.5, 97.5])

        if oddspath_puntual >= 1:
            return evidencia_acmg(ci_lo), oddspath_puntual, ci_lo, ci_hi
        else:
            return evidencia_acmg(ci_hi), oddspath_puntual, ci_lo, ci_hi

    vus_unicas = {}
    for row in filas_gen:
        if row["ClinicalSignificance"] != "Uncertain significance":
            continue
        parsed = extraer_missense(row["Name"])
        if parsed is None:
            continue
        pos, wt_aa, mut_aa = parsed
        vus_unicas[(pos, wt_aa, mut_aa)] = row
    print(f"VUS missense unicas de {gene} en ClinVar hoy: {len(vus_unicas)}")

    resultados = []
    sin_score = 0
    for (pos, wt_aa, mut_aa), row in vus_unicas.items():
        s = esm.get((pos, mut_aa))
        if s is None:
            sin_score += 1
            continue
        evidencia, oddspath_puntual, ci_lo, ci_hi = score_a_evidencia_conservadora(s)
        resultados.append({
            "variante": f"p.{AA1_TO_3[wt_aa]}{pos}{AA1_TO_3[mut_aa]}",
            "posicion": pos, "oddspath_puntual": float(oddspath_puntual),
            "oddspath_ci95": [float(ci_lo), float(ci_hi)],
            "evidencia_conservadora": evidencia,
            "variation_id": row["VariationID"],
        })

    print(f"VUS con score ESM-2 disponible: {len(resultados)} (sin score: {sin_score})")
    from collections import Counter
    conteo = Counter(r["evidencia_conservadora"] for r in resultados)
    print("Distribucion de evidencia (conservadora, limite de IC):")
    for ev, n_ev in conteo.most_common():
        print(f"  {ev}: {n_ev} ({100*n_ev/len(resultados):.1f}%)")

    pp3_n = sum(1 for r in resultados if r["evidencia_conservadora"].startswith("PP3"))
    bp4_n = sum(1 for r in resultados if r["evidencia_conservadora"].startswith("BP4"))
    pp3_moderate_mas = sum(1 for r in resultados if r["evidencia_conservadora"] in
                            ("PP3_Moderate", "PP3_Strong", "PP3_VeryStrong"))
    bp4_moderate_mas = sum(1 for r in resultados if r["evidencia_conservadora"] in
                            ("BP4_Moderate", "BP4_Strong", "BP4_VeryStrong"))
    print(f"\nPP3 cualquier fuerza: {pp3_n} ({100*pp3_n/len(resultados):.1f}%); "
          f"BP4 cualquier fuerza: {bp4_n} ({100*bp4_n/len(resultados):.1f}%)")
    print(f"Con fuerza >= Moderate (limite conservador del IC): "
          f"{pp3_moderate_mas} PP3 + {bp4_moderate_mas} BP4 = {pp3_moderate_mas+bp4_moderate_mas} "
          f"({100*(pp3_moderate_mas+bp4_moderate_mas)/len(resultados):.1f}% de las VUS)")

    return {
        "n_vus_total": len(vus_unicas), "n_con_score": len(resultados), "sin_score": sin_score,
        "prior_p": prior_p, "n_calibracion": len(y_cal),
        "distribucion_evidencia": dict(conteo),
        "pp3_cualquier_fuerza": pp3_n, "bp4_cualquier_fuerza": bp4_n,
        "fuerza_moderate_o_mas_conservadora": pp3_moderate_mas + bp4_moderate_mas,
        "pp3_moderate_o_mas": pp3_moderate_mas, "bp4_moderate_o_mas": bp4_moderate_mas,
        "resultados": resultados,
    }


def main():
    with open("/home/jesus/paper_msh6/datos/CONJUNTO_VALIDACION_EXTERNA_CONGELADO.json") as f:
        frozen = json.load(f)

    salida = {}
    salida["MSH6"] = procesar_gen("MSH6", "msh6", frozen)
    salida["PMS2"] = procesar_gen("PMS2", "pms2", frozen)

    with open("/home/jesus/paper_msh6/datos/reclasificacion_VUS_MSH6.json", "w") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    print("\nGuardado: datos/reclasificacion_VUS_MSH6.json")


if __name__ == "__main__":
    main()
