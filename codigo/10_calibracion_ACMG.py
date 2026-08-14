"""
Criterio (b) del preregistro: calibracion ExCALIBR (gen a gen, siguiendo el metodo
de Tavtigian et al. 2018/2020 y Pejaver et al. 2022) para convertir el score del
modelo de transferencia en fuerza de evidencia ACMG (PS3/BS3), y comprobar si esa
fuerza de evidencia coincide en direccion con la clasificacion funcional real
publicada (inCAMA para MSH6, CIMRA para PMS2) en al menos el 70% de los casos.

Paso 1: construir el conjunto de calibracion GEN A GEN (missense P/LP y B/LB de alta
confianza en ClinVar del propio gen huerfano -- MSH6 o PMS2 --, EXCLUYENDO las
variantes del conjunto de validacion congelado, para no calibrar y evaluar con los
mismos datos).
Paso 2: predecir el score de transferencia (modelo entrenado en el gen rico) sobre
ese conjunto de calibracion.
Paso 3: ajustar una regresion logistica score -> P(patogenico), y convertir a
OddsPath con el prior estandar de la literatura (0.10, ver nota de vigencia abajo).
Paso 4: aplicar la MISMA funcion de calibracion a las variantes congeladas y
comparar con la clasificacion funcional real.

NOTA DE VIGENCIA: el prior de 0.10 es el valor histórico recomendado por el
ClinGen SVI WG (retirado en abril de 2025, ver PREREGISTRO.md apartado 3.5). El
valor en si sigue siendo el estandar de facto citado por CIMRA, inCAMA y el propio
IGVF pillar paper -- pero debe reverificarse contra la guia vigente del ClinGen/AVE
Functional WG antes de escribir el manuscrito final. Aqui se usa de forma explicita
y declarada, no oculta.
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

ALTA_CONFIANZA = {"reviewed by expert panel", "practice guideline",
                   "criteria provided, multiple submitters, no conflicts"}

PRIOR_P = 0.10  # ver nota de vigencia en el docstring
PRIOR_ODDS = PRIOR_P / (1 - PRIOR_P)

# Umbrales de Tavtigian et al. 2018 (OddsPath)
UMBRALES_PATOGENICO = [("VeryStrong", 350), ("Strong", 18.7), ("Moderate", 4.3), ("Supporting", 2.08)]
UMBRALES_BENIGNO = [("VeryStrong", 1/350), ("Strong", 1/18.7), ("Moderate", 1/4.3), ("Supporting", 1/2.08)]


def evidencia_acmg(oddspath):
    if oddspath >= 1:
        for nombre, umbral in UMBRALES_PATOGENICO:
            if oddspath >= umbral:
                return f"PS3_{nombre}", "patogenico"
        return "sin_evidencia", "patogenico_debil"
    else:
        for nombre, umbral in UMBRALES_BENIGNO:
            if oddspath <= umbral:
                return f"BS3_{nombre}", "benigno"
        return "sin_evidencia", "benigno_debil"


def cargar_clinvar_gen(gene, path="/home/jesus/paper_msh6/datos/variant_summary.txt"):
    filas = []
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["Assembly"] != "GRCh38":
                continue
            if gene not in set(row["GeneSymbol"].split("|")):
                continue
            filas.append(row)
    return filas


def extraer_missense_pos(name):
    """'NM_...(MSH6):c.658G>A (p.Glu220Asp)' -> (220, 'E', 'D') o None si no es missense simple."""
    m = re.search(r"\(p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})\)", name)
    if not m:
        return None
    aa1_3, pos, aa2_3 = m.groups()
    aa1 = AA3_TO_1.get(aa1_3.capitalize())
    aa2 = AA3_TO_1.get(aa2_3.capitalize())
    if aa1 is None or aa2 is None or aa1 == aa2:
        return None
    return int(pos), aa1, aa2


UMBRAL_BS1_AF = 0.001  # 0,1%: umbral de trabajo declarado para "demasiado comun para ser
                        # patogenica dominante", ver docstring de abajo.


def construir_benignas_gnomad_bs1(gene, excluir_posiciones):
    """Variantes missense de gnomAD con frecuencia poblacional > UMBRAL_BS1_AF: logica
    ACMG BS1 (demasiado comunes para ser una variante patogenica dominante de alta
    penetrancia). Se usan como refuerzo del conjunto de calibracion benigno cuando
    ClinVar tiene muy pocas benignas de alta confianza para el gen (el problema
    diagnosticado para PMS2: solo 3 benignas de >=2 estrellas).

    UMBRAL declarado explicitamente: 0,1% (1 entre 1000). Es un umbral de trabajo
    generico, no calibrado especificamente para la penetrancia reducida de PMS2 (~11-20%
    de riesgo de cancer colorrectal a los 70 anos, ver CIMRA/Rayner et al. 2022) --
    se declara como simplificacion pendiente de refinar con guia especifica de gen,
    no se oculta.
    """
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
        if wt_aa is None or mut_aa is None or (int(pos), mut_aa) in excluir_posiciones:
            continue
        ben.append((int(pos), wt_aa, mut_aa))
    return ben


def construir_calibracion(gene, excluir_posiciones):
    filas = cargar_clinvar_gen(gene)
    pat, ben = [], []
    for row in filas:
        if row["ReviewStatus"] not in ALTA_CONFIANZA:
            continue
        parsed = extraer_missense_pos(row["Name"])
        if parsed is None:
            continue
        pos, wt_aa, mut_aa = parsed
        if (pos, mut_aa) in excluir_posiciones:
            continue
        cs = row["ClinicalSignificance"]
        if cs in ("Pathogenic", "Pathogenic/Likely pathogenic"):
            pat.append((pos, wt_aa, mut_aa))
        elif cs in ("Benign", "Benign/Likely benign"):
            ben.append((pos, wt_aa, mut_aa))
        elif cs == "Likely pathogenic":
            pat.append((pos, wt_aa, mut_aa))
        elif cs == "Likely benign":
            ben.append((pos, wt_aa, mut_aa))
    return pat, ben


def main():
    import importlib
    construir_mod = importlib.import_module("07_construir_dataset_H0")

    with open("/home/jesus/paper_msh6/datos/CONJUNTO_VALIDACION_EXTERNA_CONGELADO.json") as f:
        frozen = json.load(f)

    resultados = {}

    FEATURES_BASE_PLDDT = ["esm2_3B_zeroshot", "blosum62", "delta_hidrofobicidad", "delta_volumen", "plddt"]
    FEATURES_H1 = FEATURES_BASE_PLDDT + ["dist_adn", "dist_pareja"]  # MutSalpha: co-cristal 2O8C real
    FEATURES_H2 = FEATURES_BASE_PLDDT  # MutLalpha: sin co-cristal humano, ver 11_features_estructurales.py

    for etiqueta, gene_huerfano, gene_rico, dataset_rico, clave_frozen, feature_cols in [
        ("H1", "MSH6", "MSH2", "dataset_H0_MSH2.json", "msh6", FEATURES_H1),
        ("H2", "PMS2", "MLH1", "dataset_H0_MLH1.json", "pms2", FEATURES_H2),
    ]:
        print(f"\n{'='*70}\n{etiqueta}: calibracion ACMG para {gene_huerfano} "
              f"(modelo entrenado en {gene_rico})\n{'='*70}")

        # posiciones a excluir del set de calibracion: las del conjunto congelado
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

        pat, ben = construir_calibracion(gene_huerfano, excluir)
        print(f"Calibracion de {gene_huerfano} (ClinVar >=2 estrellas, excluyendo el conjunto "
              f"congelado): {len(pat)} patogenicas/probables, {len(ben)} benignas/probables")

        ben_gnomad = construir_benignas_gnomad_bs1(gene_huerfano, excluir | set((p, m) for p, w, m in ben))
        print(f"Refuerzo BS1 (gnomAD, af > {UMBRAL_BS1_AF}, sin solapar con lo anterior): "
              f"+{len(ben_gnomad)} benignas de {gene_huerfano}")
        ben_total = ben + ben_gnomad

        if len(pat) < 5 or len(ben_total) < 5:
            print(f"AVISO: conjunto de calibracion demasiado pequeno para {gene_huerfano} "
                  f"({len(pat)} pat, {len(ben_total)} ben). Se reporta igualmente pero con cautela.")

        # entrenar el modelo de transferencia (igual que en 09_evaluar_H1_H2.py)
        with open(f"/home/jesus/paper_msh6/datos/{dataset_rico}") as f:
            data_rico = json.load(f)
        data_rico = [d for d in data_rico if all(c in d for c in feature_cols)]
        import lightgbm as lgb
        X_rico = np.array([[d[c] for c in feature_cols] for d in data_rico])
        y_rico = np.array([d["score_danino"] for d in data_rico])
        modelo = lgb.LGBMRegressor(n_estimators=200, num_leaves=15, random_state=20260805,
                                     n_jobs=2, verbosity=-1)
        modelo.fit(X_rico, y_rico)

        with open(f"/home/jesus/paper_msh6/datos/esm2_3B_zeroshot/{gene_huerfano}_esm2_3B_zeroshot.json") as f:
            raw = json.load(f)
        esm_huerfano = {}
        for k, v in raw.items():
            pos_str, aa = k.rsplit("_", 1)
            esm_huerfano[(int(pos_str), aa)] = v

        def feats_de(pos, wt_aa, mut_aa):
            f = construir_mod.features_variante(pos, wt_aa, mut_aa, esm_huerfano,
                                                  gene=gene_huerfano, usar_estructura=True)
            if f is None or not all(c in f for c in feature_cols):
                return None
            return [f[c] for c in feature_cols]

        X_cal, y_cal = [], []
        for pos, wt_aa, mut_aa in pat:
            f = feats_de(pos, wt_aa, mut_aa)
            if f:
                X_cal.append(f)
                y_cal.append(1)
        for pos, wt_aa, mut_aa in ben_total:
            f = feats_de(pos, wt_aa, mut_aa)
            if f:
                X_cal.append(f)
                y_cal.append(0)
        X_cal, y_cal = np.array(X_cal), np.array(y_cal)
        print(f"Calibracion con features completas: {len(y_cal)} variantes "
              f"({sum(y_cal)} patogenicas, {len(y_cal)-sum(y_cal)} benignas)")

        pred_cal_score = modelo.predict(X_cal)  # score_danino predicho por el modelo de transferencia

        logreg = LogisticRegression()
        logreg.fit(pred_cal_score.reshape(-1, 1), y_cal)

        def score_a_oddspath(score):
            p = logreg.predict_proba([[score]])[0, 1]
            p = min(max(p, 1e-6), 1 - 1e-6)
            post_odds = p / (1 - p)
            return post_odds / PRIOR_ODDS

        # aplicar al conjunto congelado
        aciertos = 0
        total = 0
        detalle = []
        for entrada in frozen[clave_frozen]:
            if clave_frozen == "msh6":
                v = entrada["variant_1letter"].rstrip("g")
                m = re.match(r"^([A-Z])(\d+)([A-Z])$", v)
                wt_aa, pos, mut_aa = m.group(1), int(m.group(2)), m.group(3)
                clasificacion_real = entrada["clasificacion_predicha_por_inCAMA"]
                direccion_real = "patogenico" if clasificacion_real in ("P", "LP") else (
                    "benigno" if clasificacion_real in ("B", "LB") else "VUS")
            else:
                m = re.match(r"^[Pp]\.\s*([A-Za-z]{3})(\d+)([A-Za-z]{3})$", entrada["variant_protein"].strip())
                aa1_3, pos, aa2_3 = m.groups()
                wt_aa, mut_aa, pos = AA3_TO_1[aa1_3.capitalize()], AA3_TO_1[aa2_3.capitalize()], int(pos)
                acmg_real = entrada["acmg_evidence_strength"]
                direccion_real = "patogenico" if acmg_real.startswith("PS3") else (
                    "benigno" if acmg_real.startswith("BS3") else "VUS")

            if direccion_real == "VUS":
                continue  # no hay direccion de referencia con la que comparar

            f = feats_de(pos, wt_aa, mut_aa)
            if f is None:
                continue
            score_pred = modelo.predict([f])[0]
            oddspath_pred = score_a_oddspath(score_pred)
            evidencia_pred, direccion_pred = evidencia_acmg(oddspath_pred)

            direccion_pred_simple = "patogenico" if "patogenico" in direccion_pred else "benigno"
            acierta = direccion_pred_simple == direccion_real
            aciertos += int(acierta)
            total += 1
            detalle.append({
                "variante": f"{wt_aa}{pos}{mut_aa}", "direccion_real": direccion_real,
                "oddspath_predicho": float(oddspath_pred), "evidencia_predicha": evidencia_pred,
                "acierta_direccion": acierta,
            })

        tasa = aciertos / total if total else float("nan")
        print(f"\nCriterio (b): direccion correcta en {aciertos}/{total} = {100*tasa:.1f}% "
              f"(umbral declarado: 70%)")
        for d in detalle:
            marca = "OK" if d["acierta_direccion"] else "FALLO"
            print(f"  [{marca}] {d['variante']}: real={d['direccion_real']:10s} "
                  f"predicho={d['evidencia_predicha']:20s} (OddsPath={d['oddspath_predicho']:.3g})")

        resultados[etiqueta] = {
            "gene_huerfano": gene_huerfano, "gene_rico": gene_rico,
            "n_calibracion_patogenicas": int(sum(y_cal)), "n_calibracion_benignas": int(len(y_cal) - sum(y_cal)),
            "aciertos": aciertos, "total": total, "tasa_acierto": tasa,
            "cumple_criterio_b_70pct": bool(tasa >= 0.70) if total else None,
            "detalle": detalle,
        }

    with open("/home/jesus/paper_msh6/datos/resultado_calibracion_ACMG.json", "w") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print("\n\nGuardado: datos/resultado_calibracion_ACMG.json")


if __name__ == "__main__":
    main()
