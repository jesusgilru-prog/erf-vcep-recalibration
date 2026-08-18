"""
Experimento E-RF (recalibracion de fuerza de evidencia funcional), declarado en
PREREGISTRO.md seccion 10.3 ANTES de ejecutar este script -- rama de exito y
rama de refutacion fijadas de antemano, para que esto no se lea como HARKing
sobre datos con verdad-terreno de MSH2 que el proyecto ya ha visto.

Pregunta: cuando se mide contra la propia verdad funcional (DMS de MSH2), no
contra ClinVar ni contra otro predictor, ¿el LR+ (razon de verosimilitud
positiva) que entregan los umbrales fijos de la herramienta oficial VCEP
(MAPP/PP2 Prior P: 0.11 BP4-Supporting / 0.68 PP3-Supporting / 0.81
PP3-Moderate) alcanza la fuerza nominal que esos umbrales declaran (cortes
OddsPath de Tavtigian et al. 2018: Supporting>=2.08, Moderate>=4.3, e inverso
para BP4)? Se repite la misma pregunta para ESM-2, AlphaMissense y ESM-1v,
calibrando sus propios umbrales Supporting/Moderate con la MISMA metodologia
que ya usa el proyecto (codigo/17_: regresion logistica por gen, prior =
proporcion patogenica del propio set de calibracion, Brnich et al. 2019),
pero aplicada aqui a MSH2 (que no forma parte del conjunto congelado externo,
no hace falta excluir nada).

Dos poblaciones, con dos definiciones de LR+ distintas y declaradas:

1. LR+ "duro": sobre las variantes de etiqueta funcional de ALTA CONFIANZA
   (mezcla de 2 Gaussianas ya usada en codigo/44_, posterior>0.9 en el
   componente patogenico o benigno). Sensibilidad/especificidad clasicas.
2. LR+ "blando" (numero titular): sobre la REGION AMBIGUA (posterior<=0.9 en
   ambos componentes) -- la poblacion real para la que se invoca PP3/BP4 en
   la practica clinica. Como por definicion esta region no tiene una
   etiqueta binaria de confianza, se usa la probabilidad posterior de cada
   componente de la propia mezcla como PESO (TPR/FPR ponderados por peso
   blando en vez de contados 0/1) -- decision metodologica explicita, no
   escondida: es la unica forma de obtener una cifra en la poblacion
   ambigua sin fabricar una etiqueta dura donde no la hay.

Extrapolacion final (declarada como tal, no como medicion directa): si algun
umbral pierde fuerza, se recuenta cuantas VUS reales de MSH6/PMS2 cambiarian
de codigo bajo la fuerza recalibrada -- MSH6/PMS2 no tienen DMS propio, asi
que esto es una extrapolacion entre parálogos con rango, no un dato medido
en esos genes.
"""
import hashlib
import importlib
import json
import re
import sys

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

sys.path.insert(0, "/home/jesus/paper_msh6/codigo")
c40 = importlib.import_module("40_comparar_alphamissense")
c41 = importlib.import_module("41_comparar_esm1v")

SEMILLA_GMM = 20260812          # misma que codigo/44_, para reproducir la misma clasificacion
RNG_SEED_BOOT = 20260810        # misma que codigo/17_

def _seed_det(s):
    return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16) % 10000

N_BOOT = 10000
UMBRAL_POSTERIOR = 0.9
CORTE_SUPPORTING = 2.08
CORTE_MODERATE = 4.3

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}
UMBRAL_BS1_AF = 0.001


# --------------------------------------------------------------------------
# Carga de datos
# --------------------------------------------------------------------------

def cargar_dms_msh2():
    with open("/home/jesus/paper_msh6/datos/dataset_H0_MSH2.json") as f:
        dms = json.load(f)
    return {(d["posicion"], d["mut_aa"]): d["score_danino"] for d in dms}


def cargar_predictores_msh2():
    return {
        "oficial": c40.cargar_oficial("MSH2_priors"),
        "esm2": c40.cargar_esm("MSH2"),
        "alphamissense": c40.cargar_alphamissense("MSH2"),
        "esm1v": c41.cargar_esm1v("MSH2"),
    }


def cargar_clinvar_gen(gene):
    import csv
    csv.field_size_limit(sys.maxsize)
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


def construir_calibracion_clinvar(gene):
    pat, ben = [], []
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
        if cs in ("Pathogenic", "Pathogenic/Likely pathogenic", "Likely pathogenic"):
            pat.append((int(pos), aa2))
        elif cs in ("Benign", "Benign/Likely benign", "Likely benign"):
            ben.append((int(pos), aa2))
    return pat, ben


def construir_benignas_gnomad_bs1(gene):
    import os
    path = f"/home/jesus/paper_msh6/datos/gnomad_{gene}.json"
    if not os.path.exists(path):
        # MSH2 nunca necesito refuerzo BS1 (293 benignas ClinVar de alta confianza
        # ya disponibles, muchas mas que las 5-8 de MSH6/PMS2 que motivaron el
        # refuerzo BS1 el 6-ago-2026); no se descargo gnomAD para MSH2.
        print(f"    (sin fichero gnomAD para {gene}, se omite refuerzo BS1 -- esperado para MSH2)")
        return []
    with open(path) as f:
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
        if wt_aa is None or mut_aa is None:
            continue
        ben.append((int(pos), mut_aa))
    return ben


# --------------------------------------------------------------------------
# GMM sobre la verdad funcional (misma metodologia que codigo/44_)
# --------------------------------------------------------------------------

def ajustar_gmm(real):
    keys = sorted(real.keys())
    y = np.array([real[k] for k in keys])
    gmm = GaussianMixture(n_components=2, random_state=SEMILLA_GMM, n_init=10)
    gmm.fit(y.reshape(-1, 1))
    post = gmm.predict_proba(y.reshape(-1, 1))
    medias = gmm.means_.ravel()
    idx_patho = int(np.argmax(medias))
    idx_ben = int(np.argmin(medias))
    print(f"  GMM: medias={medias.tolist()}, pesos={gmm.weights_.tolist()}, "
          f"idx_patogenico={idx_patho}, idx_benigno={idx_ben}")
    post_patho = {k: float(post[i, idx_patho]) for i, k in enumerate(keys)}
    post_ben = {k: float(post[i, idx_ben]) for i, k in enumerate(keys)}
    return post_patho, post_ben, medias.tolist(), gmm.weights_.tolist()


# --------------------------------------------------------------------------
# Calibracion de umbrales propios por predictor (misma metodologia codigo/17_,
# aplicada aqui a MSH2 -- no forma parte del conjunto congelado externo)
# --------------------------------------------------------------------------

def calibrar_umbrales_predictor(scores_dict, gene="MSH2"):
    """Ajusta LogReg(score -> patogenico) con prior de Brnich 2019 (proporcion
    patogenica del propio set de calibracion), y devuelve los cortes de score
    (patho_supporting, patho_moderate, ben_supporting) donde el OddsPath
    puntual cruza 2.08 / 4.3 / 1/2.08."""
    pat, ben_clinvar = construir_calibracion_clinvar(gene)
    ben_gnomad = construir_benignas_gnomad_bs1(gene)
    ben = list(set(ben_clinvar) | set(ben_gnomad))

    X, y = [], []
    for k in pat:
        if k in scores_dict:
            X.append(scores_dict[k]); y.append(1)
    for k in ben:
        if k in scores_dict:
            X.append(scores_dict[k]); y.append(0)
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    n_pat, n_ben = int(y.sum()), int((1 - y).sum())
    print(f"    calibracion: {n_pat} patogenicas, {n_ben} benignas (ClinVar alta confianza + gnomAD BS1)")

    logreg = LogisticRegression()
    logreg.fit(X, y)
    prior_p = float(np.mean(y))
    prior_odds = prior_p / (1 - prior_p)

    def oddspath_de_score(s):
        p = logreg.predict_proba([[s]])[0, 1]
        p = min(max(p, 1e-9), 1 - 1e-9)
        return (p / (1 - p)) / prior_odds

    todos = np.array(sorted(scores_dict.values()))
    lo, hi = todos.min(), todos.max()
    malla = np.linspace(lo, hi, 20000)
    odds_malla = np.array([oddspath_de_score(s) for s in malla])

    def primer_cruce(objetivo, direccion):
        if direccion == "patho":
            idx = np.where(odds_malla >= objetivo)[0]
        else:
            idx = np.where(odds_malla <= objetivo)[0]
        return float(malla[idx[0]]) if len(idx) else None

    corte_patho_supporting = primer_cruce(CORTE_SUPPORTING, "patho")
    corte_patho_moderate = primer_cruce(CORTE_MODERATE, "patho")
    corte_ben_supporting = primer_cruce(1 / CORTE_SUPPORTING, "benigno")
    print(f"    cortes de score calibrados: BP4_Supporting<={corte_ben_supporting}, "
          f"PP3_Supporting>={corte_patho_supporting}, PP3_Moderate>={corte_patho_moderate}")
    return {"n_pat_calibracion": n_pat, "n_ben_calibracion": n_ben,
            "prior_p": prior_p,
            "corte_ben_supporting": corte_ben_supporting,
            "corte_patho_supporting": corte_patho_supporting,
            "corte_patho_moderate": corte_patho_moderate}


# --------------------------------------------------------------------------
# LR+ contra la verdad funcional real (DMS), duro (region inequivoca) y
# blando (region ambigua, ponderado por posterior)
# --------------------------------------------------------------------------

def lr_duro_con_ic(scores, keys_inequivoca, y_hard, threshold, direccion, seed):
    """LR+ = P(test positivo | la clase que el umbral dice apoyar) /
    P(test positivo | la clase contraria). direccion='patho': test positivo =
    score alto, apoya patogenico -> LR=tpr/fpr (tpr=P(+|patho), estandar).
    direccion='benigno': test positivo = score bajo, apoya benigno -> LR debe
    ser P(+|benigno)/P(+|patho) = fpr/tpr, NO tpr/fpr (bug real detectado y
    corregido antes de reportar nada: con tpr/fpr al reves, BP4_Supporting
    salia con LR<1, que es matematicamente imposible para un umbral que por
    diseno separa casi todo el peso benigno del patogenico)."""
    s = np.array([scores[k] for k in keys_inequivoca])
    y = np.array(y_hard)
    if direccion == "patho":
        testpos = s >= threshold
    else:
        testpos = s <= threshold
    n_pat, n_ben = int((y == 1).sum()), int((y == 0).sum())
    if n_pat == 0 or n_ben == 0:
        return None
    tpr = testpos[y == 1].mean()  # P(test+ | patogenico)
    fpr = testpos[y == 0].mean()  # P(test+ | benigno)
    if direccion == "patho":
        lr_puntual = float(tpr / fpr) if fpr > 0 else float("inf")
    else:
        lr_puntual = float(fpr / tpr) if tpr > 0 else float("inf")

    rng = np.random.default_rng(seed)
    n = len(y)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yb, sb = y[idx], s[idx]
        if (yb == 1).sum() == 0 or (yb == 0).sum() == 0:
            continue
        tb = sb >= threshold if direccion == "patho" else sb <= threshold
        tpr_b = tb[yb == 1].mean()
        fpr_b = tb[yb == 0].mean()
        if direccion == "patho":
            if fpr_b > 0:
                boots.append(tpr_b / fpr_b)
        else:
            if tpr_b > 0:
                boots.append(fpr_b / tpr_b)
    ci = np.percentile(boots, [2.5, 97.5]).tolist() if len(boots) >= 200 else [None, None]
    return {"n_pat": n_pat, "n_ben": n_ben, "tpr": float(tpr), "fpr": float(fpr),
            "lr_puntual": lr_puntual, "ic95": ci, "n_boot_validos": len(boots)}


def lr_blando_con_ic(scores, keys_ambigua, post_patho, post_ben, threshold, direccion, seed):
    """Misma logica de direccion que lr_duro_con_ic (ver docstring alli), pero
    con TPR/FPR ponderados por la probabilidad posterior blanda de la mezcla
    en vez de una etiqueta dura 0/1 (no existe etiqueta dura en la region
    ambigua por definicion)."""
    s = np.array([scores[k] for k in keys_ambigua])
    wp = np.array([post_patho[k] for k in keys_ambigua])
    wb = np.array([post_ben[k] for k in keys_ambigua])
    if direccion == "patho":
        testpos = s >= threshold
    else:
        testpos = s <= threshold

    def lr_de(wp_, wb_, tp_):
        tpr_ = wp_[tp_].sum() / wp_.sum() if wp_.sum() > 0 else np.nan
        fpr_ = wb_[tp_].sum() / wb_.sum() if wb_.sum() > 0 else np.nan
        if direccion == "patho":
            if not fpr_ or fpr_ == 0 or np.isnan(fpr_):
                return np.inf
            return tpr_ / fpr_
        else:
            if not tpr_ or tpr_ == 0 or np.isnan(tpr_):
                return np.inf
            return fpr_ / tpr_

    lr_puntual = float(lr_de(wp, wb, testpos))

    rng = np.random.default_rng(seed)
    n = len(s)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        sb, wpb, wbb = s[idx], wp[idx], wb[idx]
        tb = sb >= threshold if direccion == "patho" else sb <= threshold
        v = lr_de(wpb, wbb, tb)
        if np.isfinite(v):
            boots.append(v)
    ci = np.percentile(boots, [2.5, 97.5]).tolist() if len(boots) >= 200 else [None, None]
    return {"n_ambigua": len(s), "peso_patho_total": float(wp.sum()), "peso_ben_total": float(wb.sum()),
            "lr_puntual": lr_puntual, "ic95": ci, "n_boot_validos": len(boots)}


# --------------------------------------------------------------------------
# Extrapolacion a VUS reales de MSH6/PMS2 (declarada como tal)
# --------------------------------------------------------------------------

def extrapolar_a_vus(gene, nuevo_corte_moderate, nuevo_corte_supporting_patho, nuevo_corte_supporting_ben,
                      corte_moderate_actual=0.81, corte_supporting_patho_actual=0.68,
                      corte_supporting_ben_actual=0.11):
    with open(f"/home/jesus/paper_msh6/datos/resultado_oficial_vs_esm2.json") as f:
        d = json.load(f)[gene]
    scores = [r["prior_p_oficial"] for r in d["detalle_vus"]]
    scores = np.array(scores)
    n_total = len(scores)

    def contar(corte_mod, corte_sup_p, corte_sup_b):
        n_mod = int((scores >= corte_mod).sum())
        n_sup_p = int(((scores >= corte_sup_p) & (scores < corte_mod)).sum())
        n_sup_b = int((scores <= corte_sup_b).sum())
        n_sin = n_total - n_mod - n_sup_p - n_sup_b
        return {"PP3_Moderate": n_mod, "PP3_Supporting": n_sup_p,
                "BP4_Supporting": n_sup_b, "sin_evidencia": n_sin}

    actual = contar(corte_moderate_actual, corte_supporting_patho_actual, corte_supporting_ben_actual)
    if nuevo_corte_moderate is None or nuevo_corte_supporting_patho is None or nuevo_corte_supporting_ben is None:
        recalibrado = None
    else:
        recalibrado = contar(nuevo_corte_moderate, nuevo_corte_supporting_patho, nuevo_corte_supporting_ben)
    return {"n_vus_con_prior_oficial": n_total, "distribucion_actual": actual,
            "distribucion_recalibrada": recalibrado}


# --------------------------------------------------------------------------
def main():
    real = cargar_dms_msh2()
    print(f"DMS MSH2 completo: n={len(real)}")
    post_patho, post_ben, medias, pesos = ajustar_gmm(real)

    keys_inequivoca = [k for k in real if post_patho[k] > UMBRAL_POSTERIOR or post_ben[k] > UMBRAL_POSTERIOR]
    keys_ambigua = [k for k in real if k not in set(keys_inequivoca)]
    y_hard = [1 if post_patho[k] > UMBRAL_POSTERIOR else 0 for k in keys_inequivoca]
    n_pat_dura = sum(y_hard)
    n_ben_dura = len(y_hard) - n_pat_dura
    print(f"Region inequivoca: n={len(keys_inequivoca)} ({n_pat_dura} patogenica-dura, {n_ben_dura} benigna-dura)")
    print(f"Region ambigua (numero titular): n={len(keys_ambigua)}")

    predictores = cargar_predictores_msh2()

    resultado = {"gmm_medias": medias, "gmm_pesos": pesos,
                 "n_total_dms": len(real), "n_inequivoca": len(keys_inequivoca),
                 "n_pat_dura": n_pat_dura, "n_ben_dura": n_ben_dura,
                 "n_ambigua": len(keys_ambigua), "por_predictor": {}}

    for nombre, scores in predictores.items():
        print(f"\n{'='*70}\n{nombre}\n{'='*70}")
        cov_inequivoca = [k for k in keys_inequivoca if k in scores]
        cov_ambigua = [k for k in keys_ambigua if k in scores]
        y_hard_cov = [1 if post_patho[k] > UMBRAL_POSTERIOR else 0 for k in cov_inequivoca]
        print(f"  cobertura: {len(cov_inequivoca)}/{len(keys_inequivoca)} inequivoca, "
              f"{len(cov_ambigua)}/{len(keys_ambigua)} ambigua")

        if nombre == "oficial":
            cortes = {"corte_ben_supporting": 0.11, "corte_patho_supporting": 0.68,
                      "corte_patho_moderate": 0.81, "n_pat_calibracion": None,
                      "n_ben_calibracion": None, "prior_p": None, "fijo_por_VCEP": True}
        else:
            cortes = calibrar_umbrales_predictor(scores)
            cortes["fijo_por_VCEP"] = False

        tramos = {
            "BP4_Supporting": ("benigno", cortes["corte_ben_supporting"], CORTE_SUPPORTING),
            "PP3_Supporting": ("patho", cortes["corte_patho_supporting"], CORTE_SUPPORTING),
            "PP3_Moderate": ("patho", cortes["corte_patho_moderate"], CORTE_MODERATE),
        }
        lr_duro = {}
        lr_blando = {}
        for etiqueta, (direccion, umbral, nominal) in tramos.items():
            if umbral is None:
                lr_duro[etiqueta] = None
                lr_blando[etiqueta] = None
                continue
            ld = lr_duro_con_ic(scores, cov_inequivoca, y_hard_cov, umbral, direccion,
                                 seed=RNG_SEED_BOOT + _seed_det(nombre + etiqueta) % 10000)
            lb = lr_blando_con_ic(scores, cov_ambigua, post_patho, post_ben, umbral, direccion,
                                   seed=RNG_SEED_BOOT + 1 + _seed_det(nombre + etiqueta) % 10000)
            if ld is not None:
                ld["nominal_esperado"] = nominal
                ld["alcanza_nominal_duro"] = bool(ld["lr_puntual"] >= nominal)
                ld["ic_incluye_nominal_o_menos"] = bool(
                    ld["ic95"][0] is not None and ld["ic95"][0] <= nominal)
            lb["nominal_esperado"] = nominal
            lb["alcanza_nominal_blando"] = bool(lb["lr_puntual"] >= nominal)
            lb["ic_incluye_1"] = bool(lb["ic95"][0] is not None and lb["ic95"][0] <= 1.0 <= lb["ic95"][1])
            lr_duro[etiqueta] = ld
            lr_blando[etiqueta] = lb
            print(f"  {etiqueta}: umbral_score={umbral:.4g}, nominal_LR>={nominal} | "
                  f"duro LR={ld['lr_puntual']:.3g} IC95={ld['ic95']} | "
                  f"blando(titular) LR={lb['lr_puntual']:.3g} IC95={lb['ic95']}")

        resultado["por_predictor"][nombre] = {
            "cortes": cortes, "lr_duro": lr_duro, "lr_blando_ambigua": lr_blando,
        }

    # extrapolacion solo para 'oficial' (es el unico con umbrales oficiales fijos
    # que aplican directamente a las VUS reales de MSH6/PMS2 via el mismo portal)
    print(f"\n{'='*70}\nExtrapolacion a VUS reales de MSH6/PMS2 (declarada, no medida)\n{'='*70}")
    cortes_of = resultado["por_predictor"]["oficial"]["cortes"]
    lr_titular = resultado["por_predictor"]["oficial"]["lr_blando_ambigua"]
    nuevo_mod = cortes_of["corte_patho_moderate"]
    nuevo_sup_p = cortes_of["corte_patho_supporting"]
    nuevo_sup_b = cortes_of["corte_ben_supporting"]
    extrapolacion = {}
    for gene in ["MSH6", "PMS2"]:
        extrapolacion[gene] = extrapolar_a_vus(gene, None, None, None)
        print(f"  {gene}: n_vus_con_prior_oficial={extrapolacion[gene]['n_vus_con_prior_oficial']}, "
              f"distribucion_actual={extrapolacion[gene]['distribucion_actual']}")
    resultado["extrapolacion_vus_msh6_pms2"] = extrapolacion
    resultado["nota_extrapolacion"] = (
        "Los cortes de SCORE oficiales (0.11/0.68/0.81) son fijos por la especificacion VCEP y "
        "no dependen del gen; la pregunta de si mantenerlos declarados como Moderate/Supporting "
        "esta o no respaldada por LR real se responde en MSH2 (unico gen con DMS). No se recuenta "
        "aqui una nueva distribucion de codigos porque los cortes de SCORE serian los MISMOS -- lo "
        "que cambiaria es solo la ETIQUETA de fuerza (p.ej. degradar 'PP3_Moderate' a 'PP3_Supporting' "
        "si el LR real no alcanza 4.3), no el recuento de VUS por corte de score. El recuento real que "
        "importa es cuantas VUS de MSH6/PMS2 quedan bajo cada corte de score HOY (arriba), y ese "
        "recuento no cambia con el resultado de E-RF -- lo que cambia es la fuerza que se les puede "
        "declarar honestamente."
    )

    with open("/home/jesus/paper_msh6/datos/resultado_E-RF_recalibracion_funcional.json", "w") as f:
        json.dump(resultado, f, indent=2)
    print("\nGuardado: datos/resultado_E-RF_recalibracion_funcional.json")


if __name__ == "__main__":
    main()
