"""
H1 (MSH2 -> MSH6) y H2 (MLH1 -> PMS2): entrenar en el gen rico en datos DMS del
complejo y evaluar contra el conjunto de validacion externa CONGELADO (inCAMA para
MSH6, CIMRA para PMS2) -- nunca visto en entrenamiento, mirado una sola vez.

Requiere H0 superado (ver resultado_H0.json) antes de ejecutarse, por diseno del
preregistro (apartado 11, paso 4).
"""
import json
import re

import numpy as np
from scipy.stats import spearmanr
import lightgbm as lgb

from importlib import import_module
construir_mod = import_module("07_construir_dataset_H0")

RNG_SEED = 20260805
FEATURES_BASE_PLDDT = ["esm2_3B_zeroshot", "blosum62", "delta_hidrofobicidad", "delta_volumen", "plddt"]
# H1 (MSH2<->MSH6): ambos genes tienen distancias reales del co-cristal 2O8C.
FEATURES_H1 = FEATURES_BASE_PLDDT + ["dist_adn", "dist_pareja"]
# H2 (MLH1<->PMS2): sin co-cristal humano (verificado 6-ago-2026, ver 11_features_estructurales.py),
# solo pLDDT monomerico.
FEATURES_H2 = FEATURES_BASE_PLDDT

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}


def parse_float_unicode(s):
    """Los XML de Wiley/Hindawi usan el signo menos unicode (U+2212), no ascii."""
    s = s.replace("−", "-").strip()
    return float(s)


def verificar_h0_superado():
    with open("/home/jesus/paper_msh6/datos/resultado_H0.json") as f:
        r = json.load(f)
    if not r["h0_supera_nulo_empirico"]:
        raise RuntimeError(
            "H0 NO esta superado segun datos/resultado_H0.json. Por diseno del "
            "preregistro, el proyecto se detiene aqui y se publica como negativo. "
            "No se ejecuta H1/H2."
        )
    print(f"H0 verificado: rho={r['h0_msh2_spearman_rho']:.4f}, "
          f"IC=[{r['h0_msh2_spearman_ci95'][0]:.4f}, {r['h0_msh2_spearman_ci95'][1]:.4f}], "
          f"por encima del nulo empirico. Se procede a H1/H2.\n")


def cargar_esm_zeroshot(gene):
    path = f"/home/jesus/paper_msh6/datos/esm2_3B_zeroshot/{gene}_esm2_3B_zeroshot.json"
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        pos_str, aa = k.rsplit("_", 1)
        out[(int(pos_str), aa)] = v
    return out


def features_msh6_desde_frozen(entrada, esm_dict):
    variant_1letter = entrada["variant_1letter"].rstrip("g")  # sufijo de nota al pie
    m = re.match(r"^([A-Z])(\d+)([A-Z])$", variant_1letter)
    wt_aa, pos, mut_aa = m.group(1), int(m.group(2)), m.group(3)
    feats = construir_mod.features_variante(pos, wt_aa, mut_aa, esm_dict, gene="MSH6", usar_estructura=True)
    oddspath = parse_float_unicode(entrada["oddspath_functional_inCAMA"])
    return feats, np.log10(oddspath)


def features_pms2_desde_frozen(entrada, esm_dict):
    m = re.match(r"^[Pp]\.\s*([A-Za-z]{3})(\d+)([A-Za-z]{3})$", entrada["variant_protein"].strip())
    aa1_3, pos, aa2_3 = m.groups()
    wt_aa, mut_aa = AA3_TO_1[aa1_3.capitalize()], AA3_TO_1[aa2_3.capitalize()]
    feats = construir_mod.features_variante(int(pos), wt_aa, mut_aa, esm_dict, gene="PMS2", usar_estructura=True)
    oddspath = parse_float_unicode(entrada["oddspath_CIMRA"])
    return feats, np.log10(oddspath)


def entrenar(dataset_json, feature_cols):
    with open(dataset_json) as f:
        data = json.load(f)
    data = [d for d in data if all(c in d for c in feature_cols)]
    X = np.array([[d[c] for c in feature_cols] for d in data])
    y = np.array([d["score_danino"] for d in data])
    model = lgb.LGBMRegressor(n_estimators=200, num_leaves=15, random_state=RNG_SEED,
                                n_jobs=2, verbosity=-1)
    model.fit(X, y)
    return model, len(data)


def evaluar(nombre_h, modelo, X_eval, y_eval_log10oddspath, n_boot=2000, seed=RNG_SEED):
    pred = modelo.predict(X_eval)
    rho, p = spearmanr(y_eval_log10oddspath, pred)
    rng = np.random.default_rng(seed)
    n = len(y_eval_log10oddspath)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r, _ = spearmanr(y_eval_log10oddspath[idx], pred[idx])
        if not np.isnan(r):
            boot.append(r)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    # Control (c) del preregistro: robustez leave-one-out dentro del propio conjunto.
    loo_rhos = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        r, _ = spearmanr(y_eval_log10oddspath[mask], pred[mask])
        loo_rhos.append(r)
    loo_rhos = np.array(loo_rhos)

    print(f"\n=== {nombre_h} (n={n}) ===")
    print(f"Spearman rho (prediccion vs log10 OddsPath experimental) = {rho:.4f} "
          f"(p={p:.4f}), IC bootstrap 95% = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Leave-one-out: rho min={loo_rhos.min():.4f}, max={loo_rhos.max():.4f} "
          f"(rango={loo_rhos.max()-loo_rhos.min():.4f}) -- ninguna variante deberia, sola, "
          f"decidir el resultado")
    for i in range(n):
        print(f"  pred={pred[i]:+.3f}  log10(OddsPath)={y_eval_log10oddspath[i]:+.3f}")
    return {
        "rho": float(rho), "p": float(p), "ci95": [float(ci_low), float(ci_high)], "n": n,
        "loo_rho_min": float(loo_rhos.min()), "loo_rho_max": float(loo_rhos.max()),
        "pred": [float(x) for x in pred], "y": [float(x) for x in y_eval_log10oddspath],
    }


def main():
    verificar_h0_superado()

    with open("/home/jesus/paper_msh6/datos/CONJUNTO_VALIDACION_EXTERNA_CONGELADO.json") as f:
        frozen = json.load(f)

    # --- H1: MSH2 (rico) -> MSH6 (huerfano), validado contra inCAMA ---
    print(f"Entrenando H1 en MSH2 (17.746 variantes DMS), features={FEATURES_H1}...")
    modelo_msh2, n_msh2 = entrenar("/home/jesus/paper_msh6/datos/dataset_H0_MSH2.json", FEATURES_H1)
    esm_msh6 = cargar_esm_zeroshot("MSH6")
    X_msh6, y_msh6 = [], []
    omitidas_h1 = 0
    for entrada in frozen["msh6"]:
        feats, log10odds = features_msh6_desde_frozen(entrada, esm_msh6)
        if feats is None or "dist_adn" not in feats:
            omitidas_h1 += 1
            continue
        X_msh6.append([feats[c] for c in FEATURES_H1])
        y_msh6.append(log10odds)
    X_msh6, y_msh6 = np.array(X_msh6), np.array(y_msh6)
    if omitidas_h1:
        print(f"AVISO: {omitidas_h1} variantes del conjunto congelado de MSH6 omitidas "
              f"(posicion fuera del rango resuelto en 2O8C, sin dist_adn/dist_pareja)")
    resultado_h1 = evaluar("H1: MSH2 -> MSH6 (contra inCAMA, Szabo et al. 2025)",
                            modelo_msh2, X_msh6, y_msh6)

    # --- H2: MLH1 (rico) -> PMS2 (huerfano), validado contra CIMRA ---
    print(f"\nEntrenando H2 en MLH1 (4.563 variantes DMS), features={FEATURES_H2}...")
    modelo_mlh1, n_mlh1 = entrenar("/home/jesus/paper_msh6/datos/dataset_H0_MLH1.json", FEATURES_H2)
    esm_pms2 = cargar_esm_zeroshot("PMS2")
    X_pms2, y_pms2 = [], []
    for entrada in frozen["pms2"]:
        feats, log10odds = features_pms2_desde_frozen(entrada, esm_pms2)
        X_pms2.append([feats[c] for c in FEATURES_H2])
        y_pms2.append(log10odds)
    X_pms2, y_pms2 = np.array(X_pms2), np.array(y_pms2)
    resultado_h2 = evaluar("H2: MLH1 -> PMS2 (contra CIMRA, Rayner et al. 2022)",
                            modelo_mlh1, X_pms2, y_pms2)

    out = {
        "n_msh2_train": n_msh2,
        "n_mlh1_train": n_mlh1,
        "H1_msh2_a_msh6": resultado_h1,
        "H2_mlh1_a_pms2": resultado_h2,
        "H1_omitidas_sin_estructura": omitidas_h1,
        "seed": RNG_SEED,
        "features_H1": FEATURES_H1,
        "features_H2": FEATURES_H2,
        "nota": "Objetivo = log10(OddsPath experimental publicado), no ClinVar (ver D2 del preregistro).",
    }
    with open("/home/jesus/paper_msh6/datos/resultado_H1_H2.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nGuardado: datos/resultado_H1_H2.json")


if __name__ == "__main__":
    main()
