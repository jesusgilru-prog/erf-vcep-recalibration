"""
Corrige los dos hallazgos bloqueantes de la ronda 2 de revision (Claude opus,
12-ago-2026), ambos senalados como fallos de diseno del experimento, no de
redaccion:

(1) El test de ascertainment de 43_ tomaba percentiles de un z-score robusto
    sobre TODO el DMS -- eso conserva, por construccion, un subconjunto que
    sigue dominado por variantes ambiguas (top-95% de una distribucion
    continua sigue siendo ~95% continua). No podia acercarse al regimen real
    de los sets curados (95%/82% de variantes con criterio ABSOLUTO de
    inequivocidad). Aqui: (a) inequivocidad absoluta via mezcla de 2
    Gaussianas ajustada al DMS completo (post. de pertenencia a cluster
    >0.9, enfoque estandar de calibracion tipo Brnich et al. 2019), (b)
    construccion de subconjuntos que preservan la PROPORCION objetivo
    (mantener TODAS las inequivocas + solo las ambiguas justas para llegar a
    95%/82%), (c) cota superior: rho SOLO con inequivocas, (d) cruce con el
    segundo confusor identificado -- tamano muestral n=19/51 -- repitiendo
    submuestreo de cada condicion a esos tamanos (2000 repeticiones).

(2) El AUC 0.96-0.99 que sostiene "los predictores discriminan bien los
    extremos" se habia medido en MSH6/PMS2, un gen distinto del que se usa
    para medir el hueco de acierto (MSH2) -- inferencia entre genes disfrazada
    de intra-gen. Aqui: mismo AUC pareado (ESM-2, AlphaMissense, ESM-1v vs
    oficial), pero DENTRO de MSH2, contra las mismas etiquetas ClinVar P/LP
    vs B/LB ya usadas en 23_ para los otros dos genes.
"""
import csv
import importlib
import json
import re
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture

sys.path.insert(0, "/home/jesus/paper_msh6/codigo")
c23 = importlib.import_module("23_auc_pareado_esm_vs_oficial")
c40 = importlib.import_module("40_comparar_alphamissense")
c41 = importlib.import_module("41_comparar_esm1v")

SEMILLA = 20260812
N_BOOT = 2000
N_SMALLN = 2000
UMBRAL_POSTERIOR = 0.9


def cargar_todo_msh2():
    oficial = c40.cargar_oficial("MSH2_priors")
    esm2 = c40.cargar_esm("MSH2")
    am = c40.cargar_alphamissense("MSH2")
    esm1v = c41.cargar_esm1v("MSH2")
    with open("/home/jesus/paper_msh6/datos/dataset_H0_MSH2.json") as f:
        dms = json.load(f)
    real = {(d["posicion"], d["mut_aa"]): d["score_danino"] for d in dms}
    return oficial, esm2, am, esm1v, real


def bootstrap_ci_rho(y1, y2, rng, n_boot=N_BOOT):
    boots = []
    idx = np.arange(len(y1))
    for _ in range(n_boot):
        bi = rng.choice(idx, len(idx), replace=True)
        r, _ = spearmanr(y1[bi], y2[bi])
        if not np.isnan(r):
            boots.append(r)
    return np.percentile(boots, [2.5, 97.5]).tolist()


def parte1_ascertainment_corregido():
    print(f"\n{'='*70}\n1. Test de ascertainment corregido: inequivocidad ABSOLUTA + cruce con n pequeno\n{'='*70}")
    oficial, esm2, am, esm1v, real = cargar_todo_msh2()
    comunes = sorted(set(esm2) & set(real))
    keys = np.array(comunes, dtype=object)
    y_real = np.array([real[k] for k in comunes])
    y_esm2 = np.array([esm2[k] for k in comunes])
    n_total = len(comunes)
    print(f"  Base (ESM-2 x verdad real): n={n_total}")

    gmm = GaussianMixture(n_components=2, random_state=SEMILLA, n_init=10)
    gmm.fit(y_real.reshape(-1, 1))
    post = gmm.predict_proba(y_real.reshape(-1, 1))
    max_post = post.max(axis=1)
    inequivoca = max_post > UMBRAL_POSTERIOR
    frac_inequivoca = inequivoca.mean()
    medias = gmm.means_.ravel()
    print(f"  GMM 2 componentes sobre score_danino: medias={medias}, pesos={gmm.weights_}")
    print(f"  Variantes inequivocas (posterior>{UMBRAL_POSTERIOR}): {inequivoca.sum()}/{n_total} ({frac_inequivoca*100:.1f}%)")

    idx_inequivoca = np.where(inequivoca)[0]
    idx_ambigua = np.where(~inequivoca)[0]
    n_U = len(idx_inequivoca)
    n_A = len(idx_ambigua)

    resultado = {
        "n_total": n_total,
        "frac_inequivoca_gmm": float(frac_inequivoca),
        "n_inequivoca": int(n_U),
        "n_ambigua": int(n_A),
        "gmm_medias": medias.tolist(),
        "gmm_pesos": gmm.weights_.tolist(),
    }

    rng = np.random.default_rng(SEMILLA)

    rho_sin_filtrar, _ = spearmanr(y_esm2, y_real)
    ci_sin_filtrar = bootstrap_ci_rho(y_esm2, y_real, rng)
    print(f"  ESM-2 vs verdad real, SIN filtrar (n={n_total}): rho={rho_sin_filtrar:.4f}, IC95%={ci_sin_filtrar}")
    resultado["sin_filtrar"] = {"rho": float(rho_sin_filtrar), "n": n_total, "ci95": ci_sin_filtrar}

    # cota superior: SOLO inequivocas
    y_e_U, y_r_U = y_esm2[idx_inequivoca], y_real[idx_inequivoca]
    rho_U, _ = spearmanr(y_e_U, y_r_U)
    ci_U = bootstrap_ci_rho(y_e_U, y_r_U, rng)
    print(f"  SOLO inequivocas (cota superior del efecto de ascertainment), n={n_U}: rho={rho_U:.4f}, IC95%={ci_U}")
    resultado["solo_inequivocas_cota_superior"] = {"rho": float(rho_U), "n": int(n_U), "ci95": ci_U}

    condiciones = {}
    for etiqueta, frac_objetivo in [("inCAMA-like (95% inequivoca)", 0.95), ("CIMRA-like (82% inequivoca)", 0.82)]:
        # mantener TODAS las inequivocas, anadir solo las ambiguas necesarias para
        # llegar a la proporcion objetivo (no al reves)
        n_ambig_necesarias = int(round(n_U * (1 - frac_objetivo) / frac_objetivo))
        n_ambig_necesarias = min(n_ambig_necesarias, n_A)
        idx_ambig_sample = rng.choice(idx_ambigua, size=n_ambig_necesarias, replace=False)
        idx_final = np.concatenate([idx_inequivoca, idx_ambig_sample])
        y_e, y_r = y_esm2[idx_final], y_real[idx_final]
        frac_lograda = n_U / len(idx_final)
        rho_c, p_c = spearmanr(y_e, y_r)
        ci_c = bootstrap_ci_rho(y_e, y_r, rng)
        print(f"  {etiqueta}: objetivo={frac_objetivo:.0%}, lograda={frac_lograda:.1%}, "
              f"n={len(idx_final)} ({n_U} inequivocas + {n_ambig_necesarias} ambiguas), "
              f"rho={rho_c:.4f} (p={p_c:.3g}), IC95%={ci_c}")
        condiciones[etiqueta] = {
            "frac_objetivo": frac_objetivo, "frac_lograda": float(frac_lograda),
            "n": int(len(idx_final)), "n_inequivocas": int(n_U), "n_ambiguas": int(n_ambig_necesarias),
            "rho": float(rho_c), "p": float(p_c), "ci95": ci_c,
            "idx_final": idx_final.tolist(),
        }
    resultado["condiciones_proporcion_corregida"] = condiciones

    # cruce con el segundo confusor: tamano muestral n=19/51, sobre cada condicion
    print(f"\n  Cruce con el confusor de tamano muestral (n=19 como inCAMA, n=51 como CIMRA), "
          f"{N_SMALLN} repeticiones cada uno:")
    pools = {
        "DMS sin filtrar (todo, sin ascertainment)": np.arange(n_total),
        "inCAMA-like (95% inequivoca)": np.array(condiciones["inCAMA-like (95% inequivoca)"]["idx_final"]),
        "CIMRA-like (82% inequivoca)": np.array(condiciones["CIMRA-like (82% inequivoca)"]["idx_final"]),
        "solo inequivocas (cota superior)": idx_inequivoca,
    }
    cruce = {}
    for nombre_pool, idx_pool in pools.items():
        cruce[nombre_pool] = {}
        for n_pequeno in (19, 51):
            if len(idx_pool) < n_pequeno:
                continue
            rhos = []
            for _ in range(N_SMALLN):
                bi = rng.choice(idx_pool, size=n_pequeno, replace=False)
                r, _ = spearmanr(y_esm2[bi], y_real[bi])
                if not np.isnan(r):
                    rhos.append(r)
            rhos = np.array(rhos)
            mediana = float(np.median(rhos))
            ci = np.percentile(rhos, [2.5, 97.5]).tolist()
            print(f"    {nombre_pool}, n={n_pequeno}: mediana(rho)={mediana:.4f}, "
                  f"IC95% de la distribucion={ci}, sd={rhos.std():.4f}")
            cruce[nombre_pool][f"n{n_pequeno}"] = {
                "mediana_rho": mediana, "ci95_distribucion": ci, "sd": float(rhos.std()),
                "n_repeticiones": len(rhos),
            }
    resultado["cruce_tamano_muestral"] = cruce

    for cond in resultado["condiciones_proporcion_corregida"].values():
        del cond["idx_final"]

    print(f"\n  Prediccion falsable: si el ascertainment (correctamente emparejado por proporcion) "
          f"explica el contraste, rho debe subir desde {rho_sin_filtrar:.3f} hacia 0.55-0.77 SIN necesidad "
          f"de invocar el tamano muestral pequeno. Si solo sube al cruzarlo con n=19/51, el tamano "
          f"muestral es un contribuyente independiente, no un artefacto del ascertainment.")
    return resultado


def parte2_auc_msh2():
    print(f"\n{'='*70}\n2. AUC pareado DENTRO de MSH2 (ESM-2, AlphaMissense, ESM-1v vs oficial, verdad=ClinVar)\n{'='*70}")
    oficial, esm2, am, esm1v, real = cargar_todo_msh2()
    filas = c23.cargar_filas_gen("MSH2")

    etiquetados = {}
    for row in filas:
        sig = row["ClinicalSignificance"]
        if sig in c23.PATOGENICAS:
            label = 1
        elif sig in c23.BENIGNAS:
            label = 0
        else:
            continue
        parsed = c23.extraer_missense(row["Name"])
        if parsed is None:
            continue
        pos, wt_aa, mut_aa = parsed
        etiquetados[(pos, mut_aa)] = label

    modelos = {"oficial": oficial, "ESM-2": esm2, "AlphaMissense": am, "ESM-1v": esm1v}
    rng = np.random.default_rng(SEMILLA)
    resultado = {}

    for nombre, modelo in modelos.items():
        comunes = [k for k in etiquetados if k in modelo]
        y = np.array([etiquetados[k] for k in comunes])
        s = np.array([modelo[k] for k in comunes])
        n_pos, n_neg = int(y.sum()), int((1 - y).sum())
        if n_pos < 5 or n_neg < 5:
            print(f"  {nombre}: INSUFICIENTE (n_pos={n_pos}, n_neg={n_neg}, n={len(comunes)})")
            resultado[nombre] = {"n": len(comunes), "n_pos": n_pos, "n_neg": n_neg, "insuficiente": True}
            continue
        auc = roc_auc_score(y, s)
        boots = []
        idx = np.arange(len(comunes))
        for _ in range(N_BOOT):
            bi = rng.choice(idx, len(idx), replace=True)
            yb = y[bi]
            if yb.sum() < 2 or (1 - yb).sum() < 2:
                continue
            try:
                boots.append(roc_auc_score(yb, s[bi]))
            except ValueError:
                continue
        ci = np.percentile(boots, [2.5, 97.5]).tolist()
        print(f"  {nombre}: n={len(comunes)} (pat={n_pos}, ben={n_neg}), AUC={auc:.4f}, IC95%={ci}")
        resultado[nombre] = {"n": len(comunes), "n_pos": n_pos, "n_neg": n_neg,
                              "auc": float(auc), "ci95": ci}

    # DeLong pareado: oficial vs cada uno de los otros, sobre la interseccion pareada especifica
    print("\n  DeLong pareado (oficial vs cada modelo), sobre la interseccion pareada de cada par:")
    delong = {}
    for nombre, modelo in modelos.items():
        if nombre == "oficial":
            continue
        comunes = sorted(set(k for k in etiquetados if k in oficial and k in modelo))
        y = np.array([etiquetados[k] for k in comunes])
        n_pos, n_neg = int(y.sum()), int((1 - y).sum())
        if n_pos < 5 or n_neg < 5:
            continue
        s_of = np.array([oficial[k] for k in comunes])
        s_m = np.array([modelo[k] for k in comunes])
        auc_of, auc_m, diff, z, p = c23.delong_test_pareado(y, s_of, s_m)
        print(f"    oficial vs {nombre} (n={len(comunes)}): AUC_of={auc_of:.4f}, AUC_{nombre}={auc_m:.4f}, "
              f"diff={diff:+.4f}, DeLong p={p}")
        delong[nombre] = {"n": len(comunes), "auc_oficial": auc_of, "auc_otro": auc_m,
                           "diff": diff, "delong_p": p}
    resultado["delong_vs_oficial"] = delong
    return resultado


def main():
    resultado_1 = parte1_ascertainment_corregido()
    resultado_2 = parte2_auc_msh2()
    with open("/home/jesus/paper_msh6/datos/resultado_correccion_ascertainment_y_auc_msh2.json", "w") as f:
        json.dump({"1_ascertainment_corregido": resultado_1, "2_auc_msh2": resultado_2}, f, indent=2)
    print("\n\nGuardado: datos/resultado_correccion_ascertainment_y_auc_msh2.json")


if __name__ == "__main__":
    main()
