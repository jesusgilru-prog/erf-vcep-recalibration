"""
Ultimas correcciones de precision de la ronda 7 (Claude opus + Codex,
12-ago-2026), ambos con veredicto "listo en lo sustantivo" -- el mecanismo
central (composicion casi-binaria) lleva 3 rondas seguidas sin cambiar. Solo
quedan matices de precision, todos aqui:

1. El observado de AlphaMissense/CIMRA se redondeaba a 0.774 en el script 48,
   pero el valor real es 0.7735554971347479 (`resultado_experimentos_decisivos_revision.json`).
   Con el valor exacto, el margen frente al nulo de alta pureza CAMBIA DE SIGNO
   (deja de "rechazar" por los pelos). Aqui se usan los observados SIN
   REDONDEAR para las 8 celdas (CIMRA e inCAMA, los 4 predictores).
2. El balance de inCAMA usado en 49_ (10/18=55.6%, excluyendo la VUS) no
   coincide con los 19 valores reales sobre los que se calcula el rho
   observado (que SI incluye la VUS, funcionalmente benigna por OddsPath).
   Corregido a 10/19=52.6%, el balance real de las 19 variantes puntuadas.
3. El nulo de TAMANO MUESTRAL (sin composicion) tambien se recalcula a
   20.000 sorteos (antes 2.000 en 44_/46_), porque el margen de MAPP/PP2 en
   MSH6 (0.698 vs p97.5=0.688) es mas fino que el error Monte Carlo esperado
   a 2.000 sorteos.
"""
import importlib
import json
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.mixture import GaussianMixture

sys.path.insert(0, "/home/jesus/paper_msh6/codigo")
c40 = importlib.import_module("40_comparar_alphamissense")
c41 = importlib.import_module("41_comparar_esm1v")

SEMILLA = 20260812
N_ROBUSTO = 20000
FRAC_DANINO_CIMRA = 31 / 51
FRAC_PATOGENICA_INCAMA = 10 / 19  # corregido: balance real de las 19 variantes puntuadas

OBSERVADOS_CIMRA = {"oficial": 0.649579171378265, "ESM-2": 0.7667655547993151,
                     "AlphaMissense": 0.7735554971347479, "ESM-1v": 0.7432723543187173}
OBSERVADOS_INCAMA = {"oficial": 0.6982456140350877, "ESM-2": 0.7701754385964912,
                      "AlphaMissense": 0.6052631578947367, "ESM-1v": 0.5456140350877193}


def cargar_todo_msh2():
    oficial = c40.cargar_oficial("MSH2_priors")
    esm2 = c40.cargar_esm("MSH2")
    am = c40.cargar_alphamissense("MSH2")
    esm1v = c41.cargar_esm1v("MSH2")
    with open("/home/jesus/paper_msh6/datos/dataset_H0_MSH2.json") as f:
        dms = json.load(f)
    real = {(d["posicion"], d["mut_aa"]): d["score_danino"] for d in dms}
    return {"oficial": oficial, "ESM-2": esm2, "AlphaMissense": am, "ESM-1v": esm1v}, real


def nulo(y_m, y_r, idx_d_pool, idx_b_pool, n_deseado, frac, rng, n_boot=N_ROBUSTO):
    rhos = []
    for _ in range(n_boot):
        n_d = min(int(round(n_deseado * frac)), len(idx_d_pool))
        n_b = min(n_deseado - n_d, len(idx_b_pool))
        sel_d = rng.choice(idx_d_pool, n_d, replace=False)
        sel_b = rng.choice(idx_b_pool, n_b, replace=False)
        idx_s = np.concatenate([sel_d, sel_b])
        r_, _ = spearmanr(y_m[idx_s], y_r[idx_s])
        if not np.isnan(r_):
            rhos.append(r_)
    rhos = np.array(rhos)
    return {"mediana": float(np.median(rhos)), "sd": float(rhos.std()),
            "p975": float(np.percentile(rhos, 97.5)),
            "rango95": np.percentile(rhos, [2.5, 97.5]).tolist(), "n_sorteos": len(rhos)}


def nulo_tamano(y_m, y_r, n_deseado, rng, n_boot=N_ROBUSTO):
    idx_pool = np.arange(len(y_m))
    rhos = []
    for _ in range(n_boot):
        bi = rng.choice(idx_pool, n_deseado, replace=False)
        r_, _ = spearmanr(y_m[bi], y_r[bi])
        if not np.isnan(r_):
            rhos.append(r_)
    rhos = np.array(rhos)
    return {"mediana": float(np.median(rhos)), "sd": float(rhos.std()),
            "p975": float(np.percentile(rhos, 97.5)),
            "rango95": np.percentile(rhos, [2.5, 97.5]).tolist()}


def main():
    modelos, real = cargar_todo_msh2()
    todas_keys = sorted(real.keys())
    y_real_todo = np.array([real[k] for k in todas_keys])
    gmm = GaussianMixture(n_components=2, random_state=SEMILLA, n_init=10)
    gmm.fit(y_real_todo.reshape(-1, 1))
    post_todo = gmm.predict_proba(y_real_todo.reshape(-1, 1))
    hard_todo = gmm.predict(y_real_todo.reshape(-1, 1))
    means = gmm.means_.ravel()
    comp_b = int(np.argmin(means))
    comp_d = int(np.argmax(means))
    posterior_por_key = {k: post_todo[i] for i, k in enumerate(todas_keys)}
    hard_por_key = {k: hard_todo[i] for i, k in enumerate(todas_keys)}

    rng = np.random.default_rng(SEMILLA)
    resultado = {}

    for nombre, modelo in modelos.items():
        comunes = sorted(set(modelo) & set(real))
        y_m = np.array([modelo[k] for k in comunes])
        y_r = np.array([real[k] for k in comunes])
        post = np.array([posterior_por_key[k] for k in comunes])
        hard = np.array([hard_por_key[k] for k in comunes])
        idx_b_hard = np.where(hard == comp_b)[0]
        idx_d_hard = np.where(hard == comp_d)[0]
        idx_b_pure = np.where(post[:, comp_b] > 0.999)[0]
        idx_d_pure = np.where(post[:, comp_d] > 0.999)[0]

        print(f"\n{'='*70}\n{nombre} (n base={len(comunes)})\n{'='*70}")
        res_pred = {}

        for etiqueta, n_deseado, frac, observados in [
            ("CIMRA", 51, FRAC_DANINO_CIMRA, OBSERVADOS_CIMRA),
            ("inCAMA", 19, FRAC_PATOGENICA_INCAMA, OBSERVADOS_INCAMA),
        ]:
            comp = nulo(y_m, y_r, idx_d_hard, idx_b_hard, n_deseado, frac, rng)
            pure = nulo(y_m, y_r, idx_d_pure, idx_b_pure, n_deseado, frac, rng)
            tam = nulo_tamano(y_m, y_r, n_deseado, rng)
            obs = observados[nombre]
            print(f"  {etiqueta}: observado={obs:.6f}")
            print(f"    composicion: mediana={comp['mediana']:.4f} sd={comp['sd']:.4f} p975={comp['p975']:.6f} "
                  f"-> {'RECHAZA' if obs > comp['p975'] else 'no rechaza'}")
            print(f"    alta pureza: mediana={pure['mediana']:.4f} sd={pure['sd']:.4f} p975={pure['p975']:.6f} "
                  f"-> {'RECHAZA' if obs > pure['p975'] else 'no rechaza'}")
            print(f"    tamano solo: mediana={tam['mediana']:.4f} sd={tam['sd']:.4f} p975={tam['p975']:.6f} "
                  f"-> {'RECHAZA' if obs > tam['p975'] else 'no rechaza'}")
            res_pred[etiqueta] = {
                "observado": obs, "composicion": comp, "alta_pureza": pure, "tamano_solo": tam,
                "rechaza_composicion": bool(obs > comp["p975"]),
                "rechaza_pureza": bool(obs > pure["p975"]),
                "rechaza_tamano": bool(obs > tam["p975"]),
            }

        resultado[nombre] = res_pred

    with open("/home/jesus/paper_msh6/datos/resultado_50_precision_final.json", "w") as f:
        json.dump(resultado, f, indent=2)
    print("\n\nGuardado: datos/resultado_50_precision_final.json")


if __name__ == "__main__":
    main()
