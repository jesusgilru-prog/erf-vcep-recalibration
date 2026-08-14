"""
Cierra los hallazgos de la ronda 6 (Claude opus, 12-ago-2026), todos reales:

(g) inCAMA (MSH6) no tenia NINGUN nulo de composicion/pureza -- solo el de
    tamano muestral, y ese ya mostraba que MAPP/PP2 y ESM-2 rechazan su
    propio nulo (0,698 vs p97,5=0,688; 0,770 vs p97,5=0,691), contradiciendo
    la frase "compatible with the wider MSH6 confidence intervals" (cierta
    solo para AlphaMissense/ESM-1v). Aqui se construye el nulo de composicion
    y de alta pureza para inCAMA, por predictor, igual que 48_ hizo para
    CIMRA (balance real de inCAMA: 10 patogenicas-like / 8 benignas-like de
    19, excluyendo 1 VUS -- 10/18=55,6% patogenica entre las clasificadas).
(a) El test de inequivocidad absoluta (44_) se hizo SOLO con ESM-2 pero se
    citaba como si aplicase a los 4 predictores. Aqui se repite por predictor
    (mismo GMM global ya ajustado).
(h) Robustez: nulos de composicion/pureza recalculados con 20.000 sorteos
    (antes 2.000) para que el margen de AlphaMissense en alta pureza
    (0,774 vs 0,772, muy fino) no dependa del ruido Monte Carlo del
    percentil 97,5.
(i) SD de cada nulo registrada explicitamente (antes se aproximaba desde el
    IC, no se media).
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
FRAC_PATOGENICA_INCAMA = 10 / 18  # de las 18 clasificadas (excluye 1 VUS)

OBSERVADOS_CIMRA = {"oficial": 0.650, "ESM-2": 0.767, "AlphaMissense": 0.774, "ESM-1v": 0.743}
OBSERVADOS_INCAMA = {"oficial": 0.698, "ESM-2": 0.770, "AlphaMissense": 0.605, "ESM-1v": 0.546}


def cargar_todo_msh2():
    oficial = c40.cargar_oficial("MSH2_priors")
    esm2 = c40.cargar_esm("MSH2")
    am = c40.cargar_alphamissense("MSH2")
    esm1v = c41.cargar_esm1v("MSH2")
    with open("/home/jesus/paper_msh6/datos/dataset_H0_MSH2.json") as f:
        dms = json.load(f)
    real = {(d["posicion"], d["mut_aa"]): d["score_danino"] for d in dms}
    return {"oficial": oficial, "ESM-2": esm2, "AlphaMissense": am, "ESM-1v": esm1v}, real


def nulo_composicion(y_m, y_r, idx_d_pool, idx_b_pool, n_deseado, frac, rng, n_boot=N_ROBUSTO):
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

        # --- CIMRA (n=51, balance 31/51) con 20k sorteos ---
        comp_c = nulo_composicion(y_m, y_r, idx_d_hard, idx_b_hard, 51, FRAC_DANINO_CIMRA, rng)
        pure_c = nulo_composicion(y_m, y_r, idx_d_pure, idx_b_pure, 51, FRAC_DANINO_CIMRA, rng)
        obs_c = OBSERVADOS_CIMRA[nombre]
        print(f"  CIMRA composicion: mediana={comp_c['mediana']:.4f} sd={comp_c['sd']:.4f} "
              f"p975={comp_c['p975']:.4f} | observado={obs_c} -> "
              f"{'RECHAZA' if obs_c > comp_c['p975'] else 'no rechaza'}")
        print(f"  CIMRA alta pureza: mediana={pure_c['mediana']:.4f} sd={pure_c['sd']:.4f} "
              f"p975={pure_c['p975']:.4f} | observado={obs_c} -> "
              f"{'RECHAZA' if obs_c > pure_c['p975'] else 'no rechaza'}")
        res_pred["CIMRA"] = {"composicion": comp_c, "alta_pureza": pure_c, "observado": obs_c,
                              "rechaza_composicion": bool(obs_c > comp_c["p975"]),
                              "rechaza_pureza": bool(obs_c > pure_c["p975"])}

        # --- inCAMA (n=19, balance 10/18 patogenica) ---
        comp_i = nulo_composicion(y_m, y_r, idx_d_hard, idx_b_hard, 19, FRAC_PATOGENICA_INCAMA, rng)
        pure_i = nulo_composicion(y_m, y_r, idx_d_pure, idx_b_pure, 19, FRAC_PATOGENICA_INCAMA, rng)
        obs_i = OBSERVADOS_INCAMA[nombre]
        print(f"  inCAMA composicion: mediana={comp_i['mediana']:.4f} sd={comp_i['sd']:.4f} "
              f"p975={comp_i['p975']:.4f} | observado={obs_i} -> "
              f"{'RECHAZA' if obs_i > comp_i['p975'] else 'no rechaza'}")
        print(f"  inCAMA alta pureza: mediana={pure_i['mediana']:.4f} sd={pure_i['sd']:.4f} "
              f"p975={pure_i['p975']:.4f} | observado={obs_i} -> "
              f"{'RECHAZA' if obs_i > pure_i['p975'] else 'no rechaza'}")
        res_pred["inCAMA"] = {"composicion": comp_i, "alta_pureza": pure_i, "observado": obs_i,
                               "rechaza_composicion": bool(obs_i > comp_i["p975"]),
                               "rechaza_pureza": bool(obs_i > pure_i["p975"])}

        # --- test de inequivocidad absoluta, por predictor (antes solo ESM-2) ---
        unambiguo = post[:, comp_b] > 0.9
        unambiguo |= post[:, comp_d] > 0.9
        idx_uneq = np.where(unambiguo)[0]
        rho_uneq, _ = spearmanr(y_m[idx_uneq], y_r[idx_uneq])
        rho_sin_filtrar, _ = spearmanr(y_m, y_r)
        print(f"  Inequivocidad absoluta: sin filtrar rho={rho_sin_filtrar:.4f} (n={len(comunes)}), "
              f"solo inequivocas rho={rho_uneq:.4f} (n={len(idx_uneq)})")
        res_pred["inequivocidad_absoluta"] = {
            "rho_sin_filtrar": float(rho_sin_filtrar), "n_sin_filtrar": len(comunes),
            "rho_solo_inequivocas": float(rho_uneq), "n_inequivocas": int(len(idx_uneq)),
        }

        resultado[nombre] = res_pred

    with open("/home/jesus/paper_msh6/datos/resultado_49_completo.json", "w") as f:
        json.dump(resultado, f, indent=2)
    print("\n\nGuardado: datos/resultado_49_completo.json")


if __name__ == "__main__":
    main()
