"""
Corrige el ultimo hallazgo P0 de la ronda 5 (Claude opus, 12-ago-2026): los
nulos de composicion (46_ parte C) y de alta pureza (47_ parte 2) se
calcularon SOLO con ESM-2, pero el manuscrito los compara contra los 4
predictores observados en CIMRA -- exactamente el mismo tipo de error de
"mezclar predictores" que la ronda 4 corrigio para el nulo de tamano
muestral. Aqui se recalculan ambos nulos, con el balance de clase corregido
(31/51 por actividad, no 28/51 por procedencia), para los 4 predictores.
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
N_SMALLN = 2000
FRAC_DANINO_CIMRA = 31 / 51  # umbral de actividad corregido en la ronda 4/5

OBSERVADOS_CIMRA = {"oficial": 0.650, "ESM-2": 0.767, "AlphaMissense": 0.774, "ESM-1v": 0.743}


def cargar_todo_msh2():
    oficial = c40.cargar_oficial("MSH2_priors")
    esm2 = c40.cargar_esm("MSH2")
    am = c40.cargar_alphamissense("MSH2")
    esm1v = c41.cargar_esm1v("MSH2")
    with open("/home/jesus/paper_msh6/datos/dataset_H0_MSH2.json") as f:
        dms = json.load(f)
    real = {(d["posicion"], d["mut_aa"]): d["score_danino"] for d in dms}
    return {"oficial": oficial, "ESM-2": esm2, "AlphaMissense": am, "ESM-1v": esm1v}, real


def main():
    modelos, real = cargar_todo_msh2()
    resultado = {}
    rng = np.random.default_rng(SEMILLA)

    # El GMM se ajusta UNA VEZ sobre la distribucion COMPLETA del DMS real
    # (n=16.749, igual que en 44_/46_/47_), no por separado sobre el subconjunto
    # de cobertura de cada predictor -- la estructura bimodal de la verdad
    # funcional no depende de que predictor se este evaluando; solo cambia
    # que posiciones tienen score de cada modelo.
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
    print(f"GMM global: medias={means}, pesos={gmm.weights_}")

    for nombre, modelo in modelos.items():
        comunes = sorted(set(modelo) & set(real))
        y_m = np.array([modelo[k] for k in comunes])
        y_r = np.array([real[k] for k in comunes])
        post = np.array([posterior_por_key[k] for k in comunes])
        hard = np.array([hard_por_key[k] for k in comunes])
        print(f"\n{'='*70}\n{nombre} (n base={len(comunes)})\n{'='*70}")

        # nulo de composicion: asignacion dura (del GMM global), balance 31/51
        idx_b_hard = np.where(hard == comp_b)[0]
        idx_d_hard = np.where(hard == comp_d)[0]

        def muestrear(n_deseado, idx_d_pool, idx_b_pool, frac=FRAC_DANINO_CIMRA):
            n_d = min(int(round(n_deseado * frac)), len(idx_d_pool))
            n_b = min(n_deseado - n_d, len(idx_b_pool))
            sel_d = rng.choice(idx_d_pool, n_d, replace=False)
            sel_b = rng.choice(idx_b_pool, n_b, replace=False)
            return np.concatenate([sel_d, sel_b])

        rhos_comp = []
        for _ in range(N_SMALLN):
            idx_s = muestrear(51, idx_d_hard, idx_b_hard)
            r_, _ = spearmanr(y_m[idx_s], y_r[idx_s])
            if not np.isnan(r_):
                rhos_comp.append(r_)
        rhos_comp = np.array(rhos_comp)
        p975_comp = np.percentile(rhos_comp, 97.5)
        mediana_comp = float(np.median(rhos_comp))

        # nulo de alta pureza (posterior>0.999)
        idx_b_pure = np.where(post[:, comp_b] > 0.999)[0]
        idx_d_pure = np.where(post[:, comp_d] > 0.999)[0]
        rhos_pure = []
        for _ in range(N_SMALLN):
            idx_s = muestrear(51, idx_d_pure, idx_b_pure)
            r_, _ = spearmanr(y_m[idx_s], y_r[idx_s])
            if not np.isnan(r_):
                rhos_pure.append(r_)
        rhos_pure = np.array(rhos_pure)
        p975_pure = np.percentile(rhos_pure, 97.5)
        mediana_pure = float(np.median(rhos_pure))

        observado = OBSERVADOS_CIMRA[nombre]
        rechaza_comp = observado > p975_comp
        rechaza_pure = observado > p975_pure
        print(f"  Nulo composicion (balance 31/51): mediana={mediana_comp:.4f}, p97.5={p975_comp:.4f}. "
              f"Observado CIMRA={observado:.3f} -> {'RECHAZA' if rechaza_comp else 'no rechaza'} el nulo")
        print(f"  Nulo alta pureza (post>0.999): mediana={mediana_pure:.4f}, p97.5={p975_pure:.4f}. "
              f"Observado CIMRA={observado:.3f} -> {'RECHAZA' if rechaza_pure else 'no rechaza'} el nulo")

        resultado[nombre] = {
            "n_base": len(comunes),
            "composicion": {"mediana": mediana_comp, "p975": float(p975_comp),
                             "rango95": np.percentile(rhos_comp, [2.5, 97.5]).tolist(),
                             "observado_cimra": observado, "rechaza": bool(rechaza_comp)},
            "alta_pureza": {"mediana": mediana_pure, "p975": float(p975_pure),
                             "rango95": np.percentile(rhos_pure, [2.5, 97.5]).tolist(),
                             "observado_cimra": observado, "rechaza": bool(rechaza_pure),
                             "n_benigno_pureza": int(len(idx_b_pure)), "n_danino_pureza": int(len(idx_d_pure))},
        }

    n_rechaza_comp = sum(1 for v in resultado.values() if v["composicion"]["rechaza"])
    n_rechaza_pure = sum(1 for v in resultado.values() if v["alta_pureza"]["rechaza"])
    print(f"\n\nRESUMEN: {n_rechaza_comp}/4 predictores rechazan el nulo de composicion (propio); "
          f"{n_rechaza_pure}/4 rechazan el nulo de alta pureza (propio).")
    resultado["resumen"] = {"n_rechaza_composicion": n_rechaza_comp, "n_rechaza_alta_pureza": n_rechaza_pure}

    with open("/home/jesus/paper_msh6/datos/resultado_nulos_composicion_pureza_por_predictor.json", "w") as f:
        json.dump(resultado, f, indent=2)
    print("\nGuardado: datos/resultado_nulos_composicion_pureza_por_predictor.json")


if __name__ == "__main__":
    main()
