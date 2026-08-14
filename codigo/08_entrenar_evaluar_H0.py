"""
H0 (control 18 elevado, PREREGISTRO.md apartado 4): entrenar un modelo en MLH1 y
evaluar su capacidad de reproducir el mapa DMS de MSH2, oculto por completo del
entrenamiento. Prueba de fuego antes de tocar MSH6/PMS2.

Features: ESM-2 650M zero-shot (masked-marginal), BLOSUM62, delta hidrofobicidad,
delta volumen. Ninguna es especifica de parálogo — es exactamente la generalizacion
"cruda" entre familias de proteinas distintas (MutL vs MutS) que describe el control
18 original ("prediecirlo desde MLH1 y el resto").

Regla: MSH2 no se usa para entrenar ni para elegir hiperparametros. Se evalua una
sola vez.
"""
import json

import numpy as np
from scipy.stats import spearmanr
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

RNG_SEED = 20260805  # fecha del preregistro, fijada de antemano, no elegida post-hoc

FEATURE_COLS = ["esm2_3B_zeroshot", "blosum62", "delta_hidrofobicidad", "delta_volumen", "plddt"]


def cargar(gene):
    with open(f"/home/jesus/paper_msh6/datos/dataset_H0_{gene}.json") as f:
        data = json.load(f)
    X = np.array([[d[c] for c in FEATURE_COLS] for d in data])
    y = np.array([d["score_danino"] for d in data])
    return X, y, data


def evaluar_con_bootstrap(y_true, y_pred, n_boot=2000, seed=RNG_SEED):
    rho, _ = spearmanr(y_true, y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r, _ = spearmanr(y_true[idx], y_pred[idx])
        if not np.isnan(r):
            boot_rhos.append(r)
    boot_rhos = np.array(boot_rhos)
    ci_low, ci_high = np.percentile(boot_rhos, [2.5, 97.5])
    return rho, ci_low, ci_high


def nulo_empirico_por_permutacion(y_true, y_pred, n_perm=2000, seed=RNG_SEED):
    """Baraja y_true (rompe la relacion con las features) para estimar el nulo real,
    siguiendo la leccion de D13 en ARIA-Net: no asumir que el nulo esta en 0."""
    rng = np.random.default_rng(seed + 1)
    n = len(y_true)
    null_rhos = []
    for _ in range(n_perm):
        perm = rng.permutation(n)
        r, _ = spearmanr(y_true[perm], y_pred)
        if not np.isnan(r):
            null_rhos.append(r)
    return np.array(null_rhos)


def main():
    X_mlh1, y_mlh1, _ = cargar("MLH1")
    X_msh2, y_msh2, _ = cargar("MSH2")
    print(f"MLH1 (entrenamiento): {len(y_mlh1)} variantes")
    print(f"MSH2 (retenido, nunca visto en entrenamiento): {len(y_msh2)} variantes")

    # --- Validacion interna en MLH1 (5-fold), solo para saber si el modelo aprende algo en absoluto ---
    kf = KFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
    cv_rhos = []
    for train_idx, test_idx in kf.split(X_mlh1):
        model = lgb.LGBMRegressor(n_estimators=200, num_leaves=15, random_state=RNG_SEED,
                                    n_jobs=2, verbosity=-1)
        model.fit(X_mlh1[train_idx], y_mlh1[train_idx])
        pred = model.predict(X_mlh1[test_idx])
        r, _ = spearmanr(y_mlh1[test_idx], pred)
        cv_rhos.append(r)
    print(f"\nMLH1 CV interna (5-fold), Spearman rho: media={np.mean(cv_rhos):.3f}, "
          f"por fold={[round(r,3) for r in cv_rhos]}")

    # --- H0 real: entrenar en TODO MLH1, evaluar en TODO MSH2 (oculto) ---
    modelo_final = lgb.LGBMRegressor(n_estimators=200, num_leaves=15, random_state=RNG_SEED,
                                       n_jobs=2, verbosity=-1)
    modelo_final.fit(X_mlh1, y_mlh1)
    pred_msh2 = modelo_final.predict(X_msh2)

    rho, ci_low, ci_high = evaluar_con_bootstrap(y_msh2, pred_msh2)
    null_rhos = nulo_empirico_por_permutacion(y_msh2, pred_msh2)
    null_mean, null_low, null_high = null_rhos.mean(), *np.percentile(null_rhos, [2.5, 97.5])

    print(f"\n=== H0: MLH1 -> MSH2 (LightGBM, features no especificas de paralogo) ===")
    print(f"Spearman rho = {rho:.4f}, IC bootstrap 95% = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Nulo empirico (permutacion): media={null_mean:.4f}, IC 95% = [{null_low:.4f}, {null_high:.4f}]")
    supera_nulo = ci_low > null_high
    print(f"IC del rho real excluye el IC del nulo: {supera_nulo}")

    # --- Baseline trivial: solo ESM-2 zero-shot (sin BLOSUM/propiedades), Ridge lineal ---
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_mlh1[:, :1], y_mlh1)  # solo columna ESM
    pred_ridge = ridge.predict(X_msh2[:, :1])
    rho_ridge, _ = spearmanr(y_msh2, pred_ridge)
    print(f"\nBaseline (solo ESM-2 zero-shot, Ridge lineal): Spearman rho = {rho_ridge:.4f}")

    # --- Control: solo BLOSUM62 (suelo trivial, control 21 de la propuesta) ---
    rho_blosum, _ = spearmanr(y_msh2, X_msh2[:, 1])
    print(f"Control 21 (solo BLOSUM62, sin modelo): Spearman rho = {rho_blosum:.4f}")

    resultado = {
        "n_mlh1_train": len(y_mlh1),
        "n_msh2_holdout": len(y_msh2),
        "mlh1_cv_5fold_spearman_mean": float(np.mean(cv_rhos)),
        "mlh1_cv_5fold_spearman_por_fold": [float(r) for r in cv_rhos],
        "h0_msh2_spearman_rho": float(rho),
        "h0_msh2_spearman_ci95": [float(ci_low), float(ci_high)],
        "h0_msh2_nulo_empirico_media": float(null_mean),
        "h0_msh2_nulo_empirico_ci95": [float(null_low), float(null_high)],
        "h0_supera_nulo_empirico": bool(supera_nulo),
        "baseline_solo_esm_ridge_spearman": float(rho_ridge),
        "control21_solo_blosum62_spearman": float(rho_blosum),
        "seed": RNG_SEED,
        "features": FEATURE_COLS,
    }
    with open("/home/jesus/paper_msh6/datos/resultado_H0.json", "w") as f:
        json.dump(resultado, f, indent=2)
    print("\nGuardado: datos/resultado_H0.json")


if __name__ == "__main__":
    main()
