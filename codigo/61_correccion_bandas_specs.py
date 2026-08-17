"""
Correccion de bloqueante 2 de la ronda de revision multi-IA 2026-08-17
(SINTESIS.md en papers_hub/revisiones/auto/JBI_manuscript_2026-08-17_1031):
el codigo/52_ aplicaba PP3_Supporting de GN137 (MSH2) como corte unilateral
score>=0.68, cuando la especificacion viva dice ">0.68 y <=0.81" (banda
cerrada, excluye lo que ya es PP3_Moderate). Verificado contra
datos/fuentes_primarias/cspec_all/GN137_20260811.json (y re-confirmado en
vivo el 17-ago-2026): "PP3 [Moderate] ... >0.81" / "[Supporting] ... >0.68 &
<=0.81". BP4_Supporting ("<0.11") ya era unilateral de forma correcta (solo
difiere en <0.11 estricto vs <=0.11 usado por 52_, edge case sin variantes
en el limite, no se recalcula).

Recalcula SOLO la fila PP3_Supporting de la tabla E-RF de MSH2 (oficial/
MAPP-PP2) con la banda real, reusando exactamente la misma logica de LR
(duro + blando con IC bootstrap) que codigo/52_, para que el numero sea
comparable.
"""
import importlib
import json
import sys

import numpy as np
from sklearn.mixture import GaussianMixture

sys.path.insert(0, "/home/jesus/paper_msh6/codigo")
c40 = importlib.import_module("40_comparar_alphamissense")

SEMILLA_GMM = 20260812
RNG_SEED_BOOT = 20260810
N_BOOT = 10000
UMBRAL_POSTERIOR = 0.9


def cargar_dms_msh2():
    with open("/home/jesus/paper_msh6/datos/dataset_H0_MSH2.json") as f:
        dms = json.load(f)
    return {(d["posicion"], d["mut_aa"]): d["score_danino"] for d in dms}


def ajustar_gmm(real):
    keys = sorted(real.keys())
    y = np.array([real[k] for k in keys])
    gmm = GaussianMixture(n_components=2, random_state=SEMILLA_GMM, n_init=10)
    gmm.fit(y.reshape(-1, 1))
    post = gmm.predict_proba(y.reshape(-1, 1))
    medias = gmm.means_.ravel()
    idx_patho = int(np.argmax(medias))
    idx_ben = int(np.argmin(medias))
    post_patho = {k: float(post[i, idx_patho]) for i, k in enumerate(keys)}
    post_ben = {k: float(post[i, idx_ben]) for i, k in enumerate(keys)}
    return keys, post_patho, post_ben, medias.tolist(), gmm.weights_.tolist()


def lr_banda_duro(scores, keys_inequivoca, y_hard, lo, hi, seed):
    s = np.array([scores[k] for k in keys_inequivoca])
    y = np.array(y_hard)
    testpos = (s > lo) & (s <= hi)
    n_pat, n_ben = int((y == 1).sum()), int((y == 0).sum())
    tpr = testpos[y == 1].mean()
    fpr = testpos[y == 0].mean()
    lr_puntual = float(tpr / fpr) if fpr > 0 else float("inf")
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yb, sb = y[idx], s[idx]
        if (yb == 1).sum() == 0 or (yb == 0).sum() == 0:
            continue
        tb = (sb > lo) & (sb <= hi)
        tpr_b = tb[yb == 1].mean()
        fpr_b = tb[yb == 0].mean()
        if fpr_b > 0:
            boots.append(tpr_b / fpr_b)
    ci = np.percentile(boots, [2.5, 97.5]).tolist() if len(boots) >= 200 else [None, None]
    return {"n_pat": n_pat, "n_ben": n_ben, "tpr": float(tpr), "fpr": float(fpr),
            "lr_puntual": lr_puntual, "ic95": ci, "n_boot_validos": len(boots)}


def lr_banda_blando(scores, keys_ambigua, post_patho, post_ben, lo, hi, seed):
    s = np.array([scores[k] for k in keys_ambigua])
    wp = np.array([post_patho[k] for k in keys_ambigua])
    wb = np.array([post_ben[k] for k in keys_ambigua])
    testpos = (s > lo) & (s <= hi)

    def lr_de(wp_, wb_, tp_):
        tpr_ = wp_[tp_].sum() / wp_.sum() if wp_.sum() > 0 else np.nan
        fpr_ = wb_[tp_].sum() / wb_.sum() if wb_.sum() > 0 else np.nan
        if not fpr_ or fpr_ == 0 or np.isnan(fpr_):
            return np.inf
        return tpr_ / fpr_

    lr_puntual = float(lr_de(wp, wb, testpos))
    rng = np.random.default_rng(seed)
    n = len(s)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        sb, wpb, wbb = s[idx], wp[idx], wb[idx]
        tb = (sb > lo) & (sb <= hi)
        v = lr_de(wpb, wbb, tb)
        if np.isfinite(v):
            boots.append(v)
    ci = np.percentile(boots, [2.5, 97.5]).tolist() if len(boots) >= 200 else [None, None]
    return {"n_ambigua": len(s), "peso_patho_total": float(wp.sum()), "peso_ben_total": float(wb.sum()),
            "lr_puntual": lr_puntual, "ic95": ci, "n_boot_validos": len(boots)}


def main():
    dms = cargar_dms_msh2()
    scores = c40.cargar_oficial("MSH2_priors")
    keys, post_patho, post_ben, medias, pesos = ajustar_gmm(dms)

    keys_inequivoca_pat, keys_inequivoca_ben, keys_ambigua = [], [], []
    for k in keys:
        pp, pb = post_patho[k], post_ben[k]
        if pp > UMBRAL_POSTERIOR:
            keys_inequivoca_pat.append(k)
        elif pb > UMBRAL_POSTERIOR:
            keys_inequivoca_ben.append(k)
        else:
            keys_ambigua.append(k)

    keys_inequivoca_pat = [k for k in keys_inequivoca_pat if k in scores]
    keys_inequivoca_ben = [k for k in keys_inequivoca_ben if k in scores]
    keys_ambigua_cov = [k for k in keys_ambigua if k in scores]

    keys_duro = keys_inequivoca_pat + keys_inequivoca_ben
    y_hard = [1] * len(keys_inequivoca_pat) + [0] * len(keys_inequivoca_ben)

    duro = lr_banda_duro(scores, keys_duro, y_hard, 0.68, 0.81, RNG_SEED_BOOT)
    blando = lr_banda_blando(scores, keys_ambigua_cov, post_patho, post_ben, 0.68, 0.81, RNG_SEED_BOOT)

    print("PP3_Supporting banda real (0.68, 0.81]:")
    print("  duro:", duro)
    print("  blando:", blando)

    out = {
        "correccion": "bloqueante 2 SINTESIS 2026-08-17: PP3_Supporting GN137 recalculado como "
                       "banda real (0.68, 0.81], no unilateral score>=0.68",
        "n_inequivoca_cov": len(keys_duro), "n_ambigua_cov": len(keys_ambigua_cov),
        "duro": duro, "blando": blando,
    }
    with open("/home/jesus/paper_msh6/datos/resultado_61_correccion_banda_pp3_msh2.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
