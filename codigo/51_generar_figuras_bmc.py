"""
Genera las 4 figuras del manuscrito BMC Bioinformatics, todas a partir de
datos ya verificados en disco (nada inventado):

Fig 1: distribucion del DMS de MSH2 con el ajuste de mezcla de 2 Gaussianas
       (medias/pesos ya usados en el texto: -3.52/1.02, 0.85/0.15).
Fig 2: concordancia (ESM-2 vs MAPP/PP2) frente a acierto (ESM-2 vs DMS real),
       dos paneles, n=5.193 (el mismo subconjunto de la Tabla 1).
Fig 3: curvas ROC intra-MSH2 para los 4 predictores (ClinVar P/LP vs B/LB).
Fig 4: forest plot -- acierto observado en inCAMA/CIMRA vs el rango 95% de su
       propio nulo de composicion y de alta pureza (datos de
       resultado_50_precision_final.json, ya calculado y citado en el texto).
"""
import importlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, norm
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.mixture import GaussianMixture

sys.path.insert(0, "/home/jesus/paper_msh6/codigo")
c23 = importlib.import_module("23_auc_pareado_esm_vs_oficial")
c40 = importlib.import_module("40_comparar_alphamissense")
c41 = importlib.import_module("41_comparar_esm1v")

OUTDIR = "/home/jesus/paper_msh6/manuscript/figuras"
import os
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300,
})

SEMILLA = 20260812


def cargar_todo_msh2():
    oficial = c40.cargar_oficial("MSH2_priors")
    esm2 = c40.cargar_esm("MSH2")
    am = c40.cargar_alphamissense("MSH2")
    esm1v = c41.cargar_esm1v("MSH2")
    with open("/home/jesus/paper_msh6/datos/dataset_H0_MSH2.json") as f:
        dms = json.load(f)
    real = {(d["posicion"], d["mut_aa"]): d["score_danino"] for d in dms}
    return oficial, esm2, am, esm1v, real


def fig1_distribucion_dms():
    _, _, _, _, real = cargar_todo_msh2()
    scores = np.array(list(real.values()))
    gmm = GaussianMixture(n_components=2, random_state=SEMILLA, n_init=10)
    gmm.fit(scores.reshape(-1, 1))
    means = gmm.means_.ravel()
    sds = np.sqrt(gmm.covariances_.ravel())
    weights = gmm.weights_

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores, bins=80, density=True, color="#888888", alpha=0.55, edgecolor="none",
            label="MSH2 DMS scores (n=16,749)")
    xs = np.linspace(scores.min(), scores.max(), 800)
    total = np.zeros_like(xs)
    labels = ["Component 1 (functional-like)", "Component 2 (damaging-like)"]
    order = np.argsort(means)
    colors = ["#2166ac", "#b2182b"]
    for rank, comp in enumerate(order):
        dens = weights[comp] * norm.pdf(xs, means[comp], sds[comp])
        total += dens
        ax.plot(xs, dens, color=colors[rank], lw=1.8, label=labels[rank])
    ax.plot(xs, total, color="black", lw=1.2, ls="--", label="Mixture (sum)")
    ax.set_xlabel("DMS functional score (higher = more damaging)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Bimodal distribution of MSH2 deep mutational scanning scores")
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig1_dms_distribution.pdf")
    fig.savefig(f"{OUTDIR}/fig1_dms_distribution.png")
    plt.close(fig)
    print("Fig 1 saved.")


def fig2_concordancia_vs_acierto():
    oficial, esm2, am, esm1v, real = cargar_todo_msh2()
    comun = sorted(set(oficial) & set(esm2) & set(am) & set(esm1v) & set(real))
    y_of = np.array([oficial[k] for k in comun])
    y_esm2 = np.array([esm2[k] for k in comun])
    y_real = np.array([real[k] for k in comun])

    rho_conc, _ = spearmanr(y_of, y_esm2)
    rho_acc, _ = spearmanr(y_esm2, y_real)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].scatter(y_of, y_esm2, s=4, alpha=0.25, color="#2166ac", rasterized=True)
    axes[0].set_xlabel("MAPP/PP2 Prior P")
    axes[0].set_ylabel("ESM-2 score (loss-of-function)")
    axes[0].set_title(f"(a) Concordance: MAPP/PP2 vs. ESM-2\n" + r"$\rho$" + f" = {rho_conc:.3f} (n={len(comun):,})")

    axes[1].scatter(y_esm2, y_real, s=4, alpha=0.25, color="#b2182b", rasterized=True)
    axes[1].set_xlabel("ESM-2 score (loss-of-function)")
    axes[1].set_ylabel("DMS functional score (independent reference)")
    axes[1].set_title(f"(b) Accuracy: ESM-2 vs. DMS reference\n" + r"$\rho$" + f" = {rho_acc:.3f} (n={len(comun):,})")

    fig.suptitle("Predictors agree with each other more than they match the independent functional reference (MSH2)", y=1.03, fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig2_concordance_vs_accuracy.pdf", bbox_inches="tight")
    fig.savefig(f"{OUTDIR}/fig2_concordance_vs_accuracy.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Fig 2 saved. rho_conc={rho_conc:.4f} rho_acc={rho_acc:.4f}")


def fig3_roc_intra_msh2():
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

    modelos = {"MAPP/PP2 (VCEP-cited)": oficial, "ESM-2": esm2, "AlphaMissense": am, "ESM-1v": esm1v}
    colors = {"MAPP/PP2 (VCEP-cited)": "black", "ESM-2": "#2166ac", "AlphaMissense": "#b2182b", "ESM-1v": "#1b7837"}

    fig, ax = plt.subplots(figsize=(5, 5))
    for nombre, modelo in modelos.items():
        comunes = [k for k in etiquetados if k in modelo]
        y = np.array([etiquetados[k] for k in comunes])
        s = np.array([modelo[k] for k in comunes])
        fpr, tpr, _ = roc_curve(y, s)
        auc = roc_auc_score(y, s)
        ax.plot(fpr, tpr, label=f"{nombre} (AUC={auc:.3f})", color=colors[nombre], lw=1.8)
    ax.plot([0, 1], [0, 1], color="gray", lw=1, ls="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Discrimination of ClinVar-labeled MSH2 extremes")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig3_roc_intra_msh2.pdf")
    fig.savefig(f"{OUTDIR}/fig3_roc_intra_msh2.png")
    plt.close(fig)
    print("Fig 3 saved.")


def fig4_forest_composicion():
    with open("/home/jesus/paper_msh6/datos/resultado_50_precision_final.json") as f:
        d = json.load(f)

    predictores = ["oficial", "ESM-2", "AlphaMissense", "ESM-1v"]
    etiquetas_pred = {"oficial": "MAPP/PP2", "ESM-2": "ESM-2", "AlphaMissense": "AlphaMissense", "ESM-1v": "ESM-1v"}
    genes = ["inCAMA", "CIMRA"]
    etiquetas_gen = {"inCAMA": "MSH6 (inCAMA)", "CIMRA": "PMS2 (CIMRA)"}

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)
    for ax, gene in zip(axes, genes):
        yticks = []
        yticklabels = []
        y = 0
        for pred in predictores:
            entry = d[pred][gene]
            obs = entry["observado"]
            for crit, marker, color, dy in [("composicion", "o", "#2166ac", 0.18), ("alta_pureza", "s", "#b2182b", -0.18)]:
                rango = entry[crit]["rango95"]
                yy = y + dy
                ax.plot(rango, [yy, yy], color=color, lw=3, alpha=0.5, solid_capstyle="butt")
                ax.plot(entry[crit]["mediana"], yy, marker="|", color=color, ms=10)
            ax.plot(obs, y, marker="*", color="black", ms=14, zorder=5)
            yticks.append(y)
            yticklabels.append(etiquetas_pred[pred])
            y += 1
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel(r"Spearman $\rho$")
        ax.set_title(etiquetas_gen[gene])
        ax.axvline(0, color="gray", lw=0.5)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="#2166ac", lw=3, alpha=0.5, label="Composition-matched null (95% range)"),
        Line2D([0], [0], color="#b2182b", lw=3, alpha=0.5, label="High-purity null (95% range)"),
        Line2D([0], [0], marker="*", color="black", lw=0, markersize=12, label="Observed value"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Observed accuracy vs. each predictor's own composition-matched null", y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig4_forest_composicion.pdf", bbox_inches="tight")
    fig.savefig(f"{OUTDIR}/fig4_forest_composicion.png", bbox_inches="tight")
    plt.close(fig)
    print("Fig 4 saved.")


if __name__ == "__main__":
    fig1_distribucion_dms()
    fig2_concordancia_vs_acierto()
    fig3_roc_intra_msh2()
    fig4_forest_composicion()
    print("\nAll figures saved to", OUTDIR)
