"""
Analisis de sensibilidad del hallazgo central de E-RF (LR de region ambigua),
pedido de forma CONVERGENTE por Claude, Codex y Gemini al revisar
manuscript/JBI_manuscript.tex (14-ago-2026, consulta post-reescritura 3 genes):
la region "ambigua" se define por un corte arbitrario (posterior>0.9) sobre
una mezcla gaussiana de 2 componentes NO SUPERVISADA -- ningun revisor real
dejaria pasar que el colapso a LR~1 no se haya comprobado bajo (a) otros
cortes de posterior y (b) un modelo distinto al GMM-2 componentes.

Se repiten los 3 tramos "titulares" de Table~synthesis del manuscrito:
  MSH2  -- MAPP/PP2 >=0.81 (PP3_Moderate), LR ambiguo baseline = 1.26 [1.10,1.43]
  TP53  -- BayesDel noAF >=0.16 (PP3),      LR ambiguo baseline = 0.97 [0.90,1.05]
  BRCA1 -- BayesDel noAF >=0.28 (PP3),      LR ambiguo baseline = 1.32 [0.91,1.93]

(a) Barrido de UMBRAL_POSTERIOR en {0.70, 0.80, 0.90, 0.95, 0.99} con el mismo
    GMM-2 componentes ya ajustado (semilla identica a 52_/56_/57_) -- solo
    cambia donde se traza la frontera inequivoca/ambigua sobre la MISMA
    mezcla, no se reajusta el modelo.
(b) Modelo alternativo: GMM-3 componentes (benigno / ambiguo / patogenico
    explicitos, sin corte de posterior arbitrario -- la componente central
    ES la region ambigua por construccion del modelo, argmax puro).
"""
import importlib
import json
import sys

import numpy as np
from sklearn.mixture import GaussianMixture

sys.path.insert(0, "/home/jesus/paper_msh6/codigo")
c52 = importlib.import_module("52_recalibracion_LR_funcional_erf")
c55 = importlib.import_module("55_analisis_completo_TP53")
c57 = importlib.import_module("57_ERF_BRCA1")

RNG_SEED_BOOT = 20260814
UMBRALES_POSTERIOR = [0.70, 0.80, 0.90, 0.95, 0.99]

GENES = {
    "MSH2": {
        "cargar_dms": c52.cargar_dms_msh2,
        "cargar_scores": lambda: c52.cargar_predictores_msh2()["oficial"],
        "direccion": "patho", "umbral_score": 0.81, "nominal": 4.3,
        "semilla_gmm_base": 20260812,
    },
    "TP53": {
        "cargar_dms": c55.cargar_dms,
        "cargar_scores": c55.cargar_bayesdel,
        "direccion": "patho", "umbral_score": 0.16, "nominal": 2.08,
        "semilla_gmm_base": 20260814,
    },
    "BRCA1": {
        "cargar_dms": lambda: {k: v for k, v in c57.cargar_dms_brca1().items() if c57.en_dominio(k[0])},
        "cargar_scores": c57.cargar_bayesdel_brca1,
        "direccion": "patho", "umbral_score": 0.28, "nominal": 2.08,
        "semilla_gmm_base": 20260814,
    },
}


def gmm2_con_umbral(real, umbral_post, semilla):
    keys = sorted(real.keys())
    y = np.array([real[k] for k in keys])
    gmm = GaussianMixture(n_components=2, random_state=semilla, n_init=10)
    gmm.fit(y.reshape(-1, 1))
    post = gmm.predict_proba(y.reshape(-1, 1))
    medias = gmm.means_.ravel()
    idx_patho, idx_ben = int(np.argmax(medias)), int(np.argmin(medias))
    post_patho = {k: float(post[i, idx_patho]) for i, k in enumerate(keys)}
    post_ben = {k: float(post[i, idx_ben]) for i, k in enumerate(keys)}
    keys_ambigua = [k for k in keys if post_patho[k] <= umbral_post and post_ben[k] <= umbral_post]
    return post_patho, post_ben, keys_ambigua


def gmm3_ambigua_por_argmax(real, semilla):
    """Alternativa al GMM-2+corte: mezcla de 3 componentes, la region ambigua
    es la componente central por construccion (argmax puro, sin umbral de
    posterior que fijar a mano)."""
    keys = sorted(real.keys())
    y = np.array([real[k] for k in keys])
    gmm = GaussianMixture(n_components=3, random_state=semilla, n_init=10)
    gmm.fit(y.reshape(-1, 1))
    post = gmm.predict_proba(y.reshape(-1, 1))
    medias = gmm.means_.ravel()
    orden = np.argsort(medias)  # [benigno, ambigua, patogenico] por media creciente
    idx_ben, idx_amb, idx_patho = int(orden[0]), int(orden[1]), int(orden[2])
    asignacion = np.argmax(post, axis=1)
    keys_ambigua = [k for i, k in enumerate(keys) if asignacion[i] == idx_amb]
    post_patho = {k: float(post[i, idx_patho]) for i, k in enumerate(keys)}
    post_ben = {k: float(post[i, idx_ben]) for i, k in enumerate(keys)}
    return post_patho, post_ben, keys_ambigua, medias.tolist(), gmm.weights_.tolist()


def main():
    resultado = {}
    for gen, cfg in GENES.items():
        print(f"\n{'='*70}\n{gen}\n{'='*70}")
        real = cfg["cargar_dms"]()
        scores = cfg["cargar_scores"]()
        resultado[gen] = {"barrido_umbral_posterior_gmm2": {}, "gmm3_ambigua_por_argmax": None}

        for up in UMBRALES_POSTERIOR:
            post_patho, post_ben, keys_ambigua = gmm2_con_umbral(real, up, cfg["semilla_gmm_base"])
            cov_ambigua = [k for k in keys_ambigua if k in scores]
            lb = c52.lr_blando_con_ic(
                scores, cov_ambigua, post_patho, post_ben, cfg["umbral_score"], cfg["direccion"],
                seed=RNG_SEED_BOOT + int(up * 100) + hash(gen) % 1000)
            lb["n_ambigua_total"] = len(keys_ambigua)
            lb["cobertura_predictor"] = len(cov_ambigua)
            lb["ic_incluye_1"] = bool(lb["ic95"][0] is not None and lb["ic95"][0] <= 1.0 <= lb["ic95"][1])
            resultado[gen]["barrido_umbral_posterior_gmm2"][str(up)] = lb
            print(f"  posterior>{up}: n_ambigua={len(keys_ambigua)} (cov={len(cov_ambigua)}) "
                  f"LR={lb['lr_puntual']:.3g} IC95={lb['ic95']} incluye_1={lb['ic_incluye_1']}")

        post_patho3, post_ben3, keys_amb3, medias3, pesos3 = gmm3_ambigua_por_argmax(
            real, cfg["semilla_gmm_base"] + 7000)
        cov_amb3 = [k for k in keys_amb3 if k in scores]
        lb3 = c52.lr_blando_con_ic(
            scores, cov_amb3, post_patho3, post_ben3, cfg["umbral_score"], cfg["direccion"],
            seed=RNG_SEED_BOOT + 9000 + hash(gen) % 1000)
        lb3["n_ambigua_total"] = len(keys_amb3)
        lb3["cobertura_predictor"] = len(cov_amb3)
        lb3["ic_incluye_1"] = bool(lb3["ic95"][0] is not None and lb3["ic95"][0] <= 1.0 <= lb3["ic95"][1])
        lb3["gmm3_medias"] = medias3
        lb3["gmm3_pesos"] = pesos3
        resultado[gen]["gmm3_ambigua_por_argmax"] = lb3
        print(f"  GMM-3 (componente central, argmax puro): n_ambigua={len(keys_amb3)} "
              f"(cov={len(cov_amb3)}) LR={lb3['lr_puntual']:.3g} IC95={lb3['ic95']} "
              f"incluye_1={lb3['ic_incluye_1']}")

    with open("/home/jesus/paper_msh6/datos/resultado_sensibilidad_ERF.json", "w") as f:
        json.dump(resultado, f, indent=2)
    print("\nGuardado: datos/resultado_sensibilidad_ERF.json")


if __name__ == "__main__":
    main()
