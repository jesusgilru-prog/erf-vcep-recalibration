"""
Control de "range restriction" pedido explicitamente por Claude al revisar
manuscript/JBI_manuscript.tex (14-ago-2026): la region ambigua se define
sobre la INCERTIDUMBRE del propio DMS/SGE; con una correlacion predictor-DMS
solo moderada, cualquier predictor podria mostrar LR atenuado ahi por pura
estadistica (regresion-a-la-media / restriccion de rango), sin que eso
implique un fallo clinico real. Codex e independientemente Gemini pidieron
lo mismo en terminos de "es un artefacto de como se define la region ambigua".

Construccion del nulo (declarada, no escondida): para cada gen se genera un
predictor SINTETICO con la MISMA correlacion de Spearman frente al DMS/SGE
real que el predictor real (BayesDel o MAPP/PP2), pero por construccion su
fuerza de senal (relacion senal/ruido) es HOMOGENEA en todo el rango -- no
hay ninguna region donde el sintetico "sepa menos" que en otra, a diferencia
de un predictor real que podria degradarse de forma no uniforme. Se aplica
la MISMA definicion de region ambigua (mismo GMM, mismo post_patho/post_ben,
mismo umbral posterior 0.9) y se mide el LR ambiguo resultante del sintetico,
repitiendo con muchas realizaciones de ruido para obtener una distribucion
nula. Si el LR ambiguo REAL cae claramente por debajo de esa distribucion
nula, la restriccion de rango por si sola NO explica el colapso observado
-- hay un deficit real que la correlacion global no predice. Si el LR real
es indistinguible de la distribucion nula, no se puede descartar que el
colapso sea (al menos en parte) artefacto de la definicion de region.
"""
import hashlib
import importlib
import json
import sys

import numpy as np
from scipy.stats import norm, spearmanr

sys.path.insert(0, "/home/jesus/paper_msh6/codigo")
c52 = importlib.import_module("52_recalibracion_LR_funcional_erf")
c55 = importlib.import_module("55_analisis_completo_TP53")
c57 = importlib.import_module("57_ERF_BRCA1")

SEMILLA_BASE = 20260814

def _seed_det(s):
    return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16) % 10000

UMBRAL_POSTERIOR = 0.9
N_REPLICAS = 2000

GENES = {
    "MSH2": {
        "cargar_dms": c52.cargar_dms_msh2,
        "cargar_scores": lambda: c52.cargar_predictores_msh2()["oficial"],
        "umbral_score": 0.81, "semilla_gmm": 20260812,
        "lr_ambiguo_real_reportado": 1.26, "lr_inequivoco_real_reportado": 4.29,
    },
    "TP53": {
        "cargar_dms": c55.cargar_dms,
        "cargar_scores": c55.cargar_bayesdel,
        "umbral_score": 0.16, "semilla_gmm": 20260814,
        "lr_ambiguo_real_reportado": 0.97, "lr_inequivoco_real_reportado": 1.39,
    },
    "BRCA1": {
        "cargar_dms": lambda: {k: v for k, v in c57.cargar_dms_brca1().items() if c57.en_dominio(k[0])},
        "cargar_scores": c57.cargar_bayesdel_brca1,
        "umbral_score": 0.28, "semilla_gmm": 20260814,
        "lr_ambiguo_real_reportado": 1.32, "lr_inequivoco_real_reportado": 5.45,
    },
}


def gmm2(real, semilla):
    from sklearn.mixture import GaussianMixture
    keys = sorted(real.keys())
    y = np.array([real[k] for k in keys])
    gmm = GaussianMixture(n_components=2, random_state=semilla, n_init=10)
    gmm.fit(y.reshape(-1, 1))
    post = gmm.predict_proba(y.reshape(-1, 1))
    medias = gmm.means_.ravel()
    idx_patho, idx_ben = int(np.argmax(medias)), int(np.argmin(medias))
    post_patho = {k: float(post[i, idx_patho]) for i, k in enumerate(keys)}
    post_ben = {k: float(post[i, idx_ben]) for i, k in enumerate(keys)}
    return keys, y, post_patho, post_ben


def lr_punto_ambiguo(s_amb, wp_amb, wb_amb, threshold):
    testpos = s_amb >= threshold
    tpr = wp_amb[testpos].sum() / wp_amb.sum() if wp_amb.sum() > 0 else np.nan
    fpr = wb_amb[testpos].sum() / wb_amb.sum() if wb_amb.sum() > 0 else np.nan
    if not fpr or fpr == 0 or np.isnan(fpr):
        return np.inf
    return tpr / fpr


def main():
    resultado = {}
    for gen, cfg in GENES.items():
        print(f"\n{'='*70}\n{gen}\n{'='*70}")
        real = cfg["cargar_dms"]()
        scores = cfg["cargar_scores"]()
        keys, y, post_patho, post_ben = gmm2(real, cfg["semilla_gmm"])
        keys_ambigua = [k for k in keys
                        if post_patho[k] <= UMBRAL_POSTERIOR and post_ben[k] <= UMBRAL_POSTERIOR]
        keys_inequivoca = [k for k in keys if k not in set(keys_ambigua)]
        y_hard_ineq = {k: (1 if post_patho[k] > UMBRAL_POSTERIOR else 0) for k in keys_inequivoca}
        cov = [k for k in keys if k in scores]
        cov_ambigua = [k for k in keys_ambigua if k in scores]
        cov_ineq = [k for k in keys_inequivoca if k in scores]

        y_cov = np.array([real[k] for k in cov])
        s_cov = np.array([scores[k] for k in cov])
        rho_obs, pval = spearmanr(s_cov, y_cov)
        print(f"  rho(Spearman) predictor vs DMS/SGE, n={len(cov)}: {rho_obs:.3f} (p={pval:.2g})")

        # transformacion a rango normal (normal scores), para construir el
        # sintetico como modelo lineal gaussiano con la MISMA correlacion de
        # Spearman objetivo (aproximacion estandar rho_normal ~= rho_spearman)
        rank_y = np.argsort(np.argsort(y_cov)) + 1
        z_y_cov = norm.ppf(rank_y / (len(y_cov) + 1))
        z_y_by_key = dict(zip(cov, z_y_cov))

        wp_amb = np.array([post_patho[k] for k in cov_ambigua])
        wb_amb = np.array([post_ben[k] for k in cov_ambigua])
        z_y_amb = np.array([z_y_by_key[k] for k in cov_ambigua])
        y_hard_ineq_arr = np.array([y_hard_ineq[k] for k in cov_ineq])
        z_y_ineq = np.array([z_y_by_key[k] for k in cov_ineq])

        umbral_real = cfg["umbral_score"]
        s_amb_real = np.array([scores[k] for k in cov_ambigua])
        s_ineq_real = np.array([scores[k] for k in cov_ineq])
        frac_pos_real_amb = float((s_amb_real >= umbral_real).mean())
        frac_pos_real_ineq = float((s_ineq_real >= umbral_real).mean())
        print(f"  region ambigua: n={len(cov_ambigua)}, fraccion positiva (umbral REAL)={frac_pos_real_amb:.3f}")
        print(f"  region inequivoca: n={len(cov_ineq)}, fraccion positiva (umbral REAL)={frac_pos_real_ineq:.3f}")

        rng = np.random.default_rng(SEMILLA_BASE + _seed_det(gen) % 10000)
        lrs_nulo_amb, lrs_nulo_ineq = [], []
        for _ in range(N_REPLICAS):
            ruido_amb = rng.standard_normal(len(z_y_amb))
            s_sint_amb = rho_obs * z_y_amb + np.sqrt(max(1 - rho_obs**2, 0)) * ruido_amb
            umbral_sint_amb = np.quantile(s_sint_amb, 1 - frac_pos_real_amb)
            lr_a = lr_punto_ambiguo(s_sint_amb, wp_amb, wb_amb, umbral_sint_amb)
            if np.isfinite(lr_a):
                lrs_nulo_amb.append(lr_a)

            ruido_ineq = rng.standard_normal(len(z_y_ineq))
            s_sint_ineq = rho_obs * z_y_ineq + np.sqrt(max(1 - rho_obs**2, 0)) * ruido_ineq
            umbral_sint_ineq = np.quantile(s_sint_ineq, 1 - frac_pos_real_ineq)
            testpos_ineq = s_sint_ineq >= umbral_sint_ineq
            tpr = testpos_ineq[y_hard_ineq_arr == 1].mean()
            fpr = testpos_ineq[y_hard_ineq_arr == 0].mean()
            if fpr > 0:
                lrs_nulo_ineq.append(tpr / fpr)
        lrs_nulo_amb = np.array(lrs_nulo_amb)
        lrs_nulo_ineq = np.array(lrs_nulo_ineq)
        p2_5, mediana, p97_5 = np.percentile(lrs_nulo_amb, [2.5, 50, 97.5])
        p2_5_i, mediana_i, p97_5_i = np.percentile(lrs_nulo_ineq, [2.5, 50, 97.5])
        lr_real_amb = cfg["lr_ambiguo_real_reportado"]
        lr_real_ineq = cfg["lr_inequivoco_real_reportado"]
        p_valor_amb = float((lrs_nulo_amb <= lr_real_amb).mean())
        gap_real = lr_real_ineq - lr_real_amb
        gap_nulo_mediana = mediana_i - mediana
        print(f"  Nulo AMBIGUO ({len(lrs_nulo_amb)} replicas): mediana={mediana:.3g}, IC95=[{p2_5:.3g},{p97_5:.3g}]")
        print(f"  Nulo INEQUIVOCO ({len(lrs_nulo_ineq)} replicas): mediana={mediana_i:.3g}, IC95=[{p2_5_i:.3g},{p97_5_i:.3g}]")
        print(f"  LR real: inequivoco={lr_real_ineq:.3g}, ambiguo={lr_real_amb:.3g} -> GAP real={gap_real:.3g}")
        print(f"  LR nulo (mediana): inequivoco={mediana_i:.3g}, ambiguo={mediana:.3g} -> GAP nulo={gap_nulo_mediana:.3g}")
        print(f"  fraccion del nulo ambiguo <= LR real ambiguo = {p_valor_amb:.3f} "
              f"({'compatible con el nulo' if p_valor_amb > 0.05 else 'el real cae por debajo del nulo'})")

        resultado[gen] = {
            "n_overlap": len(cov), "rho_spearman_predictor_vs_dms": float(rho_obs),
            "n_ambigua": len(cov_ambigua), "n_inequivoca": len(cov_ineq),
            "frac_positiva_umbral_real_ambigua": frac_pos_real_amb,
            "frac_positiva_umbral_real_inequivoca": frac_pos_real_ineq,
            "nulo_ambiguo_lr_mediana": float(mediana), "nulo_ambiguo_lr_ic95": [float(p2_5), float(p97_5)],
            "nulo_inequivoco_lr_mediana": float(mediana_i), "nulo_inequivoco_lr_ic95": [float(p2_5_i), float(p97_5_i)],
            "lr_real_ambiguo": lr_real_amb, "lr_real_inequivoco": lr_real_ineq,
            "gap_real_ineq_menos_amb": gap_real, "gap_nulo_mediana_ineq_menos_amb": gap_nulo_mediana,
            "fraccion_nulo_ambiguo_menor_o_igual_que_real": p_valor_amb,
        }

    with open("/home/jesus/paper_msh6/datos/resultado_control_range_restriction.json", "w") as f:
        json.dump(resultado, f, indent=2)
    print("\nGuardado: datos/resultado_control_range_restriction.json")


if __name__ == "__main__":
    main()
