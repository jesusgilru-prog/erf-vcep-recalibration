"""
Corrige la sobre-correccion de la ronda 3, senalada por Claude opus (ronda 3 de
revision, 12-ago-2026): la explicacion "ruido de muestra pequena" para el
contraste MSH2-vs-curados esta REFUTADA por los propios datos (en PMS2, los 4
predictores superan el percentil 97,5 del nulo de submuestreo; AlphaMissense
esta a 3,3 DE de la mediana). El eje correcto no es el tamano muestral sino la
COMPOSICION: inCAMA (n=19) es una verdad casi binaria (9 benignas con oddspath
~1e-4, 10 patogenicas con oddspath 120-32.400, salto de 6 ordenes de magnitud,
1 caso limitrofe) y CIMRA (n=51) tiene un hueco vacio de disenio en la
actividad del ensayo entre 41,7 y 78,1 (28 GS de baja actividad + 23 Database
mayoritariamente de alta actividad) -- ambos son, de facto, benchmarks de
discriminacion binaria preclasificada, no muestras aleatorias del continuo.
Con verdad casi binaria, rho de Spearman se aproxima a un reescalado del AUC
(rho ~ sqrt(12)*sqrt(p*q)*(AUC-0.5) para p=P(clase positiva)), por lo que un
AUC alto (como el 0.91-0.93 ya medido dentro de MSH2) predice exactamente el
rango de rho observado en los sets curados, sin necesidad de invocar ruido de
n pequeno.

Este script implementa las pruebas decisivas y baratas que propuso Claude:
(A) AUC calculado DIRECTAMENTE sobre inCAMA y CIMRA (mismas etiquetas
    binarias/casi-binarias que definen la "verdad" del ensayo), comparado con
    el AUC intra-MSH2 ya calculado.
(B) rho en CIMRA excluyendo las 28 variantes "GS" (disenadas para calibrar el
    ensayo, no VUS reales) -- solo las 23 "Database".
(C) Diseno 2x2 composicion x tamano: se construyen submuestras del DMS de
    MSH2 con la MISMA composicion (balance ~45/55, laguna central vacia) que
    CIMRA, tanto a n=51 (como CIMRA) como a n grande (miles), para separar el
    efecto de composicion del efecto de tamano muestral. Prediccion de
    Claude: rho~0.7 en AMBOS tamanos si el eje correcto es composicion, no n.
(D) rho dentro del subconjunto ambiguo (1.465 variantes, complemento de las
    inequivocas de 44_) y dentro del modo neutro -- test directo de "falla en
    el medio ambiguo" en vez de la inferencia indirecta via AUC.
(E) Nulos de submuestreo por predictor (no solo ESM-2, como en 44_).
"""
import importlib
import json
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


def cargar_todo_msh2():
    oficial = c40.cargar_oficial("MSH2_priors")
    esm2 = c40.cargar_esm("MSH2")
    am = c40.cargar_alphamissense("MSH2")
    esm1v = c41.cargar_esm1v("MSH2")
    with open("/home/jesus/paper_msh6/datos/dataset_H0_MSH2.json") as f:
        dms = json.load(f)
    real = {(d["posicion"], d["mut_aa"]): d["score_danino"] for d in dms}
    return oficial, esm2, am, esm1v, real


def parse_op(s):
    if s is None:
        return None
    s = str(s).replace("−", "-").replace("e", "E")
    try:
        return float(s)
    except ValueError:
        return None


def parte_a_auc_directo_curados():
    print(f"\n{'='*70}\nA. AUC directo sobre inCAMA/CIMRA (verdad = clasificacion del propio ensayo)\n{'='*70}")
    with open("/home/jesus/paper_msh6/datos/CONJUNTO_VALIDACION_EXTERNA_CONGELADO.json") as f:
        frozen = json.load(f)
    c40l = c40
    c41l = c41

    resultado = {}

    # inCAMA (MSH6): B/LB vs P/LP, se excluye la unica VUS
    AA3_TO_1 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E",
                "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F",
                "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V"}
    import re

    def parse_1letter(v):
        m = re.match(r"^([A-Za-z])(\d+)([A-Za-z])$", v)
        if not m:
            return None
        wt, pos, mut = m.groups()
        return int(pos), mut.upper()

    def parse_3letter(v):
        m = re.match(r"^([A-Za-z]{3})(\d+)([A-Za-z]{3})$", v)
        if not m:
            return None
        wt3, pos, mut3 = m.groups()
        mut1 = AA3_TO_1.get(mut3.capitalize())
        if mut1 is None:
            return None
        return int(pos), mut1

    etiquetas_msh6 = {}
    for r in frozen["msh6"]:
        cls = r["clasificacion_predicha_por_inCAMA"]
        if cls in ("B", "LB"):
            label = 0
        elif cls in ("P", "LP"):
            label = 1
        else:
            continue
        parsed = parse_1letter(r["variant_1letter"])
        if parsed:
            etiquetas_msh6[parsed] = label

    oficial6 = c40l.cargar_oficial("MSH6_priors")
    esm2_6 = c40l.cargar_esm("MSH6")
    am6 = c40l.cargar_alphamissense("MSH6")
    esm1v_6 = c41l.cargar_esm1v("MSH6")
    print(f"  inCAMA: {len(etiquetas_msh6)} variantes con clasificacion B/LB o P/LP (1 VUS excluida)")
    rng = np.random.default_rng(SEMILLA)
    for nombre, modelo in [("oficial", oficial6), ("ESM-2", esm2_6), ("AlphaMissense", am6), ("ESM-1v", esm1v_6)]:
        comunes = [k for k in etiquetas_msh6 if k in modelo]
        y = np.array([etiquetas_msh6[k] for k in comunes])
        s = np.array([modelo[k] for k in comunes])
        if y.sum() < 2 or (1 - y).sum() < 2:
            print(f"  inCAMA {nombre}: insuficiente (n={len(comunes)})")
            continue
        auc = roc_auc_score(y, s)
        print(f"  inCAMA AUC {nombre}: {auc:.4f} (n={len(comunes)}, pos={int(y.sum())}, neg={int((1-y).sum())})")
        resultado.setdefault("inCAMA_MSH6", {})[nombre] = {"auc": float(auc), "n": len(comunes)}

    # CIMRA (PMS2): actividad<41.7 -> danino(1), actividad>=78.1 -> benigno(0) (hueco natural del ensayo)
    with open("/home/jesus/paper_msh6/datos/CIMRA_tabla1_parseada.json") as f:
        cimra_raw = json.load(f)
    etiquetas_pms2 = {}
    for r in cimra_raw:
        try:
            act = float(r["cimra_assay_activity"])
        except (KeyError, TypeError, ValueError):
            continue
        if act < 41.7:
            label = 1
        elif act >= 78.1:
            label = 0
        else:
            continue
        parsed = parse_3letter(r["variant_protein"].replace("p.", "").replace(" ", "").replace(".", ""))
        if parsed:
            etiquetas_pms2[parsed] = label

    oficial9 = c40l.cargar_oficial("PMS2_priors")
    esm2_9 = c40l.cargar_esm("PMS2")
    am9 = c40l.cargar_alphamissense("PMS2")
    esm1v_9 = c41l.cargar_esm1v("PMS2")
    print(f"\n  CIMRA: {len(etiquetas_pms2)} variantes con actividad fuera del hueco [41.7,78.1)")
    for nombre, modelo in [("oficial", oficial9), ("ESM-2", esm2_9), ("AlphaMissense", am9), ("ESM-1v", esm1v_9)]:
        comunes = [k for k in etiquetas_pms2 if k in modelo]
        y = np.array([etiquetas_pms2[k] for k in comunes])
        s = np.array([modelo[k] for k in comunes])
        if y.sum() < 2 or (1 - y).sum() < 2:
            print(f"  CIMRA {nombre}: insuficiente (n={len(comunes)})")
            continue
        auc = roc_auc_score(y, s)
        print(f"  CIMRA AUC {nombre}: {auc:.4f} (n={len(comunes)}, pos={int(y.sum())}, neg={int((1-y).sum())})")
        resultado.setdefault("CIMRA_PMS2", {})[nombre] = {"auc": float(auc), "n": len(comunes)}

    return resultado


def parte_b_cimra_sin_gs():
    print(f"\n{'='*70}\nB. rho en CIMRA excluyendo las 28 variantes GS (solo Database, n=23)\n{'='*70}")
    with open("/home/jesus/paper_msh6/datos/CONJUNTO_VALIDACION_EXTERNA_CONGELADO.json") as f:
        frozen = json.load(f)
    import re

    def parse_p(v):
        m = re.match(r"^p\.\s*([A-Za-z]{3})(\d+)([A-Za-z]{3})", v)
        if not m:
            return None
        AA3_TO_1 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E",
                    "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F",
                    "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V"}
        wt3, pos, mut3 = m.groups()
        mut1 = AA3_TO_1.get(mut3.capitalize())
        if mut1 is None:
            return None
        return int(pos), mut1

    verdad = {}
    database_only = set()
    for r in frozen["pms2"]:
        parsed = parse_p(r["variant_protein"])
        if parsed is None:
            continue
        try:
            verdad[parsed] = float(r["oddspath_CIMRA"])
        except (KeyError, TypeError, ValueError):
            continue
        if r["origen"] == "Database":
            database_only.add(parsed)

    print(f"  Total en el set congelado: {len(verdad)}. Solo Database: {len(database_only)}")
    oficial = c40.cargar_oficial("PMS2_priors")
    esm2 = c40.cargar_esm("PMS2")
    am = c40.cargar_alphamissense("PMS2")
    esm1v = c41.cargar_esm1v("PMS2")
    rng = np.random.default_rng(SEMILLA)
    resultado = {}
    for nombre, modelo in [("oficial", oficial), ("ESM-2", esm2), ("AlphaMissense", am), ("ESM-1v", esm1v)]:
        comunes = sorted(set(k for k in database_only if k in modelo))
        if len(comunes) < 5:
            print(f"  {nombre}: insuficiente (n={len(comunes)})")
            continue
        y1 = np.array([modelo[k] for k in comunes])
        y2 = np.array([verdad[k] for k in comunes])
        rho, p = spearmanr(y1, y2)
        boots = []
        idx = np.arange(len(comunes))
        for _ in range(N_BOOT):
            bi = rng.choice(idx, len(idx), replace=True)
            r_, _ = spearmanr(y1[bi], y2[bi])
            if not np.isnan(r_):
                boots.append(r_)
        ci = np.percentile(boots, [2.5, 97.5]).tolist() if boots else [None, None]
        print(f"  {nombre} (solo Database, n={len(comunes)}): rho={rho:.4f} (p={p:.3g}), IC95%={ci}")
        resultado[nombre] = {"rho": float(rho), "n": len(comunes), "p": float(p), "ci95": ci}
    return resultado


def parte_c_composicion_vs_tamano():
    print(f"\n{'='*70}\nC. Diseno 2x2: composicion (DMS tal cual / tipo-curado) x tamano (n=51 / grande)\n{'='*70}")
    oficial, esm2, am, esm1v, real = cargar_todo_msh2()
    comunes = sorted(set(esm2) & set(real))
    y_real = np.array([real[k] for k in comunes])
    y_esm2 = np.array([esm2[k] for k in comunes])
    n_total = len(comunes)

    gmm = GaussianMixture(n_components=2, random_state=SEMILLA, n_init=10)
    gmm.fit(y_real.reshape(-1, 1))
    hard = gmm.predict(y_real.reshape(-1, 1))
    means = gmm.means_.ravel()
    comp_benigno = int(np.argmin(means))
    comp_danino = int(np.argmax(means))
    idx_benigno = np.where(hard == comp_benigno)[0]
    idx_danino = np.where(hard == comp_danino)[0]
    print(f"  Cluster 'benigno' (media={means[comp_benigno]:.2f}): n={len(idx_benigno)}. "
          f"Cluster 'danino' (media={means[comp_danino]:.2f}): n={len(idx_danino)}.")

    frac_danino_cimra = 28 / 51  # 54.9%
    rng = np.random.default_rng(SEMILLA)

    def muestrear_tipo_curado(n_deseado):
        n_d = int(round(n_deseado * frac_danino_cimra))
        n_b = n_deseado - n_d
        n_d = min(n_d, len(idx_danino))
        n_b = min(n_b, len(idx_benigno))
        sel_d = rng.choice(idx_danino, n_d, replace=False)
        sel_b = rng.choice(idx_benigno, n_b, replace=False)
        return np.concatenate([sel_d, sel_b])

    resultado = {}
    # celda tipo-curado, n=51 (repetido)
    rhos_51 = []
    for _ in range(N_SMALLN):
        idx_s = muestrear_tipo_curado(51)
        r_, _ = spearmanr(y_esm2[idx_s], y_real[idx_s])
        if not np.isnan(r_):
            rhos_51.append(r_)
    rhos_51 = np.array(rhos_51)
    print(f"  Tipo-curado, n=51: mediana(rho)={np.median(rhos_51):.4f}, "
          f"IC95%={np.percentile(rhos_51,[2.5,97.5]).tolist()}, sd={rhos_51.std():.4f}")
    resultado["tipo_curado_n51"] = {"mediana_rho": float(np.median(rhos_51)),
                                     "ci95": np.percentile(rhos_51, [2.5, 97.5]).tolist(),
                                     "sd": float(rhos_51.std())}

    # celda tipo-curado, n grande (maximo posible manteniendo la proporcion 54.9/45.1)
    n_max_manteniendo_prop = min(int(len(idx_danino) / frac_danino_cimra),
                                  int(len(idx_benigno) / (1 - frac_danino_cimra)))
    idx_grande = muestrear_tipo_curado(n_max_manteniendo_prop)
    rho_grande, p_grande = spearmanr(y_esm2[idx_grande], y_real[idx_grande])
    boots = []
    idxs = np.arange(len(idx_grande))
    for _ in range(N_BOOT):
        bi = rng.choice(idxs, len(idxs), replace=True)
        r_, _ = spearmanr(y_esm2[idx_grande][bi], y_real[idx_grande][bi])
        if not np.isnan(r_):
            boots.append(r_)
    ci_grande = np.percentile(boots, [2.5, 97.5]).tolist()
    print(f"  Tipo-curado, n grande ({len(idx_grande)}): rho={rho_grande:.4f}, IC95%={ci_grande}")
    resultado["tipo_curado_n_grande"] = {"n": len(idx_grande), "rho": float(rho_grande), "ci95": ci_grande}

    return resultado


def parte_d_rho_dentro_de_grupos():
    print(f"\n{'='*70}\nD. rho DENTRO del grupo ambiguo y DENTRO del modo neutro (test directo del 'medio ambiguo')\n{'='*70}")
    oficial, esm2, am, esm1v, real = cargar_todo_msh2()
    comunes = sorted(set(esm2) & set(real))
    y_real = np.array([real[k] for k in comunes])
    y_esm2 = np.array([esm2[k] for k in comunes])

    gmm = GaussianMixture(n_components=2, random_state=SEMILLA, n_init=10)
    gmm.fit(y_real.reshape(-1, 1))
    post = gmm.predict_proba(y_real.reshape(-1, 1))
    max_post = post.max(axis=1)
    ambiguo = max_post <= 0.9
    hard = gmm.predict(y_real.reshape(-1, 1))
    means = gmm.means_.ravel()
    comp_neutro = int(np.argmin(means))
    neutro_inequivoco = (hard == comp_neutro) & (~ambiguo)

    rng = np.random.default_rng(SEMILLA)
    resultado = {}
    for etiqueta, mascara in [("ambiguo", ambiguo), ("neutro_inequivoco", neutro_inequivoco)]:
        idx = np.where(mascara)[0]
        rho, p = spearmanr(y_esm2[idx], y_real[idx])
        boots = []
        for _ in range(N_BOOT):
            bi = rng.choice(idx, len(idx), replace=True)
            r_, _ = spearmanr(y_esm2[bi], y_real[bi])
            if not np.isnan(r_):
                boots.append(r_)
        ci = np.percentile(boots, [2.5, 97.5]).tolist()
        print(f"  {etiqueta} (n={len(idx)}): rho={rho:.4f} (p={p:.3g}), IC95%={ci}")
        resultado[etiqueta] = {"n": len(idx), "rho": float(rho), "p": float(p), "ci95": ci}
    return resultado


def parte_e_nulos_por_predictor():
    print(f"\n{'='*70}\nE. Nulos de submuestreo n=19/51 POR PREDICTOR (no solo ESM-2)\n{'='*70}")
    oficial, esm2, am, esm1v, real = cargar_todo_msh2()
    rng = np.random.default_rng(SEMILLA)
    resultado = {}
    for nombre, modelo in [("ESM-2", esm2), ("AlphaMissense", am), ("ESM-1v", esm1v)]:
        comunes = sorted(set(modelo) & set(real))
        y_m = np.array([modelo[k] for k in comunes])
        y_r = np.array([real[k] for k in comunes])
        resultado[nombre] = {}
        for n_pequeno in (19, 51):
            rhos = []
            idx_pool = np.arange(len(comunes))
            for _ in range(N_SMALLN):
                bi = rng.choice(idx_pool, n_pequeno, replace=False)
                r_, _ = spearmanr(y_m[bi], y_r[bi])
                if not np.isnan(r_):
                    rhos.append(r_)
            rhos = np.array(rhos)
            print(f"  {nombre}, n={n_pequeno}: mediana={np.median(rhos):.4f}, "
                  f"IC95%={np.percentile(rhos,[2.5,97.5]).tolist()}")
            resultado[nombre][f"n{n_pequeno}"] = {"mediana_rho": float(np.median(rhos)),
                                                    "ci95": np.percentile(rhos, [2.5, 97.5]).tolist()}
    return resultado


def main():
    resultado = {
        "A_auc_directo_curados": parte_a_auc_directo_curados(),
        "B_cimra_sin_gs": parte_b_cimra_sin_gs(),
        "C_composicion_vs_tamano": parte_c_composicion_vs_tamano(),
        "D_rho_dentro_de_grupos": parte_d_rho_dentro_de_grupos(),
        "E_nulos_por_predictor": parte_e_nulos_por_predictor(),
    }
    with open("/home/jesus/paper_msh6/datos/resultado_composicion_no_ruido_muestral.json", "w") as f:
        json.dump(resultado, f, indent=2)
    print("\n\nGuardado: datos/resultado_composicion_no_ruido_muestral.json")


if __name__ == "__main__":
    main()
