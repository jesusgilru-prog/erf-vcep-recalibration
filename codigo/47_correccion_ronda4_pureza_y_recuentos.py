"""
Corrige lo que encontro Claude opus en la ronda 4 de revision (12-ago-2026):

(1) El hueco de CIMRA por ACTIVIDAD no separa 28 GS / 23 Database: separa 31
    variantes de baja actividad (<=41.7: los 28 GS/GSa + 3 Database) de 20 de
    alta actividad (>=78.1: solo Database). Recalculado con el umbral
    correcto (<=41.7, incluye la variante limite p.Lys690Glu, activity=41.7
    exacto, que antes quedaba fuera con "<41.7").
(2) El nulo de composicion (46_, parte C) tambien es rechazado por el propio
    criterio del paper (percentil 97.5): el observado de CIMRA (rho=0.767,
    ESM-2) supera el p97.5 del nulo compuesto (0.725). La composicion
    explica ~ la mitad del exceso, no todo. Aqui se cierra del todo con el
    experimento que propuso Claude: repetir el muestreo con la MISMA
    proporcion pero restringido a miembros de ALTA PUREZA de cada cluster
    (posterior>0.999 en vez de la asignacion dura), el analogo directo del
    hueco vacio de CIMRA (que no tiene absolutamente ningun caso intermedio).
(3) Nulo de submuestreo de MAPP/PP2 (n=19/51), que faltaba en 46_ parte E
    (solo se habia hecho para ESM-2/AlphaMissense/ESM-1v).
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


def parte1_cimra_umbral_correcto():
    print(f"\n{'='*70}\n1. CIMRA con el umbral de actividad correcto (<=41.7 / >=78.1), n=31/20\n{'='*70}")
    with open("/home/jesus/paper_msh6/datos/CIMRA_tabla1_parseada.json") as f:
        cimra_raw = json.load(f)
    import re
    AA3_TO_1 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E",
                "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F",
                "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V"}

    def parse_3letter(v):
        m = re.match(r"^([A-Za-z]{3})(\d+)([A-Za-z]{3})$", v)
        if not m:
            return None
        wt3, pos, mut3 = m.groups()
        mut1 = AA3_TO_1.get(mut3.capitalize())
        if mut1 is None:
            return None
        return int(pos), mut1

    etiquetas = {}
    n_low = n_high = 0
    for r in cimra_raw:
        try:
            act = float(r["cimra_assay_activity"])
        except (KeyError, TypeError, ValueError):
            continue
        if act <= 41.7:
            label = 1
            n_low += 1
        elif act >= 78.1:
            label = 0
            n_high += 1
        else:
            continue
        parsed = parse_3letter(r["variant_protein"].replace("p.", "").replace(" ", "").replace(".", ""))
        if parsed:
            etiquetas[parsed] = label
    print(f"  Por actividad: {n_low} damaging (<=41.7), {n_high} benign (>=78.1), total={n_low+n_high}")

    oficial = c40.cargar_oficial("PMS2_priors")
    esm2 = c40.cargar_esm("PMS2")
    am = c40.cargar_alphamissense("PMS2")
    esm1v = c41.cargar_esm1v("PMS2")
    resultado = {"n_low_damaging": n_low, "n_high_benign": n_high}
    for nombre, modelo in [("oficial", oficial), ("ESM-2", esm2), ("AlphaMissense", am), ("ESM-1v", esm1v)]:
        comunes = [k for k in etiquetas if k in modelo]
        y = np.array([etiquetas[k] for k in comunes])
        s = np.array([modelo[k] for k in comunes])
        if y.sum() < 2 or (1 - y).sum() < 2:
            continue
        auc = roc_auc_score(y, s)
        print(f"  CIMRA AUC {nombre}: {auc:.4f} (n={len(comunes)}, pos={int(y.sum())}, neg={int((1-y).sum())})")
        resultado[nombre] = {"auc": float(auc), "n": len(comunes)}
    return resultado


def parte2_nulo_composicion_alta_pureza():
    print(f"\n{'='*70}\n2. Nulo de composicion restringido a ALTA PUREZA (posterior>0.999), analogo al hueco vacio\n{'='*70}")
    oficial, esm2, am, esm1v, real = cargar_todo_msh2()
    comunes = sorted(set(esm2) & set(real))
    y_real = np.array([real[k] for k in comunes])
    y_esm2 = np.array([esm2[k] for k in comunes])

    gmm = GaussianMixture(n_components=2, random_state=SEMILLA, n_init=10)
    gmm.fit(y_real.reshape(-1, 1))
    post = gmm.predict_proba(y_real.reshape(-1, 1))
    means = gmm.means_.ravel()
    comp_benigno = int(np.argmin(means))
    comp_danino = int(np.argmax(means))

    for umbral in (0.9, 0.999, 0.9999):
        alta_pureza_b = np.where(post[:, comp_benigno] > umbral)[0]
        alta_pureza_d = np.where(post[:, comp_danino] > umbral)[0]
        print(f"  umbral={umbral}: cluster benigno alta pureza n={len(alta_pureza_b)}, "
              f"danino alta pureza n={len(alta_pureza_d)}")

    idx_b = np.where(post[:, comp_benigno] > 0.999)[0]
    idx_d = np.where(post[:, comp_danino] > 0.999)[0]
    frac_danino_cimra = 31 / 51  # umbral corregido (parte 1)
    rng = np.random.default_rng(SEMILLA)

    def muestrear(n_deseado, idx_d_pool, idx_b_pool):
        n_d = min(int(round(n_deseado * frac_danino_cimra)), len(idx_d_pool))
        n_b = min(n_deseado - n_d, len(idx_b_pool))
        sel_d = rng.choice(idx_d_pool, n_d, replace=False)
        sel_b = rng.choice(idx_b_pool, n_b, replace=False)
        return np.concatenate([sel_d, sel_b])

    rhos_51 = []
    for _ in range(N_SMALLN):
        idx_s = muestrear(51, idx_d, idx_b)
        r_, _ = spearmanr(y_esm2[idx_s], y_real[idx_s])
        if not np.isnan(r_):
            rhos_51.append(r_)
    rhos_51 = np.array(rhos_51)
    print(f"  Alta pureza (posterior>0.999), n=51: mediana(rho)={np.median(rhos_51):.4f}, "
          f"rango95%={np.percentile(rhos_51,[2.5,97.5]).tolist()}")

    resultado = {
        "n_benigno_alta_pureza": int(len(idx_b)), "n_danino_alta_pureza": int(len(idx_d)),
        "mediana_rho_n51": float(np.median(rhos_51)),
        "rango95_n51": np.percentile(rhos_51, [2.5, 97.5]).tolist(),
    }
    print(f"\n  Comparacion con lo observado en CIMRA: ESM-2={0.767}, AlphaMissense={0.774}, "
          f"ESM-1v={0.743}, oficial={0.650}")
    return resultado


def parte3_nulo_mapp_pp2():
    print(f"\n{'='*70}\n3. Nulo de submuestreo n=19/51 para MAPP/PP2 (faltaba)\n{'='*70}")
    oficial, esm2, am, esm1v, real = cargar_todo_msh2()
    comunes = sorted(set(oficial) & set(real))
    y_of = np.array([oficial[k] for k in comunes])
    y_r = np.array([real[k] for k in comunes])
    rng = np.random.default_rng(SEMILLA)
    resultado = {}
    for n_pequeno in (19, 51):
        rhos = []
        idx_pool = np.arange(len(comunes))
        for _ in range(N_SMALLN):
            bi = rng.choice(idx_pool, n_pequeno, replace=False)
            r_, _ = spearmanr(y_of[bi], y_r[bi])
            if not np.isnan(r_):
                rhos.append(r_)
        rhos = np.array(rhos)
        print(f"  MAPP/PP2, n={n_pequeno}: mediana={np.median(rhos):.4f}, "
              f"rango95%={np.percentile(rhos,[2.5,97.5]).tolist()}, sd={rhos.std():.4f}")
        resultado[f"n{n_pequeno}"] = {"mediana_rho": float(np.median(rhos)),
                                       "rango95": np.percentile(rhos, [2.5, 97.5]).tolist(),
                                       "sd": float(rhos.std())}
    return resultado


def main():
    resultado = {
        "1_cimra_umbral_correcto": parte1_cimra_umbral_correcto(),
        "2_nulo_composicion_alta_pureza": parte2_nulo_composicion_alta_pureza(),
        "3_nulo_mapp_pp2": parte3_nulo_mapp_pp2(),
    }
    with open("/home/jesus/paper_msh6/datos/resultado_correccion_ronda4_pureza_y_recuentos.json", "w") as f:
        json.dump(resultado, f, indent=2)
    print("\n\nGuardado: datos/resultado_correccion_ronda4_pureza_y_recuentos.json")


if __name__ == "__main__":
    main()
