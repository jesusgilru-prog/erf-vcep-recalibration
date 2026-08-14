"""
Re-evalua los mecanismos de transferencia MSH2->MSH6 / MLH1->PMS2 (los que NO
necesitan GPU: modelo base H1/H2, ensemble, alineamiento directo -- el quinto,
LoRA fine-tuning, queda pendiente de GPU libre, ver PREREGISTRO.md) contra el
prior OFICIAL a escala (miles de variantes con MAPP/PP2 Prior P de hci-lovd),
en vez del conjunto congelado inCAMA/CIMRA (n=18/51).

Motivo (Claude, veredicto final Q1, punto no bloqueante 3): los 5 mecanismos de
transferencia se declararon "fallidos" con un n tan pequeño que el IC de
Spearman es demasiado ancho para concluir nada con solidez -- puede que no se
haya detectado señal real, no que no exista. Con miles de variantes en vez de
18/51 el negativo (o positivo, si lo hay) tiene potencia real.

Comparacion de referencia (ya calculada, `codigo/22_...py`, mismo objetivo =
prior oficial): ESM-2 650M solo, rho=0.8026 (MSH6, n=7822), rho=0.7323 (PMS2,
n=2747).
"""
import json
import re

import numpy as np
from scipy.stats import spearmanr
import lightgbm as lgb

import importlib
construir_mod = importlib.import_module("07_construir_dataset_H0")

RNG_SEED = 20260805
FEATURES_BASE = ["esm2_3B_zeroshot", "blosum62", "delta_hidrofobicidad", "delta_volumen", "plddt"]
FEATURES_H1 = FEATURES_BASE + ["dist_adn", "dist_pareja"]
FEATURES_H2 = FEATURES_BASE
NUEVAS_ALINEAMIENTO = ["pos_rica_score_medio", "pos_rica_score_max_abs"]

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}


def parse_protein_change(pc):
    m = re.match(r"^p\.([A-Za-z]{1,3})(\d+)([A-Za-z]{1,3})$", pc)
    if not m:
        return None
    aa1, pos, aa2 = m.groups()
    if len(aa1) == 3:
        aa1 = AA3_TO_1.get(aa1.capitalize())
    if len(aa2) == 3:
        aa2 = AA3_TO_1.get(aa2.capitalize())
    if aa1 is None or aa2 is None or len(aa1) != 1 or len(aa2) != 1:
        return None
    return int(pos), aa1, aa2


def cargar_oficial(select_db):
    with open(f"/home/jesus/paper_msh6/datos/{select_db}_hci_lovd.json") as f:
        raw = json.load(f)
    out = {}
    for r in raw:
        parsed = parse_protein_change(r["protein_change"])
        if parsed is None:
            continue
        pos, wt_aa, mut_aa = parsed
        out[(pos, wt_aa, mut_aa)] = r["prior_p"]
    return out


def cargar_esm(gene, size):
    dirname = "esm2_zeroshot" if size == "650M" else f"esm2_{size}_zeroshot"
    path = f"/home/jesus/paper_msh6/datos/{dirname}/{gene}_esm2_{size}_zeroshot.json"
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        pos_str, aa = k.rsplit("_", 1)
        out[(int(pos_str), aa)] = v
    return out


def entrenar_lgb(rows, feature_cols):
    rows = [r for r in rows if all(c in r for c in feature_cols)]
    X = np.array([[r[c] for c in feature_cols] for r in rows])
    y = np.array([r["score_danino"] for r in rows])
    model = lgb.LGBMRegressor(n_estimators=200, num_leaves=15, random_state=RNG_SEED,
                                n_jobs=2, verbosity=-1)
    model.fit(X, y)
    return model, len(rows)


def construir_tabla_posicion(dataset_json):
    with open(dataset_json) as f:
        data = json.load(f)
    por_pos = {}
    for d in data:
        por_pos.setdefault(d["posicion"], []).append(d["score_danino"])
    return {pos: {"medio": float(np.mean(s)), "max_abs": float(np.max(np.abs(s)))}
            for pos, s in por_pos.items()}


def construir_mapa_pos(par):
    with open("/home/jesus/paper_msh6/datos/alineamiento_paralogos.json") as f:
        d = json.load(f)
    mapa_a_a_b = d[par]["mapa"]
    return {info["pos_b"]: int(pos_a_str) for pos_a_str, info in mapa_a_a_b.items()}


def rango(x):
    order = np.argsort(np.argsort(x))
    return order / (len(x) - 1)


def evaluar_gen(nombre, gene_rico, dataset_rico_json, gene_huerfano, select_db_oficial,
                 feature_cols, par_alineamiento):
    print(f"\n{'='*70}\n{nombre}: {gene_rico} -> {gene_huerfano}, contra prior oficial a escala\n{'='*70}")
    oficial = cargar_oficial(select_db_oficial)
    esm3b = cargar_esm(gene_huerfano, "3B")
    esm650 = cargar_esm(gene_huerfano, "650M")

    # --- mecanismo 1: modelo base de transferencia (estructura + ESM-2 3B) ---
    modelo_base, n_train = entrenar_lgb(
        [dict(d) for d in json.load(open(dataset_rico_json))], feature_cols)
    print(f"Modelo base entrenado en {gene_rico}: {n_train} variantes")

    # --- mecanismo 2: alineamiento directo (feature extra: tolerancia posicional en el gen rico) ---
    tabla_rica = construir_tabla_posicion(dataset_rico_json)
    mapa_huerfano_a_rica = construir_mapa_pos(par_alineamiento)
    modelo_align, n_train_align = entrenar_lgb(
        [dict(d, pos_rica_score_medio=0.0, pos_rica_score_max_abs=0.0)
         for d in json.load(open(dataset_rico_json))],
        feature_cols + NUEVAS_ALINEAMIENTO)

    filas = []
    omitidas_estructura = omitidas_alineamiento = 0
    for (pos, wt_aa, mut_aa), prior_p in oficial.items():
        feats = construir_mod.features_variante(pos, wt_aa, mut_aa, esm3b,
                                                   gene=gene_huerfano, usar_estructura=True)
        e650 = esm650.get((pos, mut_aa))
        if feats is None or e650 is None:
            omitidas_estructura += 1
            continue
        fila = dict(feats, prior_p=prior_p, esm650=e650)
        pos_rica = mapa_huerfano_a_rica.get(pos)
        if pos_rica is not None and pos_rica in tabla_rica:
            info = tabla_rica[pos_rica]
            fila["pos_rica_score_medio"] = info["medio"]
            fila["pos_rica_score_max_abs"] = info["max_abs"]
        else:
            omitidas_alineamiento += 1
        filas.append(fila)

    n_antes_filtro_base = len(filas)
    filas = [f for f in filas if all(c in f for c in feature_cols)]
    print(f"Variantes con prior oficial y features de estructura: {len(filas)}/{len(oficial)} "
          f"({omitidas_estructura} omitidas por falta de estructura/ESM-2, "
          f"{n_antes_filtro_base - len(filas)} omitidas por falta de {feature_cols})")
    print(f"De esas, con alineamiento directo disponible: "
          f"{sum(1 for f in filas if 'pos_rica_score_medio' in f)}")

    X_base = np.array([[f[c] for c in feature_cols] for f in filas])
    pred_base = modelo_base.predict(X_base)
    y = np.array([f["prior_p"] for f in filas])
    # Convencion verificada en 22_: el valor CRUDO de zero-shot (masked-marginal) tiene
    # alto=MENOS danino: hay que negarlo para que alto=mas danino, igual que prior_p.
    esm650_arr = np.array([-f["esm650"] for f in filas])
    esm3b_arr = np.array([-f["esm2_3B_zeroshot"] for f in filas])

    rho_base, p_base = spearmanr(y, pred_base)
    rho_esm650, _ = spearmanr(y, esm650_arr)
    rho_esm3b, _ = spearmanr(y, esm3b_arr)

    ens = rango(esm650_arr) + rango(pred_base)
    rho_ens, _ = spearmanr(y, ens)

    filas_align = [f for f in filas if "pos_rica_score_medio" in f]
    X_align = np.array([[f[c] for c in feature_cols + NUEVAS_ALINEAMIENTO] for f in filas_align])
    pred_align = modelo_align.predict(X_align)
    y_align = np.array([f["prior_p"] for f in filas_align])
    rho_align, p_align = spearmanr(y_align, pred_align)

    # intra-clase: dividir por prior_p (>0.5 = lado patogenico, <0.5 = lado benigno)
    intraclase = {}
    for etiqueta, mask in [("lado_patogenico (prior>0.5)", y > 0.5), ("lado_benigno (prior<=0.5)", y <= 0.5)]:
        n_m = int(mask.sum())
        if n_m < 10:
            continue
        r_base_m, _ = spearmanr(y[mask], pred_base[mask])
        r_esm_m, _ = spearmanr(y[mask], esm650_arr[mask])
        intraclase[etiqueta] = {"n": n_m, "rho_transfer": float(r_base_m), "rho_esm650_solo": float(r_esm_m)}

    print(f"\nESM-2 650M solo:              rho={rho_esm650:+.4f}")
    print(f"ESM-2 3B solo:                 rho={rho_esm3b:+.4f}")
    print(f"Modelo de transferencia base:  rho={rho_base:+.4f} (p={p_base:.3g}, n={len(filas)})")
    print(f"Ensemble (650M + transfer):    rho={rho_ens:+.4f}")
    print(f"Alineamiento directo:          rho={rho_align:+.4f} (p={p_align:.3g}, n={len(filas_align)})")
    print("Intra-clase:")
    for k, v in intraclase.items():
        print(f"  {k}: n={v['n']}, transfer={v['rho_transfer']:+.4f}, esm650_solo={v['rho_esm650_solo']:+.4f}")

    mejor_mecanismo_transferencia = max(rho_base, rho_ens, rho_align)
    mejor_solo = max(rho_esm650, rho_esm3b)
    print(f"\n¿Algun mecanismo de transferencia bate a ESM-2 solo, A ESCALA? "
          f"{mejor_mecanismo_transferencia > mejor_solo} "
          f"({mejor_mecanismo_transferencia:.4f} vs {mejor_solo:.4f})")

    return {
        "gene_huerfano": gene_huerfano, "n": len(filas), "n_align": len(filas_align),
        "rho_esm650_solo": float(rho_esm650), "rho_esm3b_solo": float(rho_esm3b),
        "rho_modelo_transferencia_base": float(rho_base), "p_base": float(p_base),
        "rho_ensemble": float(rho_ens),
        "rho_alineamiento_directo": float(rho_align), "p_align": float(p_align),
        "intraclase": intraclase,
        "mejor_mecanismo_transferencia": float(mejor_mecanismo_transferencia),
        "mejor_solo": float(mejor_solo),
        "transferencia_bate_a_esm_solo": bool(mejor_mecanismo_transferencia > mejor_solo),
    }


def main():
    resultado = {}
    resultado["H1_MSH6"] = evaluar_gen(
        "H1", "MSH2", "/home/jesus/paper_msh6/datos/dataset_H0_MSH2.json", "MSH6",
        "MSH6_priors", FEATURES_H1, "MSH2_a_MSH6")
    resultado["H2_PMS2"] = evaluar_gen(
        "H2", "MLH1", "/home/jesus/paper_msh6/datos/dataset_H0_MLH1.json", "PMS2",
        "PMS2_priors", FEATURES_H2, "MLH1_a_PMS2")

    with open("/home/jesus/paper_msh6/datos/resultado_reevaluacion_transferencia_escala.json", "w") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    print("\n\nGuardado: datos/resultado_reevaluacion_transferencia_escala.json")


if __name__ == "__main__":
    main()
