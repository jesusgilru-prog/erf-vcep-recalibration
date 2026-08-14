"""
Mecanismo de transferencia EXPLICITO (item 14 del grupo B de la Propuesta 7, nunca
probado hasta ahora pese a tener el alineamiento calculado desde el 5-ago): para
cada variante del gen huerfano, usar directamente el score medio/maximo medido en
la POSICION ALINEADA del gen rico (via Needleman-Wunsch/BLOSUM62,
codigo/05_alineamiento_paralogos.py) como feature adicional -- no un modelo
generico entrenado con features de secuencia/estructura, sino la analogia
estructural mas literal posible: "esta posicion, en el gen que si tenemos medido,
tolera poco/mucho la mutacion".

Se anade como feature EXTRA al mismo LightGBM (no sustituye nada), y se compara
otra vez contra ESM-2 solo. Si esto tampoco bate a ESM-2 solo, se acepta el
resultado negativo sin mas intentos -- ya serian 4 intentos honestos
(estructura+ESM2 3B, ensemble, intra-clase, alineamiento directo).
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
NUEVAS = ["pos_rica_score_medio", "pos_rica_score_max_abs"]

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}


def f_uni(s):
    return float(s.replace("−", "-").strip())


def construir_tabla_posicion(dataset_json):
    """Para cada posicion del gen rico: media y maximo absoluto de score_danino
    entre todas las sustituciones medidas ahi (tolerancia posicional a mutar)."""
    with open(dataset_json) as f:
        data = json.load(f)
    por_pos = {}
    for d in data:
        por_pos.setdefault(d["posicion"], []).append(d["score_danino"])
    tabla = {}
    for pos, scores in por_pos.items():
        tabla[pos] = {
            "medio": float(np.mean(scores)),
            "max_abs": float(np.max(np.abs(scores))),
        }
    return tabla


def construir_mapa_pos(alineamiento_json, par):
    with open(alineamiento_json) as f:
        d = json.load(f)
    mapa_a_a_b = d[par]["mapa"]  # pos_gen_rico(str) -> {pos_b: pos_gen_huerfano, ...}
    mapa_b_a_a = {}
    for pos_a_str, info in mapa_a_a_b.items():
        mapa_b_a_a[info["pos_b"]] = int(pos_a_str)
    return mapa_b_a_a


def anadir_features_alineamiento(dataset_json, tabla_posicion_rica, mapa_huerfano_a_rica, feature_cols):
    with open(dataset_json) as f:
        data = json.load(f)
    out = []
    for d in data:
        pos_rica = mapa_huerfano_a_rica.get(d["posicion"])
        if pos_rica is None or pos_rica not in tabla_posicion_rica:
            continue
        info = tabla_posicion_rica[pos_rica]
        d2 = dict(d)
        d2["pos_rica_score_medio"] = info["medio"]
        d2["pos_rica_score_max_abs"] = info["max_abs"]
        out.append(d2)
    return out


def entrenar(rows, feature_cols):
    rows = [r for r in rows if all(c in r for c in feature_cols)]
    X = np.array([[r[c] for c in feature_cols] for r in rows])
    y = np.array([r["score_danino"] for r in rows])
    model = lgb.LGBMRegressor(n_estimators=200, num_leaves=15, random_state=RNG_SEED,
                                n_jobs=2, verbosity=-1)
    model.fit(X, y)
    return model, len(rows)


def cargar_esm(gene, size="3B"):
    path = f"/home/jesus/paper_msh6/datos/esm2_{size}_zeroshot/{gene}_esm2_{size}_zeroshot.json"
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        pos_str, aa = k.rsplit("_", 1)
        out[(int(pos_str), aa)] = v
    return out


def evaluar_h(nombre, key, gene_huerfano, tabla_posicion_rica, mapa_huerfano_a_rica,
              modelo, feature_cols, frozen):
    esm_dict = cargar_esm(gene_huerfano)
    X, y = [], []
    omitidas = 0
    for entrada in frozen[key]:
        if key == "msh6":
            v = entrada["variant_1letter"].rstrip("g")
            m = re.match(r"^([A-Z])(\d+)([A-Z])$", v)
            wt_aa, pos, mut_aa = m.group(1), int(m.group(2)), m.group(3)
            log10odds = np.log10(f_uni(entrada["oddspath_functional_inCAMA"]))
        else:
            m = re.match(r"^[Pp]\.\s*([A-Za-z]{3})(\d+)([A-Za-z]{3})$", entrada["variant_protein"].strip())
            aa1_3, pos, aa2_3 = m.groups()
            wt_aa, mut_aa = AA3_TO_1[aa1_3.capitalize()], AA3_TO_1[aa2_3.capitalize()]
            pos = int(pos)
            log10odds = np.log10(f_uni(entrada["oddspath_CIMRA"]))

        feats = construir_mod.features_variante(pos, wt_aa, mut_aa, esm_dict,
                                                   gene=gene_huerfano, usar_estructura=True)
        pos_rica = mapa_huerfano_a_rica.get(pos)
        if feats is None or pos_rica is None or pos_rica not in tabla_posicion_rica:
            omitidas += 1
            continue
        info = tabla_posicion_rica[pos_rica]
        feats["pos_rica_score_medio"] = info["medio"]
        feats["pos_rica_score_max_abs"] = info["max_abs"]
        if not all(c in feats for c in feature_cols):
            omitidas += 1
            continue
        X.append([feats[c] for c in feature_cols])
        y.append(log10odds)
    X, y = np.array(X), np.array(y)
    pred = modelo.predict(X)
    rho, p = spearmanr(y, pred)
    print(f"{nombre}: n={len(y)} (omitidas {omitidas}), rho={rho:+.4f} (p={p:.4f})")
    return {"n": len(y), "omitidas": omitidas, "rho": float(rho), "p": float(p)}


def main():
    with open("/home/jesus/paper_msh6/datos/CONJUNTO_VALIDACION_EXTERNA_CONGELADO.json") as f:
        frozen = json.load(f)

    # --- H1: MSH2 -> MSH6 ---
    tabla_msh2 = construir_tabla_posicion("/home/jesus/paper_msh6/datos/dataset_H0_MSH2.json")
    mapa_msh6_a_msh2 = construir_mapa_pos("/home/jesus/paper_msh6/datos/alineamiento_paralogos.json", "MSH2_a_MSH6")
    rows_msh2_ext = anadir_features_alineamiento(
        "/home/jesus/paper_msh6/datos/dataset_H0_MSH2.json", tabla_msh2, mapa_msh6_a_msh2, FEATURES_H1)
    # nota: aqui el "gen rico" es MSH2 y necesitamos la tabla de posiciones del
    # PROPIO MSH2 mapeada a si mismo no tiene sentido -- la tabla_posicion_rica
    # para entrenar el modelo de MSH2 debe ser irrelevante (se usa target real).
    # Repetimos: para ENTRENAR, no anadimos la feature (no tiene sentido "score en
    # posicion alineada de si mismo"); la feature solo se anade en TEST (MSH6).
    modelo_msh2, n_msh2 = entrenar(
        [dict(d, pos_rica_score_medio=0.0, pos_rica_score_max_abs=0.0)
         for d in json.load(open("/home/jesus/paper_msh6/datos/dataset_H0_MSH2.json"))],
        FEATURES_H1 + NUEVAS)
    print(f"H1 entrenado en MSH2 ({n_msh2} var.) con features + alineamiento directo")
    r1 = evaluar_h("H1 (MSH2->MSH6, +alineamiento directo)", "msh6", "MSH6", tabla_msh2,
                    mapa_msh6_a_msh2, modelo_msh2, FEATURES_H1 + NUEVAS, frozen)

    # --- H2: MLH1 -> PMS2 ---
    tabla_mlh1 = construir_tabla_posicion("/home/jesus/paper_msh6/datos/dataset_H0_MLH1.json")
    mapa_pms2_a_mlh1 = construir_mapa_pos("/home/jesus/paper_msh6/datos/alineamiento_paralogos.json", "MLH1_a_PMS2")
    modelo_mlh1, n_mlh1 = entrenar(
        [dict(d, pos_rica_score_medio=0.0, pos_rica_score_max_abs=0.0)
         for d in json.load(open("/home/jesus/paper_msh6/datos/dataset_H0_MLH1.json"))],
        FEATURES_H2 + NUEVAS)
    print(f"H2 entrenado en MLH1 ({n_mlh1} var.) con features + alineamiento directo")
    r2 = evaluar_h("H2 (MLH1->PMS2, +alineamiento directo)", "pms2", "PMS2", tabla_mlh1,
                    mapa_pms2_a_mlh1, modelo_mlh1, FEATURES_H2 + NUEVAS, frozen)

    with open("/home/jesus/paper_msh6/datos/resultado_alineamiento_directo.json", "w") as f:
        json.dump({"H1": r1, "H2": r2}, f, indent=2)
    print("\nGuardado: datos/resultado_alineamiento_directo.json")
    print("\nComparar contra ESM-2 solo: MSH6 rho=0.6299 (3B) / 0.7549 (650M); "
          "PMS2 rho=0.8197 (3B) / 0.7668 (650M)")


if __name__ == "__main__":
    main()
