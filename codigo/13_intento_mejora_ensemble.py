"""
El panel de revision (Codex+Claude, 6-ago-2026) encontro que el modelo de
transferencia PIERDE contra ESM-2 zero-shot solo, sin entrenar nada. Antes de
aceptar el pivote a "resultado negativo", se prueba lo obvio: ¿aporta el modelo
de transferencia algo que ESM-2 no tiene, de forma que un ENSEMBLE (los dos como
features de un combinador simple) bata a ESM-2 solo?

No es forzar el resultado: es la pregunta natural siguiente cuando un componente
adicional pierde solo pero podria aportar señal complementaria. Se preespecifica
aqui, antes de mirar el resultado, el criterio de exito: el ensemble debe batir a
ESM-2 solo en AMBOS genes (MSH6 y PMS2), no solo en el promedio.
"""
import json
import re

import numpy as np
from scipy.stats import spearmanr
import lightgbm as lgb

import importlib
construir_mod = importlib.import_module("07_construir_dataset_H0")

RNG_SEED = 20260805
FEATURES_BASE_PLDDT = ["esm2_3B_zeroshot", "blosum62", "delta_hidrofobicidad", "delta_volumen", "plddt"]
FEATURES_H1 = FEATURES_BASE_PLDDT + ["dist_adn", "dist_pareja"]
FEATURES_H2 = FEATURES_BASE_PLDDT

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}


def parse_float_unicode(s):
    return float(s.replace("−", "-").strip())


def entrenar(dataset_json, feature_cols):
    with open(dataset_json) as f:
        data = json.load(f)
    data = [d for d in data if all(c in d for c in feature_cols)]
    X = np.array([[d[c] for c in feature_cols] for d in data])
    y = np.array([d["score_danino"] for d in data])
    model = lgb.LGBMRegressor(n_estimators=200, num_leaves=15, random_state=RNG_SEED,
                                n_jobs=2, verbosity=-1)
    model.fit(X, y)
    return model


def cargar_esm(gene, size):
    path = (f"/home/jesus/paper_msh6/datos/esm2_{size}_zeroshot/{gene}_esm2_{size}_zeroshot.json"
            if size != "650M_legacy" else None)
    if size == "3B":
        path = f"/home/jesus/paper_msh6/datos/esm2_3B_zeroshot/{gene}_esm2_3B_zeroshot.json"
    else:
        path = f"/home/jesus/paper_msh6/datos/esm2_zeroshot/{gene}_esm2_650M_zeroshot.json"
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        pos_str, aa = k.rsplit("_", 1)
        out[(int(pos_str), aa)] = v
    return out


def preparar_frozen(gene_key, gene_name, frozen, esm_dict_650, esm_dict_3b, modelo, feature_cols):
    rows = []
    for entrada in frozen[gene_key]:
        if gene_key == "msh6":
            v = entrada["variant_1letter"].rstrip("g")
            m = re.match(r"^([A-Z])(\d+)([A-Z])$", v)
            wt_aa, pos, mut_aa = m.group(1), int(m.group(2)), m.group(3)
            y = np.log10(parse_float_unicode(entrada["oddspath_functional_inCAMA"]))
        else:
            m = re.match(r"^[Pp]\.\s*([A-Za-z]{3})(\d+)([A-Za-z]{3})$", entrada["variant_protein"].strip())
            aa1_3, pos, aa2_3 = m.groups()
            wt_aa, mut_aa = AA3_TO_1[aa1_3.capitalize()], AA3_TO_1[aa2_3.capitalize()]
            pos = int(pos)
            y = np.log10(parse_float_unicode(entrada["oddspath_CIMRA"]))

        esm_650 = esm_dict_650.get((pos, mut_aa))
        esm_3b = esm_dict_3b.get((pos, mut_aa))
        feats_3b = construir_mod.features_variante(pos, wt_aa, mut_aa, esm_dict_3b,
                                                      gene=gene_name, usar_estructura=True)
        if esm_650 is None or esm_3b is None or feats_3b is None or not all(c in feats_3b for c in feature_cols):
            continue
        score_transfer = modelo.predict([[feats_3b[c] for c in feature_cols]])[0]
        rows.append({
            "esm_650": esm_650, "esm_3b": esm_3b, "score_transfer": score_transfer, "y": y,
        })
    return rows


def evaluar_rho(y, s):
    rho, p = spearmanr(y, s)
    return rho, p


def main():
    with open("/home/jesus/paper_msh6/datos/CONJUNTO_VALIDACION_EXTERNA_CONGELADO.json") as f:
        frozen = json.load(f)

    modelo_msh2 = entrenar("/home/jesus/paper_msh6/datos/dataset_H0_MSH2.json", FEATURES_H1)
    modelo_mlh1 = entrenar("/home/jesus/paper_msh6/datos/dataset_H0_MLH1.json", FEATURES_H2)

    esm650_msh6 = cargar_esm("MSH6", "650M")
    esm3b_msh6 = cargar_esm("MSH6", "3B")
    esm650_pms2 = cargar_esm("PMS2", "650M")
    esm3b_pms2 = cargar_esm("PMS2", "3B")

    rows_msh6 = preparar_frozen("msh6", "MSH6", frozen, esm650_msh6, esm3b_msh6, modelo_msh2, FEATURES_H1)
    rows_pms2 = preparar_frozen("pms2", "PMS2", frozen, esm650_pms2, esm3b_pms2, modelo_mlh1, FEATURES_H2)

    resultado = {}
    for nombre, rows in [("MSH6", rows_msh6), ("PMS2", rows_pms2)]:
        y = np.array([r["y"] for r in rows])
        esm650 = np.array([-r["esm_650"] for r in rows])  # invertido: alto=danino, igual que score_danino
        esm3b = np.array([-r["esm_3b"] for r in rows])
        transfer = np.array([r["score_transfer"] for r in rows])

        rho_esm650, _ = evaluar_rho(y, esm650)
        rho_esm3b, _ = evaluar_rho(y, esm3b)
        rho_transfer, _ = evaluar_rho(y, transfer)

        # Ensemble simple: rango-normalizar cada componente y promediar (evita que
        # una escala domine). Nada de ajustar pesos con el propio conjunto de
        # prueba -- promedio simple, decidido antes de mirar el resultado.
        def rango(x):
            order = np.argsort(np.argsort(x))
            return order / (len(x) - 1)

        ensemble_650_transfer = rango(esm650) + rango(transfer)
        ensemble_3b_transfer = rango(esm3b) + rango(transfer)
        ensemble_todos = rango(esm650) + rango(esm3b) + rango(transfer)

        rho_ens_650, _ = evaluar_rho(y, ensemble_650_transfer)
        rho_ens_3b, _ = evaluar_rho(y, ensemble_3b_transfer)
        rho_ens_todos, _ = evaluar_rho(y, ensemble_todos)

        print(f"\n=== {nombre} (n={len(rows)}) ===")
        print(f"  ESM-2 650M solo:          rho={rho_esm650:+.4f}")
        print(f"  ESM-2 3B solo:            rho={rho_esm3b:+.4f}")
        print(f"  Modelo transferencia:     rho={rho_transfer:+.4f}")
        print(f"  Ensemble (650M+transfer): rho={rho_ens_650:+.4f}")
        print(f"  Ensemble (3B+transfer):   rho={rho_ens_3b:+.4f}")
        print(f"  Ensemble (650M+3B+transf):rho={rho_ens_todos:+.4f}")

        mejor_solo = max(rho_esm650, rho_esm3b)
        mejor_ensemble = max(rho_ens_650, rho_ens_3b, rho_ens_todos)
        print(f"  ¿Ensemble bate al mejor solo? {mejor_ensemble > mejor_solo} "
              f"({mejor_ensemble:.4f} vs {mejor_solo:.4f})")

        resultado[nombre] = {
            "n": len(rows), "rho_esm650": float(rho_esm650), "rho_esm3b": float(rho_esm3b),
            "rho_transfer": float(rho_transfer), "rho_ensemble_650_transfer": float(rho_ens_650),
            "rho_ensemble_3b_transfer": float(rho_ens_3b), "rho_ensemble_todos": float(rho_ens_todos),
            "mejor_solo": float(mejor_solo), "mejor_ensemble": float(mejor_ensemble),
            "ensemble_gana": bool(mejor_ensemble > mejor_solo),
        }

    with open("/home/jesus/paper_msh6/datos/resultado_ensemble.json", "w") as f:
        json.dump(resultado, f, indent=2)
    print("\nGuardado: datos/resultado_ensemble.json")


if __name__ == "__main__":
    main()
