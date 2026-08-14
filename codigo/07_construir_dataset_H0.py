"""
Construye el dataset de entrenamiento/evaluacion para H0: features independientes
de parálogo (ESM-2 zero-shot, BLOSUM62, delta de propiedades fisicoquimicas) mas la
etiqueta real (score DMS de MaveDB), para MLH1 (entrenamiento) y MSH2 (retencion,
oculto durante el entrenamiento).

IMPORTANTE (verificado en los metadatos oficiales de MaveDB, methodText de cada
score set, 5-ago-2026): las dos escalas tienen CONVENCION DE SIGNO OPUESTA.
- MSH2 (urn:mavedb:00000050-a-1): "Positive scores correspond to loss-of-function
  and negative scores correspond to functionally neutral variants." -> alto = danino.
- MLH1 (urn:mavedb:00001218-a-1): normalizado a variantes sinonimas silvestres
  (~0 = neutro); las variantes sin sentido (Ter) tienen mediana -1.00 frente a la
  mediana global -0.16 -> bajo = danino, alto = neutro.
Se genera aqui 'score_danino' con una unica convencion (alto = mas danino) para
las dos, invirtiendo el signo de MLH1. Sin esta correccion, cualquier modelo
entrenado en una escala y evaluado en la otra sale con el signo invertido -- el
mismo tipo de error que invirtio un modelo entero por un mapeo de APOE mal hecho.
"""
import json
import re
import csv

from Bio.Align import substitution_matrices

BLOSUM62 = substitution_matrices.load("BLOSUM62")

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V", "Ter": "*",
}

# Kyte-Doolittle hydrophobicity y volumen (A^3, Zamyatnin)
HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8,
    "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6,
    "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8,
    "Y": 193.6, "V": 140.0,
}


def parse_hgvs_pro(hgvs):
    """p.Met1Ala -> (1, 'M', 'A'); ignora sinonimos/stop/frameshift."""
    m = re.match(r"^p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})$", hgvs.strip())
    if not m:
        return None
    aa1_3, pos, aa2_3 = m.groups()
    aa1 = AA3_TO_1.get(aa1_3.capitalize())
    aa2 = AA3_TO_1.get(aa2_3.capitalize())
    if aa1 is None or aa2 is None or aa1 == "*" or aa2 == "*":
        return None
    return int(pos), aa1, aa2


def cargar_esm_zeroshot(gene):
    path = f"/home/jesus/paper_msh6/datos/esm2_3B_zeroshot/{gene}_esm2_3B_zeroshot.json"
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        pos_str, aa = k.rsplit("_", 1)
        out[(int(pos_str), aa)] = v
    return out


_ESTRUCTURA_CACHE = None


def cargar_estructura():
    """Carga datos/features_estructurales.json (pLDDT de los 4 genes, distancias
    ADN/pareja solo para MSH2 y MSH6, ver 11_features_estructurales.py)."""
    global _ESTRUCTURA_CACHE
    if _ESTRUCTURA_CACHE is None:
        with open("/home/jesus/paper_msh6/datos/features_estructurales.json") as f:
            _ESTRUCTURA_CACHE = json.load(f)
    return _ESTRUCTURA_CACHE


def features_variante(pos, wt_aa, mut_aa, esm_dict, gene=None, usar_estructura=False):
    esm_score = esm_dict.get((pos, mut_aa))
    if esm_score is None:
        return None
    blosum = BLOSUM62[wt_aa][mut_aa]
    dh = HYDROPHOBICITY[mut_aa] - HYDROPHOBICITY[wt_aa]
    dv = VOLUME[mut_aa] - VOLUME[wt_aa]
    feats = {
        "posicion": pos,
        "wt_aa": wt_aa,
        "mut_aa": mut_aa,
        "esm2_3B_zeroshot": esm_score,
        "blosum62": blosum,
        "delta_hidrofobicidad": dh,
        "delta_volumen": dv,
    }
    if usar_estructura:
        estr = cargar_estructura()
        plddt = estr["plddt"].get(gene, {}).get(str(pos))
        if plddt is None:
            return None
        feats["plddt"] = plddt
        dist_gen = estr["distancias_mutsalpha"].get(gene, {})
        d = dist_gen.get(str(pos))
        if d is not None:
            feats["dist_adn"] = d["dist_adn"]
            feats["dist_pareja"] = d["dist_pareja"]
    return feats


def construir(gene, mavedb_csv_path, score_col="scores.score", usar_estructura=False):
    esm_dict = cargar_esm_zeroshot(gene)
    filas = []
    with open(mavedb_csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hgvs = row.get("hgvs_pro", "NA")
            score = row.get(score_col, "NA")
            if hgvs in ("NA", "") or score in ("NA", ""):
                continue
            parsed = parse_hgvs_pro(hgvs)
            if parsed is None:
                continue
            pos, wt_aa, mut_aa = parsed
            feats = features_variante(pos, wt_aa, mut_aa, esm_dict, gene=gene, usar_estructura=usar_estructura)
            if feats is None:
                continue
            feats["score_dms"] = float(score)
            feats["gen"] = gene
            filas.append(feats)
    return filas


# Orientacion verificada contra el methodText de MaveDB (ver docstring del modulo):
# alto score_danino = mas danino/LOF, en las dos escalas.
SIGNO_DANINO = {"MSH2": 1.0, "MLH1": -1.0}


def main():
    mlh1 = construir("MLH1", "/home/jesus/paper_msh6/datos/mavedb/urn_mavedb_00001218-a-1_variants.csv",
                      usar_estructura=True)
    msh2 = construir("MSH2", "/home/jesus/paper_msh6/datos/mavedb/urn_mavedb_00000050-a-1_variants.csv",
                      usar_estructura=True)

    for gene, data in [("MLH1", mlh1), ("MSH2", msh2)]:
        for d in data:
            d["score_danino"] = SIGNO_DANINO[gene] * d["score_dms"]

    print(f"MLH1: {len(mlh1)} variantes con features completas (de 5.056 en MaveDB)")
    print(f"MSH2: {len(msh2)} variantes con features completas (de 17.746 en MaveDB)")

    out_dir = "/home/jesus/paper_msh6/datos"
    for gene, data in [("MLH1", mlh1), ("MSH2", msh2)]:
        path = f"{out_dir}/dataset_H0_{gene}.json"
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Guardado: {path}")


if __name__ == "__main__":
    main()
