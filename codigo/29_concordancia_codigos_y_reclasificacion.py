"""
Tres analisis pedidos por Claude como los que de verdad sostienen (o no) la
afirmacion central del paper, ninguno hecho hasta ahora:

(B2) AUC pareado ESM-2 vs prior oficial restringido a variantes RARAS (gnomAD
     AF<1e-4 o ausentes de gnomAD) -- la preocupacion es que el AUC~0.97-0.99
     original este inflado por variantes benignas comunes, faciles de separar
     para cualquier metodo.
(B3) Concordancia a nivel de CODIGO ACMG (no AUC) entre ESM-2 calibrado
     (resultado_calibracion_ESM_solo via reclasificacion_VUS_MSH6.json) y el
     prior oficial (resultado_oficial_vs_esm2.json), sobre las VUS reales
     donde ambos estan disponibles. Es el objeto real del paper.
(B4) Cota inferior de reclasificacion real: VUS con BP4_Supporting oficial Y
     BS1 (frecuencia poblacional gnomAD, derivable independientemente) ->
     Likely Benign bajo Richards 2015 (1 Strong + 1 Supporting benigno).
"""
import json
import re

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}
UMBRAL_BS1_AF = 0.001  # 0.1%, mismo umbral ya usado en el proyecto (refuerzo BS1 6-ago-2026)


def parse_hgvsp_3letra(hgvsp):
    m = re.match(r"^p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})$", hgvsp or "")
    if not m:
        return None
    aa1_3, pos, aa2_3 = m.groups()
    aa1, aa2 = AA3_TO_1.get(aa1_3.capitalize()), AA3_TO_1.get(aa2_3.capitalize())
    if aa1 is None or aa2 is None:
        return None
    return int(pos), aa1, aa2


def cargar_gnomad_af(gene):
    """(pos, mut_aa) -> AF maxima entre exome/genome (0.0 si ausente de gnomAD)."""
    with open(f"/home/jesus/paper_msh6/datos/gnomad_{gene}.json") as f:
        raw = json.load(f)
    out = {}
    for r in raw:
        if r["consequence"] != "missense_variant":
            continue
        parsed = parse_hgvsp_3letra(r.get("hgvsp"))
        if parsed is None:
            continue
        pos, wt_aa, mut_aa = parsed
        af_exome = (r.get("exome") or {}).get("af") or 0.0
        af_genome = (r.get("genome") or {}).get("af") or 0.0
        af = max(af_exome, af_genome)
        key = (pos, mut_aa)
        out[key] = max(af, out.get(key, 0.0))
    return out


def parse_protein_change(pc):
    m = re.match(r"^p\.([A-Za-z]{1,3})(\d+)([A-Za-z]{1,3})$", pc)
    if not m:
        return None
    aa1, pos, aa2 = m.groups()
    if len(aa1) == 3:
        aa1 = AA3_TO_1.get(aa1.capitalize())
    if len(aa2) == 3:
        aa2 = AA3_TO_1.get(aa2.capitalize())
    if aa1 is None or aa2 is None:
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
        out[(pos, mut_aa)] = r["prior_p"]
    return out


def cargar_esm(gene):
    path = f"/home/jesus/paper_msh6/datos/esm2_zeroshot/{gene}_esm2_650M_zeroshot.json"
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        pos_str, aa = k.rsplit("_", 1)
        out[(int(pos_str), aa)] = -v
    return out


AA1_TO_3 = {v: k for k, v in AA3_TO_1.items()}


def parse_variante_1letra(s):
    """'p.Lys185Glu' o 'K185E' -> (pos, wt, mut)"""
    m = re.match(r"^p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})$", s)
    if m:
        aa1_3, pos, aa2_3 = m.groups()
        return int(pos), AA3_TO_1.get(aa1_3.capitalize()), AA3_TO_1.get(aa2_3.capitalize())
    m = re.match(r"^([A-Z])(\d+)([A-Z])$", s)
    if m:
        return int(m.group(2)), m.group(1), m.group(3)
    return None


def b2_auc_variantes_raras(gene, select_db):
    print(f"\n{'='*70}\n{gene} -- B2: AUC pareado restringido a variantes RARAS\n{'='*70}")
    import csv, sys
    csv.field_size_limit(sys.maxsize)
    esm = cargar_esm(gene)
    oficial = cargar_oficial(select_db)
    gnomad_af = cargar_gnomad_af(gene)

    PATOGENICAS = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
    BENIGNAS = {"Benign", "Likely benign", "Benign/Likely benign"}
    etiquetados = {}
    with open("/home/jesus/paper_msh6/datos/variant_summary.txt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["Assembly"] != "GRCh38" or gene not in set(row["GeneSymbol"].split("|")):
                continue
            sig = row["ClinicalSignificance"]
            if sig in PATOGENICAS:
                label = 1
            elif sig in BENIGNAS:
                label = 0
            else:
                continue
            m = re.search(r"\(p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})\)", row["Name"])
            if not m:
                continue
            aa1_3, pos, aa2_3 = m.groups()
            mut_aa = AA3_TO_1.get(aa2_3.capitalize())
            if mut_aa is None:
                continue
            etiquetados[(int(pos), mut_aa)] = label

    comunes = [k for k in etiquetados if k in esm and k in oficial]
    y_all = np.array([etiquetados[k] for k in comunes])
    print(f"  Conjunto pareado completo (como en 23_): n={len(comunes)}, "
          f"pat={int(y_all.sum())}, ben={int((1-y_all).sum())}")

    raras = [k for k in comunes if gnomad_af.get(k, 0.0) < 1e-4]
    y_r = np.array([etiquetados[k] for k in raras])
    n_pos_r, n_neg_r = int(y_r.sum()), int((1 - y_r).sum())
    print(f"  Restringido a AF<1e-4 (o ausentes de gnomAD): n={len(raras)}, pat={n_pos_r}, ben={n_neg_r}")
    if n_pos_r < 5 or n_neg_r < 5:
        print("  Insuficiente para AUC fiable tras restringir a raras.")
        return {"gene": gene, "n_raras": len(raras), "n_pos": n_pos_r, "n_neg": n_neg_r, "insuficiente": True}

    s_esm_r = np.array([esm[k] for k in raras])
    s_of_r = np.array([oficial[k] for k in raras])
    auc_esm_r = roc_auc_score(y_r, s_esm_r)
    auc_of_r = roc_auc_score(y_r, s_of_r)
    print(f"  AUC ESM-2 (solo raras):     {auc_esm_r:.4f}")
    print(f"  AUC oficial (solo raras):   {auc_of_r:.4f}")
    print(f"  Diferencia (oficial-ESM2):  {auc_of_r - auc_esm_r:+.4f}")

    return {"gene": gene, "n_raras": len(raras), "n_pos": n_pos_r, "n_neg": n_neg_r,
            "auc_esm_raras": float(auc_esm_r), "auc_oficial_raras": float(auc_of_r)}


def b3_concordancia_codigos(gene, clave_reclasificacion):
    print(f"\n{'='*70}\n{gene} -- B3: concordancia a nivel de codigo ACMG (ESM-2 calibrado vs oficial)\n{'='*70}")
    with open("/home/jesus/paper_msh6/datos/reclasificacion_VUS_MSH6.json") as f:
        recl = json.load(f)[gene]
    with open("/home/jesus/paper_msh6/datos/resultado_oficial_vs_esm2.json") as f:
        ofic = json.load(f)[gene]

    esm_por_vid = {r["variation_id"]: r["evidencia_conservadora"] for r in recl["resultados"]}
    ofic_por_vid = {r["variation_id"]: r["evidencia_oficial"] for r in ofic["detalle_vus"]}

    comunes = set(esm_por_vid) & set(ofic_por_vid)
    print(f"  VUS con codigo ESM-2 calibrado Y codigo oficial simultaneos: {len(comunes)}")

    tabla = {}
    for vid in comunes:
        a, b = esm_por_vid[vid], ofic_por_vid[vid]
        tabla.setdefault(a, {}).setdefault(b, 0)
        tabla[a][b] += 1

    print("  Tabla de contingencia (filas=ESM-2 calibrado, columnas=oficial):")
    codigos = ["PP3_Moderate", "PP3_Supporting", "sin_evidencia", "BP4_Supporting"]
    header = "                    " + "".join(f"{c:>16}" for c in codigos)
    print(header)
    for a in codigos:
        fila = tabla.get(a, {})
        print(f"  {a:<18}" + "".join(f"{fila.get(b, 0):>16}" for b in codigos))

    n_exacto = sum(tabla.get(c, {}).get(c, 0) for c in codigos)
    n_discordancia_direccion_opuesta = (
        tabla.get("PP3_Moderate", {}).get("BP4_Supporting", 0)
        + tabla.get("PP3_Supporting", {}).get("BP4_Supporting", 0)
        + tabla.get("BP4_Supporting", {}).get("PP3_Moderate", 0)
        + tabla.get("BP4_Supporting", {}).get("PP3_Supporting", 0)
    )
    print(f"\n  Coincidencia exacta de codigo: {n_exacto}/{len(comunes)} ({100*n_exacto/len(comunes):.1f}%)")
    print(f"  Discordancia de DIRECCION OPUESTA (uno dice PP3, el otro BP4): "
          f"{n_discordancia_direccion_opuesta}/{len(comunes)} "
          f"({100*n_discordancia_direccion_opuesta/len(comunes):.1f}%) -- el caso grave")

    return {"gene": gene, "n": len(comunes), "tabla_contingencia": tabla,
            "n_coincidencia_exacta": n_exacto, "pct_coincidencia_exacta": 100 * n_exacto / len(comunes),
            "n_discordancia_direccion_opuesta": n_discordancia_direccion_opuesta,
            "pct_discordancia_direccion_opuesta": 100 * n_discordancia_direccion_opuesta / len(comunes)}


def b4_reclasificacion_bp4_mas_bs1(gene, select_db):
    print(f"\n{'='*70}\n{gene} -- B4: cota inferior de reclasificacion real (BP4_Supporting oficial + BS1 gnomAD)\n{'='*70}")
    with open("/home/jesus/paper_msh6/datos/resultado_oficial_vs_esm2.json") as f:
        ofic = json.load(f)[gene]
    gnomad_af = cargar_gnomad_af(gene)

    n_bp4 = 0
    n_bp4_mas_bs1 = 0
    for r in ofic["detalle_vus"]:
        if r["evidencia_oficial"] != "BP4_Supporting":
            continue
        n_bp4 += 1
        parsed = parse_variante_1letra(r["variante"])
        if parsed is None:
            continue
        pos, wt_aa, mut_aa = parsed
        af = gnomad_af.get((pos, mut_aa), 0.0)
        if af >= UMBRAL_BS1_AF:
            n_bp4_mas_bs1 += 1

    print(f"  VUS con BP4_Supporting oficial: {n_bp4}")
    print(f"  De esas, tambien con BS1 (AF gnomAD >= {UMBRAL_BS1_AF:.1%}, Strong benigno bajo Richards 2015): "
          f"{n_bp4_mas_bs1}")
    print(f"  1 Strong + 1 Supporting benigno = Likely Benign bajo Richards 2015.")
    print(f"  Cota inferior de VUS reclasificables HOY a Likely Benign, con evidencia ya existente y "
          f"verificable de forma independiente: {n_bp4_mas_bs1}/{ofic['n_vus_total']} "
          f"({100*n_bp4_mas_bs1/ofic['n_vus_total']:.1f}% de todas las VUS del gen)")

    return {"gene": gene, "n_vus_total": ofic["n_vus_total"], "n_bp4_supporting": n_bp4,
            "n_bp4_mas_bs1_likely_benign": n_bp4_mas_bs1,
            "pct_reclasificable_ahora": 100 * n_bp4_mas_bs1 / ofic["n_vus_total"]}


def main():
    resultado = {"B2_auc_raras": {}, "B3_concordancia_codigos": {}, "B4_reclasificacion": {}}
    for gene, select_db in [("MSH6", "MSH6_priors"), ("PMS2", "PMS2_priors")]:
        resultado["B2_auc_raras"][gene] = b2_auc_variantes_raras(gene, select_db)
        resultado["B3_concordancia_codigos"][gene] = b3_concordancia_codigos(gene, select_db)
        resultado["B4_reclasificacion"][gene] = b4_reclasificacion_bp4_mas_bs1(gene, select_db)

    with open("/home/jesus/paper_msh6/datos/resultado_concordancia_y_reclasificacion.json", "w") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    print("\n\nGuardado: datos/resultado_concordancia_y_reclasificacion.json")


if __name__ == "__main__":
    main()
