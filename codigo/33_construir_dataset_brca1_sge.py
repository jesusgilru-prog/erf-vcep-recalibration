"""
Construye el dataset BRCA1 SGE (Findlay et al. 2018, Nature, "Saturation genome
editing of BRCA1 RING and BRCT domains" -- MaveDB urn:mavedb:00000097-0-2,
"BRCA1 SGE Normalized Scores", 3.893 variantes) a nivel de proteina, para usarlo
como arbitro FUNCIONAL independiente (celda de validacion propuesta por Claude,
debate de redireccion 11-ago-2026): a diferencia de MAPP/PP2 (alineamiento) y
ESM-2 (evolutivo), el SGE mide directamente el efecto funcional real de cada
variante en un ensayo de edicion genomica -- no es concordancia entre dos
metodos afines.

El dataset de MaveDB solo trae hgvs_nt (NM_007294.3:c.NNNX>Y), sin hgvs_pro
-- se traduce aqui usando la CDS real de NM_007294.3 (descargada de NCBI,
verificada 1864 codones incl. stop = 1863 aa de proteina, coincide con
UniProt P38398).
"""
import csv
import json
import re

CODIGO_GENETICO = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}


def leer_cds(path):
    with open(path) as f:
        lines = f.readlines()
    return "".join(l.strip() for l in lines if not l.startswith(">"))


def parse_hgvs_c(hgvs):
    """'NM_007294.3:c.5565A>T' -> (pos_1based_cds, ref, alt). Solo SNV codificantes
    simples (descarta splice/intron/indel: c.123+45..., c.123-45..., c.123_124..., etc.)."""
    m = re.match(r"^NM_\d+\.\d+:c\.(\d+)([ACGT])>([ACGT])$", hgvs.strip())
    if not m:
        return None
    pos, ref, alt = m.groups()
    return int(pos), ref, alt


def traducir_variante(cds, pos_nt, ref, alt):
    if cds[pos_nt - 1] != ref:
        return None  # discrepancia con la CDS de referencia, descartar por seguridad
    codon_idx = (pos_nt - 1) // 3
    pos_en_codon = (pos_nt - 1) % 3
    codon_wt = cds[codon_idx * 3: codon_idx * 3 + 3]
    codon_mut = list(codon_wt)
    codon_mut[pos_en_codon] = alt
    codon_mut = "".join(codon_mut)
    wt_aa = CODIGO_GENETICO.get(codon_wt)
    mut_aa = CODIGO_GENETICO.get(codon_mut)
    if wt_aa is None or mut_aa is None:
        return None
    pos_aa = codon_idx + 1  # 1-based, codon 1 = Met inicial
    return pos_aa, wt_aa, mut_aa


def main():
    cds = leer_cds("/tmp/brca1_cds.fasta")
    print(f"CDS: {len(cds)} nt, {len(cds)//3} codones")

    filas = []
    omitidas_no_snv, omitidas_discrepancia, sinonimas, sin_sentido = 0, 0, 0, 0
    with open("/home/jesus/paper_msh6/datos/mavedb/urn_mavedb_00000097-0-2_variants.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = parse_hgvs_c(row["hgvs_nt"])
            if parsed is None:
                omitidas_no_snv += 1
                continue
            pos_nt, ref, alt = parsed
            trad = traducir_variante(cds, pos_nt, ref, alt)
            if trad is None:
                omitidas_discrepancia += 1
                continue
            pos_aa, wt_aa, mut_aa = trad
            if row["score"] in ("NA", ""):
                continue
            if wt_aa == mut_aa:
                sinonimas += 1
                continue
            if mut_aa == "*":
                sin_sentido += 1
            filas.append({"posicion": pos_aa, "wt_aa": wt_aa, "mut_aa": mut_aa,
                           "score_sge": float(row["score"]), "es_sin_sentido": mut_aa == "*"})

    print(f"Filas totales CSV: omitidas no-SNV/splice/indel={omitidas_no_snv}, "
          f"discrepancia con CDS={omitidas_discrepancia}, sinonimas={sinonimas}, "
          f"sin sentido incluidas={sin_sentido}")
    print(f"Variantes missense+sinsentido traducidas: {len(filas)}")

    # verificar convencion de signo: mediana sin_sentido (deberia ser clara y
    # dañina) vs mediana missense global, siguiendo el mismo protocolo que 07_
    import numpy as np
    scores_sinsentido = [f["score_sge"] for f in filas if f["es_sin_sentido"]]
    scores_missense = [f["score_sge"] for f in filas if not f["es_sin_sentido"]]
    print(f"Mediana score sin_sentido (n={len(scores_sinsentido)}): {np.median(scores_sinsentido):.4f}")
    print(f"Mediana score missense (n={len(scores_missense)}): {np.median(scores_missense):.4f}")
    if np.median(scores_sinsentido) < np.median(scores_missense):
        print("-> Confirmado: score BAJO = mas danino (sin sentido mas bajo que missense en general),"
              " igual convencion que MLH1 en este proyecto.")
        signo_danino = -1.0
    else:
        print("-> Confirmado: score ALTO = mas danino (sin sentido mas alto que missense en general),"
              " igual convencion que MSH2 en este proyecto.")
        signo_danino = 1.0

    missense_final = [f for f in filas if not f["es_sin_sentido"]]
    for f in missense_final:
        f["score_danino"] = signo_danino * f["score_sge"]

    with open("/home/jesus/paper_msh6/datos/dataset_BRCA1_SGE.json", "w") as f_out:
        json.dump(missense_final, f_out)
    print(f"\nGuardado: datos/dataset_BRCA1_SGE.json ({len(missense_final)} variantes missense, "
          f"signo_danino={signo_danino:+.0f})")


if __name__ == "__main__":
    main()
