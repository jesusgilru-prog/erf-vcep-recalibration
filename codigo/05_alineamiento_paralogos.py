"""
Alineamiento global (Needleman-Wunsch, BLOSUM62) entre los pares de paralogos del
mismo complejo: MSH2<->MSH6 (MutSalpha) y MLH1<->PMS2 (MutLalpha). Produce el mapa
de posiciones equivalentes que usa el modelo de transferencia (item 14 del grupo B
de la Propuesta 7).
"""
import json
from Bio import Align
from Bio.Align import substitution_matrices


def leer_fasta(path):
    with open(path) as f:
        lines = f.readlines()
    seq = "".join(l.strip() for l in lines if not l.startswith(">"))
    return seq


SEQ_DIR = "/home/jesus/paper_msh6/datos/secuencias"
seqs = {
    "MSH2": leer_fasta(f"{SEQ_DIR}/MSH2_P43246.fasta"),
    "MSH6": leer_fasta(f"{SEQ_DIR}/MSH6_P52701.fasta"),
    "MLH1": leer_fasta(f"{SEQ_DIR}/MLH1_P40692.fasta"),
    "PMS2": leer_fasta(f"{SEQ_DIR}/PMS2_P54278.fasta"),
}
for g, s in seqs.items():
    print(f"{g}: {len(s)} aa")

aligner = Align.PairwiseAligner()
aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
aligner.open_gap_score = -11
aligner.extend_gap_score = -1
aligner.mode = "global"


def alinear_y_mapear(gen_a, gen_b):
    seq_a, seq_b = seqs[gen_a], seqs[gen_b]
    alignments = aligner.align(seq_a, seq_b)
    best = alignments[0]
    aligned_a, aligned_b = str(best[0]), str(best[1])

    pos_a = 0  # posicion 1-based en gen_a
    pos_b = 0
    identicos = 0
    alineados = 0
    mapa_a_a_b = {}
    for ca, cb in zip(aligned_a, aligned_b):
        if ca != "-":
            pos_a += 1
        if cb != "-":
            pos_b += 1
        if ca != "-" and cb != "-":
            alineados += 1
            mapa_a_a_b[pos_a] = {"pos_b": pos_b, "aa_a": ca, "aa_b": cb, "identico": ca == cb}
            if ca == cb:
                identicos += 1

    identidad = 100 * identicos / alineados if alineados else 0
    print(f"\n{gen_a} <-> {gen_b}: score={best.score:.1f}, "
          f"posiciones alineadas (sin huecos)={alineados}, identidad={identidad:.1f}%")
    return mapa_a_a_b, {"score": best.score, "alineadas": alineados, "identidad_pct": identidad,
                          "aligned_a": aligned_a, "aligned_b": aligned_b}


mapa_msh2_msh6, stats_1 = alinear_y_mapear("MSH2", "MSH6")
mapa_mlh1_pms2, stats_2 = alinear_y_mapear("MLH1", "PMS2")

out = {
    "MSH2_a_MSH6": {"mapa": mapa_msh2_msh6, "stats": {k: v for k, v in stats_1.items() if k != "aligned_a" and k != "aligned_b"}},
    "MLH1_a_PMS2": {"mapa": mapa_mlh1_pms2, "stats": {k: v for k, v in stats_2.items() if k != "aligned_a" and k != "aligned_b"}},
}

with open("/home/jesus/paper_msh6/datos/alineamiento_paralogos.json", "w") as f:
    json.dump(out, f, indent=2)

with open("/home/jesus/paper_msh6/datos/alineamiento_paralogos_secuencias.txt", "w") as f:
    f.write("=== MSH2 vs MSH6 ===\n")
    f.write(stats_1["aligned_a"] + "\n")
    f.write(stats_2 if False else "")
    f.write(stats_1["aligned_b"] + "\n\n")
    f.write("=== MLH1 vs PMS2 ===\n")
    f.write(stats_2["aligned_a"] + "\n")
    f.write(stats_2["aligned_b"] + "\n")

print("\nGuardado: datos/alineamiento_paralogos.json, datos/alineamiento_paralogos_secuencias.txt")
