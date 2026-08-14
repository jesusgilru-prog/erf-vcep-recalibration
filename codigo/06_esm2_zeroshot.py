"""
Calcula el score zero-shot de ESM-2 (650M) por variante missense, metodo de las
marginales enmascaradas (masked-marginal, Meier et al. 2021): para cada posicion,
se enmascara el residuo salvaje y se leen los logits del modelo; el score de la
variante es log P(mut) - log P(wt) en esa posicion, con el resto de la secuencia
visible sin enmascarar.

Es el mismo procedimiento que usa el campo para "ESM-2 3B (zero-shot)" (item 5 del
grupo A de la Propuesta 7); aqui se usa ESM-2 650M como primera pasada mas barata en
disco/computo (2.5 GB en vez de ~11 GB), documentado como decision de implementacion,
no como desviacion del preregistro (que no fija el tamano del modelo).
"""
import json
import os
import sys
import time

os.environ["HF_HOME"] = "/home/jesus/paper_msh6/modelos/hf_cache"

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")


def leer_fasta(path):
    with open(path) as f:
        lines = f.readlines()
    return "".join(l.strip() for l in lines if not l.startswith(">"))


def calcular_zeroshot(gene, seq, tokenizer, model, batch_size=16):
    """Devuelve dict {(pos_1based, aa_mut): score} para las 19 sustituciones posibles
    en cada posicion, usando el metodo de marginales enmascaradas."""
    L = len(seq)
    inputs = tokenizer(seq, return_tensors="pt")
    input_ids = inputs["input_ids"].to(DEVICE)
    mask_token_id = tokenizer.mask_token_id
    # offset: token 0 es <cls>, luego posiciones 1..L son los residuos, luego <eos>
    scores = {}
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, L, batch_size):
            end = min(start + batch_size, L)
            batch_ids = input_ids.repeat(end - start, 1).clone()
            for i, pos in enumerate(range(start, end)):
                batch_ids[i, pos + 1] = mask_token_id
            logits = model(batch_ids).logits  # (batch, seq_len, vocab)
            log_probs = torch.log_softmax(logits, dim=-1)
            for i, pos in enumerate(range(start, end)):
                wt_aa = seq[pos]
                wt_id = tokenizer.convert_tokens_to_ids(wt_aa)
                lp_wt = log_probs[i, pos + 1, wt_id].item()
                for aa in AA_ORDER:
                    if aa == wt_aa:
                        continue
                    aa_id = tokenizer.convert_tokens_to_ids(aa)
                    lp_mut = log_probs[i, pos + 1, aa_id].item()
                    scores[(pos + 1, aa)] = lp_mut - lp_wt
            if start % (batch_size * 10) == 0:
                elapsed = time.time() - t0
                print(f"  {gene}: posicion {end}/{L} ({elapsed:.1f}s)", flush=True)
    return scores


def main():
    seq_dir = "/home/jesus/paper_msh6/datos/secuencias"
    genes = {
        "MSH2": f"{seq_dir}/MSH2_P43246.fasta",
        "MSH6": f"{seq_dir}/MSH6_P52701.fasta",
        "MLH1": f"{seq_dir}/MLH1_P40692.fasta",
        "PMS2": f"{seq_dir}/PMS2_P54278.fasta",
    }

    print("Cargando ESM-2 650M...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    print(f"Dispositivo: {DEVICE}")

    out_dir = "/home/jesus/paper_msh6/datos/esm2_zeroshot"
    os.makedirs(out_dir, exist_ok=True)

    for gene, path in genes.items():
        seq = leer_fasta(path)
        print(f"\n=== {gene}: {len(seq)} aa ===")
        t0 = time.time()
        scores = calcular_zeroshot(gene, seq, tokenizer, model)
        elapsed = time.time() - t0
        print(f"{gene}: {len(scores)} variantes puntuadas en {elapsed:.1f}s")

        out_path = f"{out_dir}/{gene}_esm2_650M_zeroshot.json"
        serializable = {f"{pos}_{aa}": v for (pos, aa), v in scores.items()}
        with open(out_path, "w") as f:
            json.dump(serializable, f)
        print(f"Guardado: {out_path}")


if __name__ == "__main__":
    main()
