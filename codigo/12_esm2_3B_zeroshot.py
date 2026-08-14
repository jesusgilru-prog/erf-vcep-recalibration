"""
Igual que 06_esm2_zeroshot.py pero con ESM-2 3B (facebook/esm2_t36_3B_UR50D), el
modelo que pide la Propuesta 7 (item 5 del grupo A). El de 650M sirvio para prototipar
barato; con la senal ya confirmada (H0 superada, H1/H2 con correlacion real), toca
la version completa.
"""
import json
import os
import time

os.environ["HF_HOME"] = "/home/jesus/paper_msh6/modelos/hf_cache"

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "facebook/esm2_t36_3B_UR50D"

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")


def leer_fasta(path):
    with open(path) as f:
        lines = f.readlines()
    return "".join(l.strip() for l in lines if not l.startswith(">"))


def calcular_zeroshot(gene, seq, tokenizer, model, batch_size=8):
    L = len(seq)
    inputs = tokenizer(seq, return_tensors="pt")
    input_ids = inputs["input_ids"].to(DEVICE)
    mask_token_id = tokenizer.mask_token_id
    scores = {}
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, L, batch_size):
            end = min(start + batch_size, L)
            batch_ids = input_ids.repeat(end - start, 1).clone()
            for i, pos in enumerate(range(start, end)):
                batch_ids[i, pos + 1] = mask_token_id
            logits = model(batch_ids).logits
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
            if start % (batch_size * 20) == 0:
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

    print("Cargando ESM-2 3B...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE).eval()
    print(f"Dispositivo: {DEVICE}")

    out_dir = "/home/jesus/paper_msh6/datos/esm2_3B_zeroshot"
    os.makedirs(out_dir, exist_ok=True)

    for gene, path in genes.items():
        seq = leer_fasta(path)
        print(f"\n=== {gene}: {len(seq)} aa ===")
        t0 = time.time()
        scores = calcular_zeroshot(gene, seq, tokenizer, model)
        elapsed = time.time() - t0
        print(f"{gene}: {len(scores)} variantes puntuadas en {elapsed:.1f}s")

        out_path = f"{out_dir}/{gene}_esm2_3B_zeroshot.json"
        serializable = {f"{pos}_{aa}": v for (pos, aa), v in scores.items()}
        with open(out_path, "w") as f:
            json.dump(serializable, f)
        print(f"Guardado: {out_path}")


if __name__ == "__main__":
    main()
