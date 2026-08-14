"""
ESM-2 650M zero-shot para BRCA1 (mismo metodo que 06_, marginales enmascaradas),
para validar contra la verdad funcional real SGE (Findlay 2018, `33_`) a escala
-- comprobacion externa e independiente de la fiabilidad de ESM-2 fuera de la
familia MMR, en un gen con ensayo funcional real disponible.
"""
import json
import os
import time

os.environ["HF_HOME"] = "/home/jesus/paper_msh6/modelos/hf_cache"

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

import importlib
base = importlib.import_module("06_esm2_zeroshot")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"


def main():
    seq = base.leer_fasta("/home/jesus/paper_msh6/datos/secuencias/BRCA1_P38398.fasta")
    print(f"BRCA1: {len(seq)} aa. Dispositivo: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    t0 = time.time()
    scores = base.calcular_zeroshot("BRCA1", seq, tokenizer, model)
    print(f"BRCA1: {len(scores)} variantes puntuadas en {time.time()-t0:.1f}s")

    out_dir = "/home/jesus/paper_msh6/datos/esm2_zeroshot"
    serializable = {f"{pos}_{aa}": v for (pos, aa), v in scores.items()}
    with open(f"{out_dir}/BRCA1_esm2_650M_zeroshot.json", "w") as f:
        json.dump(serializable, f)
    print(f"Guardado: {out_dir}/BRCA1_esm2_650M_zeroshot.json")


if __name__ == "__main__":
    main()
