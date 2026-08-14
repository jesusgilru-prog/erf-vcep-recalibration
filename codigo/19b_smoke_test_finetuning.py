"""
Prueba rapida (subconjunto pequeno, 1 epoca, batch chico) antes de comprometer
otra hora de computo al fine-tuning completo. Objetivo: confirmar que la memoria
de GPU se mantiene estable paso a paso (no crece sin control como en los dos
intentos anteriores, que fallaron con CUDA error tras 35-60 min).
"""
import json
import os

os.environ["HF_HOME"] = "/home/jesus/paper_msh6/modelos/hf_cache"

import sys
sys.path.insert(0, "/home/jesus/paper_msh6/codigo")
from importlib import import_module
mod = import_module("19_finetuning_esm2_lora")

import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(mod.MODEL_NAME)
seq = mod.leer_fasta("/home/jesus/paper_msh6/datos/secuencias/MSH2_P43246.fasta")
variantes = mod.cargar_dataset_variantes("/home/jesus/paper_msh6/datos/dataset_H0_MSH2.json")[:300]

print(f"Subconjunto de prueba: {len(variantes)} variantes, secuencia {len(seq)} aa")
modelo = mod.entrenar_finetuning("MSH2-smoke", seq, variantes, tokenizer, epochs=1, batch_size=4)
print("Smoke test OK, sin caida.")
