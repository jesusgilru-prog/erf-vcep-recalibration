"""
Extiende el fine-tuning LoRA (19_, ya completado contra el set congelado n=18/51)
al prior oficial a escala, cerrando el ultimo hueco que senalo Claude en la
re-evaluacion de la transferencia (item 9, `26_`): de los 5 mecanismos de
transferencia, LoRA es el unico que no se habia podido re-evaluar a escala
porque no se guardaron los pesos entrenados. Aqui se entrena igual que en 19_
(mismo LoRA, mismos hiperparametros) y, en la misma ejecucion (sin recargar
nada), se generan predicciones para TODAS las variantes con prior oficial
disponible -- igual que se hizo con los otros 3 mecanismos CPU en 26_.
"""
import json
import os

os.environ["HF_HOME"] = "/home/jesus/paper_msh6/modelos/hf_cache"

import re
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from scipy.stats import spearmanr

import importlib
ft = importlib.import_module("19_finetuning_esm2_lora")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AA3_TO_1 = ft.AA3_TO_1


def parse_protein_change(pc):
    m = re.match(r"^p\.([A-Za-z]{1,3})(\d+)([A-Za-z]{1,3})$", pc)
    if not m:
        return None
    aa1, pos, aa2 = m.groups()
    if len(aa1) == 3:
        aa1 = AA3_TO_1.get(aa1.capitalize())
    if len(aa2) == 3:
        aa2 = AA3_TO_1.get(aa2.capitalize())
    if aa1 is None or aa2 is None or len(aa1) != 1 or len(aa2) != 1:
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


def predecir_a_escala(modelo, wt_seq, variantes_pos_aa, tokenizer, batch_size=16):
    """variantes_pos_aa: lista de (pos_1based, mut_aa). Devuelve array de predicciones,
    mismo orden. Reutiliza el mismo forward que evaluar_en_frozen pero por lotes."""
    modelo.eval()
    preds = []
    t0 = time.time()
    for i in range(0, len(variantes_pos_aa), batch_size):
        lote = variantes_pos_aa[i:i + batch_size]
        seqs = []
        for pos, mut_aa in lote:
            seq = list(wt_seq)
            seq[pos - 1] = mut_aa
            seqs.append("".join(seq))
        enc = tokenizer(seqs, return_tensors="pt", padding=True)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        posT = torch.tensor([p for p, _ in lote], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            pred = modelo(enc["input_ids"], enc["attention_mask"], posT)
        preds.extend(pred.cpu().numpy().tolist())
        del enc, posT, pred
        if i % (batch_size * 50) == 0:
            torch.cuda.empty_cache()
            transcurrido = time.time() - t0
            print(f"    {i}/{len(variantes_pos_aa)} ({transcurrido:.0f}s, "
                  f"mem_reserved={torch.cuda.memory_reserved()/1e9:.2f}GB)", flush=True)
    return np.array(preds)


def procesar(nombre_h, gene_rico, dataset_rico_json, seq_rico_path, gene_huerfano,
             seq_huerfano_path, select_db_oficial, frozen_key, es_msh6, tokenizer, frozen):
    print(f"\n{'='*70}\n{nombre_h}: fine-tuning LoRA en {gene_rico}, evaluar a escala en {gene_huerfano}\n{'='*70}")
    wt_seq_rico = ft.leer_fasta(seq_rico_path)
    variantes_rico = ft.cargar_dataset_variantes(dataset_rico_json)
    modelo = ft.entrenar_finetuning(gene_rico, wt_seq_rico, variantes_rico, tokenizer)

    wt_seq_huerfano = ft.leer_fasta(seq_huerfano_path)
    rho_frozen, p_frozen, n_frozen = ft.evaluar_en_frozen(
        modelo, gene_huerfano, wt_seq_huerfano, tokenizer, frozen[frozen_key], es_msh6)
    print(f"  Contra el set congelado (n={n_frozen}): rho={rho_frozen:.4f} (p={p_frozen:.4f})")

    oficial = cargar_oficial(select_db_oficial)
    claves = list(oficial.keys())
    print(f"  Generando predicciones a escala para {len(claves)} variantes con prior oficial...")
    preds = predecir_a_escala(modelo, wt_seq_huerfano, claves, tokenizer)
    y_oficial = np.array([oficial[k] for k in claves])
    rho_escala, p_escala = spearmanr(y_oficial, preds)
    print(f"  Contra el prior oficial A ESCALA (n={len(claves)}): rho={rho_escala:.4f} (p={p_escala:.3g})")

    return {
        "gene_huerfano": gene_huerfano,
        "frozen": {"rho": float(rho_frozen), "p": float(p_frozen), "n": n_frozen},
        "escala": {"rho": float(rho_escala), "p": float(p_escala), "n": len(claves)},
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(ft.MODEL_NAME)
    with open("/home/jesus/paper_msh6/datos/CONJUNTO_VALIDACION_EXTERNA_CONGELADO.json") as f:
        frozen = json.load(f)
    seq_dir = "/home/jesus/paper_msh6/datos/secuencias"

    resultado = {}
    resultado["H1_MSH6"] = procesar(
        "H1", "MSH2", "/home/jesus/paper_msh6/datos/dataset_H0_MSH2.json", f"{seq_dir}/MSH2_P43246.fasta",
        "MSH6", f"{seq_dir}/MSH6_P52701.fasta", "MSH6_priors", "msh6", True, tokenizer, frozen)
    resultado["H2_PMS2"] = procesar(
        "H2", "MLH1", "/home/jesus/paper_msh6/datos/dataset_H0_MLH1.json", f"{seq_dir}/MLH1_P40692.fasta",
        "PMS2", f"{seq_dir}/PMS2_P54278.fasta", "PMS2_priors", "pms2", False, tokenizer, frozen)

    print("\n\n=== Comparacion final: LoRA a escala vs ESM-2 650M solo a escala ===")
    print("  MSH6: ESM-2 650M solo rho=0.8266 (referencia de 26_)")
    print(f"  MSH6: LoRA a escala rho={resultado['H1_MSH6']['escala']['rho']:.4f}")
    print("  PMS2: ESM-2 650M solo rho=0.7386 (referencia de 26_)")
    print(f"  PMS2: LoRA a escala rho={resultado['H2_PMS2']['escala']['rho']:.4f}")

    with open("/home/jesus/paper_msh6/datos/resultado_lora_a_escala.json", "w") as f:
        json.dump(resultado, f, indent=2)
    print("\nGuardado: datos/resultado_lora_a_escala.json")


if __name__ == "__main__":
    main()
