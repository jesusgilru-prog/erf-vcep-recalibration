"""
ESM-2 650M + ESM-1v (ensemble de 5) zero-shot para TP53 (393 aa, no necesita
ventaneado), mismo metodo de marginales enmascaradas que el resto del proyecto
(06_/39_). Extension a un segundo gen/enfermedad (Li-Fraumeni) fuera de la
familia MMR, a peticion del usuario tras el veredicto de 4 revisores
independientes de que el manuscrito depende demasiado de un solo gen (MSH2).
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
ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"
ESM1V_MODELOS = [f"facebook/esm1v_t33_650M_UR90S_{i}" for i in range(1, 6)]
SEQ_PATH = "/home/jesus/paper_msh6/datos/secuencias/TP53_P04637.fasta"


def main():
    seq = base.leer_fasta(SEQ_PATH)
    print(f"TP53: {len(seq)} aa. Dispositivo: {DEVICE}", flush=True)

    # ESM-2 650M
    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL)
    model = AutoModelForMaskedLM.from_pretrained(ESM2_MODEL).to(DEVICE).eval()
    t0 = time.time()
    scores = base.calcular_zeroshot("TP53", seq, tokenizer, model)
    print(f"TP53 ESM-2: {len(scores)} variantes en {time.time()-t0:.1f}s", flush=True)
    out_dir = "/home/jesus/paper_msh6/datos/esm2_zeroshot"
    os.makedirs(out_dir, exist_ok=True)
    serializable = {f"{pos}_{aa}": v for (pos, aa), v in scores.items()}
    with open(f"{out_dir}/TP53_esm2_650M_zeroshot.json", "w") as f:
        json.dump(serializable, f)
    print(f"Guardado: {out_dir}/TP53_esm2_650M_zeroshot.json", flush=True)
    del model
    torch.cuda.empty_cache()

    # ESM-1v ensemble de 5
    acumulado = {}
    for i, model_name in enumerate(ESM1V_MODELOS, 1):
        print(f"=== ESM-1v miembro {i}/5: {model_name} ===", flush=True)
        tok = AutoTokenizer.from_pretrained(model_name)
        m = AutoModelForMaskedLM.from_pretrained(model_name).to(DEVICE).eval()
        t0 = time.time()
        s = base.calcular_zeroshot("TP53", seq, tok, m)
        print(f"  {len(s)} variantes en {time.time()-t0:.1f}s", flush=True)
        for k, v in s.items():
            acumulado.setdefault(k, []).append(v)
        del m
        torch.cuda.empty_cache()
        media_parcial = {f"{p}_{a}": sum(vv)/len(vv) for (p, a), vv in acumulado.items()}
        out_dir_v = "/home/jesus/paper_msh6/datos/esm1v_zeroshot"
        os.makedirs(out_dir_v, exist_ok=True)
        with open(f"{out_dir_v}/TP53_esm1v_ensemble5_zeroshot_parcial.json", "w") as f:
            json.dump({"n_miembros_acumulados": i, "scores": media_parcial}, f)

    media = {f"{p}_{a}": sum(vv)/len(vv) for (p, a), vv in acumulado.items()}
    out_path = "/home/jesus/paper_msh6/datos/esm1v_zeroshot/TP53_esm1v_ensemble5_zeroshot.json"
    with open(out_path, "w") as f:
        json.dump(media, f)
    print(f"Guardado: {out_path} ({len(media)} variantes)", flush=True)


if __name__ == "__main__":
    main()
