"""
ESM-1v (Meier et al. 2021, el mismo paper que ya se cita para el metodo de
marginales enmascaradas usado en todo el proyecto) es un ensemble de 5 modelos
ENTRENADOS Y VALIDADOS ESPECIFICAMENTE PARA PREDECIR EFECTO DE VARIANTES -- a
diferencia de ESM-2, que es un modelo generalista de proteinas. Nunca se habia
probado el modelo en si, solo el metodo aplicado sobre ESM-2. A peticion del
usuario ("has probado todo tipo de modelos... de hugging face"), se anade aqui
para los 4 genes del proyecto (MSH2, MSH6, MLH1, PMS2).

Score final = media de los 5 miembros del ensemble (protocolo estandar de
Meier et al. 2021), no solo el primero.
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
MODELOS = [f"facebook/esm1v_t33_650M_UR90S_{i}" for i in range(1, 6)]
# ESM-1v (a diferencia de ESM-2, que usa posicion rotativa) usa posicion
# ABSOLUTA aprendida con tabla de tamano fijo -- max_position_embeddings=1026
# (verificado via AutoConfig), falla con IndexError/CUDA assert en secuencias
# mas largas (MSH6, 1360 aa). Ventana deslizante para las que no caben enteras.
VENTANA_MAX = 1000  # margen bajo 1026 para <cls>/<eos> y seguridad


def calcular_zeroshot_ventaneado(gene, seq, tokenizer, model, batch_size=16):
    """Igual que base.calcular_zeroshot pero, si la secuencia no cabe entera,
    la trocea en ventanas solapadas de VENTANA_MAX residuos y para cada
    posicion usa la ventana donde queda mas centrada (mejor contexto)."""
    L = len(seq)
    if L <= VENTANA_MAX:
        return base.calcular_zeroshot(gene, seq, tokenizer, model, batch_size=batch_size)

    paso = VENTANA_MAX // 2
    ventanas = []
    inicio = 0
    while inicio < L:
        fin = min(inicio + VENTANA_MAX, L)
        ventanas.append((inicio, fin))
        if fin == L:
            break
        inicio += paso
    print(f"  {gene}: secuencia de {L}aa > {VENTANA_MAX}, {len(ventanas)} ventanas solapadas", flush=True)

    scores_por_ventana = {}
    for (v_inicio, v_fin) in ventanas:
        sub_seq = seq[v_inicio:v_fin]
        sub_scores = base.calcular_zeroshot(gene, sub_seq, tokenizer, model, batch_size=batch_size)
        for (pos_local, aa), score in sub_scores.items():
            pos_global = pos_local + v_inicio
            centro = (v_fin - v_inicio) / 2
            distancia_al_centro = abs(pos_local - centro)
            actual = scores_por_ventana.get((pos_global, aa))
            if actual is None or distancia_al_centro < actual[1]:
                scores_por_ventana[(pos_global, aa)] = (score, distancia_al_centro)

    return {k: v[0] for k, v in scores_por_ventana.items()}


def main():
    seq_dir = "/home/jesus/paper_msh6/datos/secuencias"
    genes = {
        "MSH2": f"{seq_dir}/MSH2_P43246.fasta",
        "MSH6": f"{seq_dir}/MSH6_P52701.fasta",
        "MLH1": f"{seq_dir}/MLH1_P40692.fasta",
        "PMS2": f"{seq_dir}/PMS2_P54278.fasta",
    }
    out_dir = "/home/jesus/paper_msh6/datos/esm1v_zeroshot"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Dispositivo: {DEVICE}")
    scores_por_gen = {g: {} for g in genes}

    for i, model_name in enumerate(MODELOS, 1):
        print(f"\n=== Miembro {i}/5: {model_name} ===")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name).to(DEVICE).eval()

        for gene, path in genes.items():
            seq = base.leer_fasta(path)
            t0 = time.time()
            scores = calcular_zeroshot_ventaneado(gene, seq, tokenizer, model)
            print(f"  {gene}: {len(scores)} variantes en {time.time()-t0:.1f}s")
            for k, v in scores.items():
                scores_por_gen[gene].setdefault(k, []).append(v)

        del model
        torch.cuda.empty_cache()

        # guardado incremental tras cada miembro del ensemble -- si algo falla
        # a mitad, no se pierde el trabajo ya hecho (fallo real 11-ago-2026:
        # crash en MSH6 por limite de posicion de ESM-1v, sin nada guardado)
        for gene, acumulado in scores_por_gen.items():
            media_parcial = {f"{pos}_{aa}": sum(vals) / len(vals) for (pos, aa), vals in acumulado.items()}
            with open(f"{out_dir}/{gene}_esm1v_ensemble5_zeroshot_parcial.json", "w") as f:
                json.dump({"n_miembros_acumulados": i, "scores": media_parcial}, f)
        print(f"  [guardado parcial tras {i}/5 miembros]")

    for gene, acumulado in scores_por_gen.items():
        media = {f"{pos}_{aa}": sum(vals) / len(vals) for (pos, aa), vals in acumulado.items()}
        out_path = f"{out_dir}/{gene}_esm1v_ensemble5_zeroshot.json"
        with open(out_path, "w") as f:
            json.dump(media, f)
        print(f"Guardado: {out_path} ({len(media)} variantes, media de 5 miembros)")


if __name__ == "__main__":
    main()
