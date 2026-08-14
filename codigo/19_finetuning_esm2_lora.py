"""
Ultimo mecanismo de transferencia por probar, y el mas parecido en espiritu a la
hipotesis original: en vez de usar ESM-2 congelado como una feature mas de un
modelo separado (lo que se probo y perdio contra ESM-2 solo en 4 variantes
distintas), AJUSTAR LOS PROPIOS PESOS de ESM-2 con LoRA sobre los datos DMS del
gen rico (MSH2 o MLH1), y comprobar si el modelo AJUSTADO -- que sigue siendo
fundamentalmente un modelo de lenguaje de proteinas, no un arbol de decision
sobre 4 features -- generaliza mejor al gen huerfano que el ESM-2 sin ajustar.

Referencia: "Fine-tuning Protein Language Models with Deep Mutational Scanning
improves Variant Effect Prediction" (arXiv:2405.06729) reporta hasta +25.6% de
Spearman DENTRO del propio set de DMS. La pregunta aqui es si eso se mantiene al
saltar de gen (MSH2 ajustado -> evaluado en MSH6), no solo dentro del mismo gen.

Diseno: cabeza de regresion sobre el embedding medio del token de la posicion
mutada (wildtype secuencia con la sustitucion puntual introducida), LoRA sobre
las capas de atencion, entrenado a predecir score_danino. Evaluado exactamente
igual que el resto del proyecto, contra el conjunto congelado.
"""
import json
import os

os.environ["HF_HOME"] = "/home/jesus/paper_msh6/modelos/hf_cache"

import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
from scipy.stats import spearmanr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
RNG_SEED = 20260805
torch.manual_seed(RNG_SEED)

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}


def leer_fasta(path):
    with open(path) as f:
        lines = f.readlines()
    return "".join(l.strip() for l in lines if not l.startswith(">"))


class VarianteDataset(Dataset):
    def __init__(self, wt_seq, variantes, tokenizer, max_len=1400):
        self.wt_seq = wt_seq
        self.variantes = variantes  # lista de (pos_1based, mut_aa, score_danino)
        self.tok = tokenizer

    def __len__(self):
        return len(self.variantes)

    def __getitem__(self, idx):
        pos, mut_aa, score = self.variantes[idx]
        seq = list(self.wt_seq)
        seq[pos - 1] = mut_aa
        seq = "".join(seq)
        return seq, pos, score


def collate(batch, tokenizer):
    seqs, posiciones, scores = zip(*batch)
    enc = tokenizer(list(seqs), return_tensors="pt", padding=True)
    return enc, torch.tensor(posiciones, dtype=torch.long), torch.tensor(scores, dtype=torch.float32)


class ModeloRegresion(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        self.cabeza = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, input_ids, attention_mask, posiciones):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # posiciones son 1-based sobre la secuencia; token 0 es <cls>, por eso +1
        idx = (posiciones + 1).clamp(max=out.shape[1] - 1)
        emb_pos = out[torch.arange(out.shape[0]), idx]
        return self.cabeza(emb_pos).squeeze(-1)


def cargar_dataset_variantes(dataset_json):
    with open(dataset_json) as f:
        data = json.load(f)
    return [(d["posicion"], d["mut_aa"], d["score_danino"]) for d in data]


def entrenar_finetuning(gene, wt_seq, variantes, tokenizer, epochs=3, batch_size=8, lr=1e-4):
    backbone = AutoModel.from_pretrained(MODEL_NAME)
    lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["query", "key", "value"],
                           lora_dropout=0.05, bias="none")
    backbone = get_peft_model(backbone, lora_cfg)
    modelo = ModeloRegresion(backbone).to(DEVICE)

    n_val = max(1, int(0.1 * len(variantes)))
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.permutation(len(variantes))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_vars = [variantes[i] for i in train_idx]
    val_vars = [variantes[i] for i in val_idx]

    train_ds = VarianteDataset(wt_seq, train_vars, tokenizer)
    val_ds = VarianteDataset(wt_seq, val_vars, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                           collate_fn=lambda b: collate(b, tokenizer))
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                         collate_fn=lambda b: collate(b, tokenizer))

    # Solo los parametros entrenables (LoRA + cabeza) -- evita cualquier ambiguedad
    # sobre si AdamW reserva estado para los ~650M congelados.
    params_entrenables = [p for p in modelo.parameters() if p.requires_grad]
    n_entrenables = sum(p.numel() for p in params_entrenables)
    n_total = sum(p.numel() for p in modelo.parameters())
    print(f"  [{gene}] parametros entrenables: {n_entrenables:,} de {n_total:,} totales", flush=True)
    opt = torch.optim.AdamW(params_entrenables, lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        modelo.train()
        total_loss = 0.0
        for step, (enc, pos, y) in enumerate(train_dl):
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            pos, y = pos.to(DEVICE), y.to(DEVICE)
            pred = modelo(enc["input_ids"], enc["attention_mask"], pos)
            loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(y)
            del enc, pos, y, pred, loss
            if step % 100 == 0:
                alloc = torch.cuda.memory_allocated() / 1e9
                reserv = torch.cuda.memory_reserved() / 1e9
                print(f"  [{gene}] epoch {epoch+1} step {step}/{len(train_dl)}: "
                      f"mem_allocated={alloc:.2f}GB mem_reserved={reserv:.2f}GB", flush=True)
            if step % 200 == 0:
                torch.cuda.empty_cache()
        train_loss = total_loss / len(train_ds)

        modelo.eval()
        preds, ys = [], []
        with torch.no_grad():
            for enc, pos, y in val_dl:
                enc = {k: v.to(DEVICE) for k, v in enc.items()}
                pos = pos.to(DEVICE)
                pred = modelo(enc["input_ids"], enc["attention_mask"], pos)
                preds.extend(pred.cpu().numpy())
                ys.extend(y.numpy())
        rho, _ = spearmanr(ys, preds)
        print(f"  [{gene}] epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f} val_rho={rho:.4f}", flush=True)

    return modelo


def evaluar_en_frozen(modelo, gene_huerfano, wt_seq_huerfano, tokenizer, frozen_entries, es_msh6):
    modelo.eval()
    preds, ys = [], []
    for entrada in frozen_entries:
        if es_msh6:
            v = entrada["variant_1letter"].rstrip("g")
            m = re.match(r"^([A-Z])(\d+)([A-Z])$", v)
            pos, mut_aa = int(m.group(2)), m.group(3)
            log10odds = np.log10(float(entrada["oddspath_functional_inCAMA"].replace("−", "-")))
        else:
            m = re.match(r"^[Pp]\.\s*([A-Za-z]{3})(\d+)([A-Za-z]{3})$", entrada["variant_protein"].strip())
            aa1_3, pos, aa2_3 = m.groups()
            mut_aa = AA3_TO_1[aa2_3.capitalize()]
            pos = int(pos)
            log10odds = np.log10(float(entrada["oddspath_CIMRA"].replace("−", "-")))
        seq = list(wt_seq_huerfano)
        seq[pos - 1] = mut_aa
        seq = "".join(seq)
        enc = tokenizer([seq], return_tensors="pt", padding=True)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        posT = torch.tensor([pos], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            pred = modelo(enc["input_ids"], enc["attention_mask"], posT).item()
        preds.append(pred)
        ys.append(log10odds)
    rho, p = spearmanr(ys, preds)
    return rho, p, len(ys)


def main():
    seq_dir = "/home/jesus/paper_msh6/datos/secuencias"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    with open("/home/jesus/paper_msh6/datos/CONJUNTO_VALIDACION_EXTERNA_CONGELADO.json") as f:
        frozen = json.load(f)

    resultado = {}

    print("=== Fine-tuning en MSH2, evaluar en MSH6 ===", flush=True)
    msh2_seq = leer_fasta(f"{seq_dir}/MSH2_P43246.fasta")
    msh2_vars = cargar_dataset_variantes("/home/jesus/paper_msh6/datos/dataset_H0_MSH2.json")
    modelo_msh2 = entrenar_finetuning("MSH2", msh2_seq, msh2_vars, tokenizer)
    msh6_seq = leer_fasta(f"{seq_dir}/MSH6_P52701.fasta")
    rho1, p1, n1 = evaluar_en_frozen(modelo_msh2, "MSH6", msh6_seq, tokenizer, frozen["msh6"], True)
    print(f"H1 fine-tuned: MSH2->MSH6 rho={rho1:.4f} (p={p1:.4f}, n={n1})")
    resultado["H1_finetuned"] = {"rho": float(rho1), "p": float(p1), "n": n1}

    print("\n=== Fine-tuning en MLH1, evaluar en PMS2 ===", flush=True)
    mlh1_seq = leer_fasta(f"{seq_dir}/MLH1_P40692.fasta")
    mlh1_vars = cargar_dataset_variantes("/home/jesus/paper_msh6/datos/dataset_H0_MLH1.json")
    modelo_mlh1 = entrenar_finetuning("MLH1", mlh1_seq, mlh1_vars, tokenizer)
    pms2_seq = leer_fasta(f"{seq_dir}/PMS2_P54278.fasta")
    rho2, p2, n2 = evaluar_en_frozen(modelo_mlh1, "PMS2", pms2_seq, tokenizer, frozen["pms2"], False)
    print(f"H2 fine-tuned: MLH1->PMS2 rho={rho2:.4f} (p={p2:.4f}, n={n2})")
    resultado["H2_finetuned"] = {"rho": float(rho2), "p": float(p2), "n": n2}

    with open("/home/jesus/paper_msh6/datos/resultado_finetuning_lora.json", "w") as f:
        json.dump(resultado, f, indent=2)
    print("\nGuardado: datos/resultado_finetuning_lora.json")
    print("Comparar: ESM-2 650M zero-shot solo -> MSH6 rho=0.770; PMS2 rho=0.767")


if __name__ == "__main__":
    main()
