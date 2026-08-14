"""
Benchmark honesto contra los predictores establecidos del grupo A de la Propuesta 7
(item E1 del plan del panel de revision, 6-ago-2026), via myvariant.info/dbNSFP:
AlphaMissense, REVEL, CADD, EVE, PrimateAI, ClinPred, VARITY. Ninguno se habia
probado hasta ahora -- son "los mejores modelos ya publicados", la pregunta
directa que planteo el usuario.

Se consulta variante a variante (posicion genomica hg38 derivada de ClinVar) contra
el mismo conjunto congelado (inCAMA/CIMRA), mismas metricas que el resto del
proyecto.
"""
import csv
import json
import re
import sys
import time
import urllib.request
import urllib.parse
import urllib.error

import numpy as np
from scipy.stats import spearmanr

csv.field_size_limit(sys.maxsize)

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}


def f_uni(s):
    return float(s.replace("−", "-").strip())


def construir_indice_clinvar(genes):
    """Un unico barrido de variant_summary.txt (9M filas) para los genes pedidos,
    en vez de un barrido por variante -- 70 barridos completos era lo que colgaba
    el script."""
    indice = {}
    with open("/home/jesus/paper_msh6/datos/variant_summary.txt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["Assembly"] != "GRCh38":
                continue
            genes_fila = set(row["GeneSymbol"].split("|"))
            interseccion = genes_fila & genes
            if not interseccion:
                continue
            m = re.search(r"\(p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})\)", row["Name"])
            if not m:
                continue
            aa1_3, p, aa2_3 = m.groups()
            aa2 = AA3_TO_1.get(aa2_3.capitalize())
            if aa2 is None:
                continue
            for gene in interseccion:
                indice[(gene, int(p), aa2)] = (
                    row["Chromosome"], row["PositionVCF"], row["ReferenceAlleleVCF"], row["AlternateAlleleVCF"])
    return indice


def consultar_myvariant(chrom, pos, ref, alt, max_reintentos=3):
    hgvs_id = f"chr{chrom}:g.{pos}{ref}>{alt}"
    url = "https://myvariant.info/v1/variant/" + urllib.parse.quote(hgvs_id, safe="")
    qs = urllib.parse.urlencode({
        "assembly": "hg38",
        "fields": "dbnsfp.alphamissense,dbnsfp.revel,dbnsfp.cadd,dbnsfp.eve,dbnsfp.primateai,dbnsfp.clinpred,dbnsfp.varity",
    })
    for intento in range(max_reintentos):
        try:
            with urllib.request.urlopen(f"{url}?{qs}", timeout=20) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {}
            if intento == max_reintentos - 1:
                return {"error": str(e)}
            time.sleep(2)
        except Exception as e:
            if intento == max_reintentos - 1:
                return {"error": str(e)}
            time.sleep(2)


def extraer_scores(d):
    dbnsfp = d.get("dbnsfp", {}) if isinstance(d, dict) else {}

    def sacar(campo, subcampo="score"):
        v = dbnsfp.get(campo, {})
        if not isinstance(v, dict):
            return None
        s = v.get(subcampo)
        if isinstance(s, list):
            s = [x for x in s if x is not None]
            return float(np.mean(s)) if s else None
        return float(s) if s is not None else None

    def sacar_anidado(*campos, subcampo="score"):
        """Para dbnsfp.varity.er.score: estructura anidada, no plana."""
        v = dbnsfp
        for campo in campos:
            v = v.get(campo, {}) if isinstance(v, dict) else {}
        if not isinstance(v, dict):
            return None
        s = v.get(subcampo)
        if isinstance(s, list):
            s = [x for x in s if x is not None]
            return float(np.mean(s)) if s else None
        return float(s) if s is not None else None

    return {
        "alphamissense": sacar("alphamissense"),
        "revel": sacar("revel"),
        "cadd": sacar("cadd", "phred") or sacar("cadd", "raw_score"),
        "eve": sacar("eve"),
        "primateai": sacar("primateai"),
        "clinpred": sacar("clinpred"),
        # subcampo real: dbnsfp.varity.er.score (verificado 6-ago-2026 via metadata/fields de
        # myvariant.info -- "varity_r" y "varity" planos NUNCA tuvieron datos, de ahi el n=0
        # del primer intento, log 20260806_151929).
        "varity": sacar_anidado("varity", "er"),
    }


def main():
    with open("/home/jesus/paper_msh6/datos/CONJUNTO_VALIDACION_EXTERNA_CONGELADO.json") as f:
        frozen = json.load(f)

    filas_por_variante = []
    for gene, key, oddspath_field in [("MSH6", "msh6", "oddspath_functional_inCAMA"),
                                        ("PMS2", "pms2", "oddspath_CIMRA")]:
        for entrada in frozen[key]:
            if key == "msh6":
                m = re.match(r"^([A-Z])(\d+)([A-Z])$", entrada["variant_1letter"].rstrip("g"))
                pos, mut_aa = int(m.group(2)), m.group(3)
            else:
                m = re.match(r"^[Pp]\.\s*([A-Za-z]{3})(\d+)([A-Za-z]{3})$", entrada["variant_protein"].strip())
                pos, mut_aa = int(m.group(2)), AA3_TO_1[m.group(3).capitalize()]
            y = np.log10(f_uni(entrada[oddspath_field]))
            filas_por_variante.append({"gene": gene, "pos": pos, "mut_aa": mut_aa, "y": y})

    print(f"Total variantes a consultar: {len(filas_por_variante)}")
    indice = construir_indice_clinvar({"MSH6", "PMS2"})
    print(f"Indice ClinVar construido: {len(indice)} variantes missense indexadas")

    for i, fila in enumerate(filas_por_variante):
        coords = indice.get((fila["gene"], fila["pos"], fila["mut_aa"]))
        if coords is None:
            fila["scores"] = None
            print(f"  [{i+1}/{len(filas_por_variante)}] {fila['gene']} pos{fila['pos']}{fila['mut_aa']}: "
                  f"sin coordenadas ClinVar")
            continue
        chrom, pos_vcf, ref, alt = coords
        d = consultar_myvariant(chrom, pos_vcf, ref, alt)
        scores = extraer_scores(d)
        fila["scores"] = scores
        print(f"  [{i+1}/{len(filas_por_variante)}] {fila['gene']} pos{fila['pos']}{fila['mut_aa']}: {scores}",
              flush=True)

    with open("/home/jesus/paper_msh6/datos/benchmark_predictores_raw.json", "w") as f:
        json.dump(filas_por_variante, f, indent=2)

    print("\n=== Correlaciones por predictor y gen ===")
    resultado = {}
    for gene in ("MSH6", "PMS2"):
        filas_gen = [f for f in filas_por_variante if f["gene"] == gene and f["scores"]]
        resultado[gene] = {}
        for predictor in ("alphamissense", "revel", "cadd", "eve", "primateai", "clinpred", "varity"):
            ys, ss = [], []
            for f in filas_gen:
                s = f["scores"].get(predictor)
                if s is not None:
                    ys.append(f["y"])
                    ss.append(s)
            if len(ys) < 5:
                print(f"{gene} {predictor}: n={len(ys)}, insuficiente")
                resultado[gene][predictor] = {"n": len(ys), "rho": None}
                continue
            rho, p = spearmanr(ys, ss)
            print(f"{gene} {predictor}: n={len(ys)}, rho={rho:+.4f} (p={p:.4g})")
            resultado[gene][predictor] = {"n": len(ys), "rho": float(rho), "p": float(p)}

    with open("/home/jesus/paper_msh6/datos/resultado_benchmark_predictores.json", "w") as f:
        json.dump(resultado, f, indent=2)
    print("\nGuardado: datos/resultado_benchmark_predictores.json")
    print("\nComparar: ESM-2 650M solo -> MSH6 rho=0.770; PMS2 rho=0.767")


if __name__ == "__main__":
    main()
