"""
Extiende el hallazgo central (4 modelos concuerdan mucho entre si pero aciertan
poco frente a verdad funcional real) de MSH2 (unico gen con DMS masivo propio)
a MSH6 y PMS2 -- los genes reales del proyecto -- usando el conjunto congelado
inCAMA/CIMRA (ensayos funcionales arrayed reales, n pequeno pero genuinamente
independiente). Si el patron se repite tambien en MSH6/PMS2, deja de ser
"un fenomeno de MSH2" y pasa a ser el patron de la familia MMR completa,
justo los genes que le importan al proyecto.
"""
import json
import re

import numpy as np
from scipy.stats import spearmanr

import importlib
c40 = importlib.import_module("40_comparar_alphamissense")
c41 = importlib.import_module("41_comparar_esm1v")

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}


def f_uni(s):
    return float(s.replace("−", "-").strip())


def cargar_verdad_frozen(gene_key, frozen):
    """Devuelve {(pos, mut_aa): log10(OddsPath)} -- alto = mas patogenico,
    misma convencion que el resto del proyecto."""
    out = {}
    for entrada in frozen[gene_key]:
        if gene_key == "msh6":
            v = entrada["variant_1letter"].rstrip("g")
            m = re.match(r"^([A-Z])(\d+)([A-Z])$", v)
            pos, mut_aa = int(m.group(2)), m.group(3)
            oddspath = f_uni(entrada["oddspath_functional_inCAMA"])
        else:
            m = re.match(r"^[Pp]\.\s*([A-Za-z]{3})(\d+)([A-Za-z]{3})$", entrada["variant_protein"].strip())
            aa1_3, pos, aa2_3 = m.groups()
            mut_aa = AA3_TO_1[aa2_3.capitalize()]
            pos = int(pos)
            oddspath = f_uni(entrada["oddspath_CIMRA"])
        out[(pos, mut_aa)] = np.log10(oddspath)
    return out


def evaluar_gen(gene, select_db, gene_key, frozen):
    print(f"\n{'='*70}\n{gene}: 4 modelos vs verdad funcional real (conjunto congelado)\n{'='*70}")
    verdad = cargar_verdad_frozen(gene_key, frozen)
    oficial = c40.cargar_oficial(select_db)
    esm2 = c40.cargar_esm(gene)
    am = c40.cargar_alphamissense(gene)
    esm1v = c41.cargar_esm1v(gene)

    resultado = {}
    for nombre, modelo in [("oficial", oficial), ("ESM-2", esm2), ("AlphaMissense", am), ("ESM-1v", esm1v)]:
        comunes = sorted(set(modelo) & set(verdad))
        if len(comunes) < 5:
            print(f"  {nombre}: n={len(comunes)}, insuficiente")
            resultado[nombre] = {"n": len(comunes), "insuficiente": True}
            continue
        y1 = np.array([modelo[k] for k in comunes])
        y2 = np.array([verdad[k] for k in comunes])
        rho, p = spearmanr(y1, y2)
        print(f"  {nombre} vs verdad funcional real: rho={rho:+.4f} (n={len(comunes)}, p={p:.3g})")
        resultado[nombre] = {"rho": float(rho), "n": len(comunes), "p": float(p)}
    return resultado


def main():
    with open("/home/jesus/paper_msh6/datos/CONJUNTO_VALIDACION_EXTERNA_CONGELADO.json") as f:
        frozen = json.load(f)

    resultado = {}
    resultado["MSH6"] = evaluar_gen("MSH6", "MSH6_priors", "msh6", frozen)
    resultado["PMS2"] = evaluar_gen("PMS2", "PMS2_priors", "pms2", frozen)

    print(f"\n{'='*70}\nRESUMEN: acierto de los 4 modelos frente a verdad funcional real, LOS 3 GENES\n{'='*70}")
    print(f"{'Modelo':<15}{'MSH2 (n=grande)':<18}{'MSH6 (n=19)':<15}{'PMS2 (n=51)':<15}")
    resumen_msh2 = {"oficial": 0.2957, "ESM-2": 0.2796, "AlphaMissense": 0.3514, "ESM-1v": 0.4002}
    for nombre in ["oficial", "ESM-2", "AlphaMissense", "ESM-1v"]:
        r6 = resultado["MSH6"].get(nombre, {})
        r2 = resultado["PMS2"].get(nombre, {})
        v6 = f"{r6['rho']:+.3f}" if "rho" in r6 else "n/a"
        v2 = f"{r2['rho']:+.3f}" if "rho" in r2 else "n/a"
        print(f"{nombre:<15}{resumen_msh2[nombre]:<18.4f}{v6:<15}{v2:<15}")

    with open("/home/jesus/paper_msh6/datos/resultado_4modelos_vs_frozen_msh6_pms2.json", "w") as f:
        json.dump(resultado, f, indent=2)
    print("\nGuardado: datos/resultado_4modelos_vs_frozen_msh6_pms2.json")


if __name__ == "__main__":
    main()
