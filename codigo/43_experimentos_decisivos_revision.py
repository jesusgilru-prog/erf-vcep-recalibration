"""
Tres analisis pedidos en la revision de ronda 1 del manuscrito (Codex+Claude+Kimi,
11/12-ago-2026), todos con datos ya en disco:

(A) Recalcular los 4 predictores sobre la MISMA interseccion comun (denominador
    identico) -- para saber si "ESM-1v es el mejor" es el modelo o el subconjunto.
(B) Test decisivo del sesgo de ascertainment: submuestrear el DMS de MSH2 para
    igualar la fraccion de casos "extremos" de inCAMA (95%) y CIMRA (82%), y ver
    si rho sube hacia el rango 0.55-0.77 -- convierte la afirmacion "we show" de
    especulacion a resultado falsable.
(C) Bootstrap con IC para TODAS las correlaciones de la Tabla 2 (n=19/51), no
    solo el par oficial-ESM2 de MSH2.
"""
import json
import re

import numpy as np
from scipy.stats import spearmanr

import importlib
c40 = importlib.import_module("40_comparar_alphamissense")
c41 = importlib.import_module("41_comparar_esm1v")

SEMILLA = 20260812
N_BOOT = 2000


def cargar_todo_msh2():
    oficial = c40.cargar_oficial("MSH2_priors")
    esm2 = c40.cargar_esm("MSH2")
    am = c40.cargar_alphamissense("MSH2")
    esm1v = c41.cargar_esm1v("MSH2")
    with open("/home/jesus/paper_msh6/datos/dataset_H0_MSH2.json") as f:
        dms = json.load(f)
    real = {(d["posicion"], d["mut_aa"]): d["score_danino"] for d in dms}
    return oficial, esm2, am, esm1v, real


def parte_a_interseccion_comun():
    print(f"\n{'='*70}\nA. Los 4 predictores sobre la MISMA interseccion (denominador comun)\n{'='*70}")
    oficial, esm2, am, esm1v, real = cargar_todo_msh2()

    comun = sorted(set(oficial) & set(esm2) & set(am) & set(esm1v) & set(real))
    print(f"Interseccion comun a los 4 modelos + verdad real: n={len(comun)}")

    y_real = np.array([real[k] for k in comun])
    resultado = {}
    rng = np.random.default_rng(SEMILLA)
    for nombre, modelo in [("oficial", oficial), ("ESM-2", esm2), ("AlphaMissense", am), ("ESM-1v", esm1v)]:
        y = np.array([modelo[k] for k in comun])
        rho, p = spearmanr(y, y_real)
        boots = []
        idx = np.arange(len(comun))
        for _ in range(N_BOOT):
            bi = rng.choice(idx, len(idx), replace=True)
            r, _ = spearmanr(y[bi], y_real[bi])
            if not np.isnan(r):
                boots.append(r)
        ci = np.percentile(boots, [2.5, 97.5])
        print(f"  {nombre}: rho={rho:.4f} (n={len(comun)}, p={p:.3g}), IC95%=[{ci[0]:.4f}, {ci[1]:.4f}]")
        resultado[nombre] = {"rho": float(rho), "n": len(comun), "p": float(p),
                              "ci95": [float(ci[0]), float(ci[1])]}

    # tambien: ESM-2 y ESM-1v sobre el DMS COMPLETO (n grande), para ver si el n mayor
    # explica la ventaja de ESM-1v o si se sostiene igual restringido
    print("\n  Para contraste, ESM-2 y ESM-1v sobre el DMS COMPLETO (todas las variantes con score):")
    for nombre, modelo in [("ESM-2", esm2), ("ESM-1v", esm1v)]:
        comunes_full = sorted(set(modelo) & set(real))
        y = np.array([modelo[k] for k in comunes_full])
        yr = np.array([real[k] for k in comunes_full])
        rho, p = spearmanr(y, yr)
        print(f"    {nombre} (n={len(comunes_full)}): rho={rho:.4f}")
        resultado[f"{nombre}_dms_completo"] = {"rho": float(rho), "n": len(comunes_full)}

    return resultado, comun


def parte_b_test_ascertainment(comun_a):
    print(f"\n{'='*70}\nB. Test decisivo: submuestrear MSH2 por extremidad, ¿sube rho?\n{'='*70}")
    oficial, esm2, am, esm1v, real = cargar_todo_msh2()

    # extremidad: z-score robusto (MAD) de score_danino sobre TODO el DMS (poblacion de referencia)
    todos_scores = np.array(list(real.values()))
    mediana = np.median(todos_scores)
    mad = np.median(np.abs(todos_scores - mediana)) * 1.4826  # factor de consistencia normal
    print(f"  DMS completo: mediana={mediana:.3f}, MAD robusto={mad:.3f}")

    def z(v):
        return abs(v - mediana) / mad if mad > 0 else 0.0

    comunes = sorted(set(esm2) & set(real))  # base para el submuestreo (ESM-2, mayor cobertura)
    y_real_full = np.array([real[k] for k in comunes])
    y_esm2_full = np.array([esm2[k] for k in comunes])
    z_scores = np.array([z(real[k]) for k in comunes])

    rho_sin_filtrar, _ = spearmanr(y_esm2_full, y_real_full)
    print(f"  ESM-2 vs verdad real, SIN filtrar (n={len(comunes)}): rho={rho_sin_filtrar:.4f}")

    resultado = {"rho_sin_filtrar": float(rho_sin_filtrar), "n_sin_filtrar": len(comunes)}
    rng = np.random.default_rng(SEMILLA)

    for etiqueta, frac_extremo in [("inCAMA-like (95% extremo)", 0.95), ("CIMRA-like (82% extremo)", 0.82)]:
        # tomar el (frac_extremo*100)% de variantes con mayor |z| (mas extremas),
        # descartando el resto central (ambiguo) -- igual que hacen los sets curados
        umbral_pos = int((1 - frac_extremo) * len(comunes))  # cuantas quedan fuera (el centro)
        orden = np.argsort(z_scores)  # ascendente: primero las mas centrales/ambiguas
        idx_mantener = orden[umbral_pos:]  # descartamos las mas centrales
        y_e = y_esm2_full[idx_mantener]
        y_r = y_real_full[idx_mantener]
        rho_sub, p_sub = spearmanr(y_e, y_r)

        boots = []
        idxs = np.arange(len(idx_mantener))
        for _ in range(N_BOOT):
            bi = rng.choice(idxs, len(idxs), replace=True)
            r, _ = spearmanr(y_e[bi], y_r[bi])
            if not np.isnan(r):
                boots.append(r)
        ci = np.percentile(boots, [2.5, 97.5])

        print(f"  {etiqueta}: n={len(idx_mantener)}, rho={rho_sub:.4f} (p={p_sub:.3g}), "
              f"IC95%=[{ci[0]:.4f}, {ci[1]:.4f}]")
        resultado[etiqueta] = {"rho": float(rho_sub), "n": len(idx_mantener), "p": float(p_sub),
                                "ci95": [float(ci[0]), float(ci[1])]}

    print(f"\n  Prediccion falsable: si el sesgo de ascertainment explica el contraste, rho debe subir "
          f"desde {rho_sin_filtrar:.3f} hacia el rango 0.55-0.77 al restringir a subconjuntos extremos.")
    return resultado


def parte_c_ic_tabla2():
    print(f"\n{'='*70}\nC. IC bootstrap para TODAS las correlaciones de la Tabla 2 (MSH6/PMS2)\n{'='*70}")
    import sys
    sys.path.insert(0, "/home/jesus/paper_msh6/codigo")
    c42 = importlib.import_module("42_cuatro_modelos_vs_frozen")
    with open("/home/jesus/paper_msh6/datos/CONJUNTO_VALIDACION_EXTERNA_CONGELADO.json") as f:
        frozen = json.load(f)

    resultado = {}
    rng = np.random.default_rng(SEMILLA)
    for gene, select_db, gene_key in [("MSH6", "MSH6_priors", "msh6"), ("PMS2", "PMS2_priors", "pms2")]:
        verdad = c42.cargar_verdad_frozen(gene_key, frozen)
        oficial = c40.cargar_oficial(select_db)
        esm2 = c40.cargar_esm(gene)
        am = c40.cargar_alphamissense(gene)
        esm1v = c41.cargar_esm1v(gene)
        resultado[gene] = {}
        print(f"\n  {gene}:")
        for nombre, modelo in [("oficial", oficial), ("ESM-2", esm2), ("AlphaMissense", am), ("ESM-1v", esm1v)]:
            comunes = sorted(set(modelo) & set(verdad))
            if len(comunes) < 5:
                continue
            y1 = np.array([modelo[k] for k in comunes])
            y2 = np.array([verdad[k] for k in comunes])
            rho, p = spearmanr(y1, y2)
            boots = []
            idx = np.arange(len(comunes))
            for _ in range(N_BOOT):
                bi = rng.choice(idx, len(idx), replace=True)
                r, _ = spearmanr(y1[bi], y2[bi])
                if not np.isnan(r):
                    boots.append(r)
            ci = np.percentile(boots, [2.5, 97.5]) if boots else [np.nan, np.nan]
            print(f"    {nombre}: rho={rho:.4f} (n={len(comunes)}), IC95%=[{ci[0]:.4f}, {ci[1]:.4f}]")
            resultado[gene][nombre] = {"rho": float(rho), "n": len(comunes), "p": float(p),
                                        "ci95": [float(ci[0]), float(ci[1])]}
    return resultado


def main():
    resultado_a, comun_a = parte_a_interseccion_comun()
    resultado_b = parte_b_test_ascertainment(comun_a)
    resultado_c = parte_c_ic_tabla2()

    with open("/home/jesus/paper_msh6/datos/resultado_experimentos_decisivos_revision.json", "w") as f:
        json.dump({"A_interseccion_comun": resultado_a, "B_test_ascertainment": resultado_b,
                    "C_ic_tabla2": resultado_c}, f, indent=2)
    print("\n\nGuardado: datos/resultado_experimentos_decisivos_revision.json")


if __name__ == "__main__":
    main()
