"""
Version barata y ya disponible de la 'celda de validacion independiente' que
propuso Claude (BRCA1/SGE) -- se descubrio al intentarlo que BRCA1 no sirve tal
cual: su especificacion VCEP actual (GN092/GN097, ENIGMA) ya cita un predictor
moderno servido en dbNSFP, no el portal manual MAPP/PP2 de HCI (verificado en
`32_`), asi que no es el mismo caso que MSH6/PMS2 y el portal LOVD de HCI no
sirve BRCA1_priors en absoluto (verificado: solo MLH1/MSH2/MSH6/PMS2).

Pero el proyecto YA TIENE una verdad funcional real e independiente para
MSH6 y PMS2 (inCAMA y CIMRA, ensayos CRISPR arrayed -- ni alineamiento ni
evolutivo, a diferencia de MAPP/PP2 y ESM-2): el conjunto congelado
(CONJUNTO_VALIDACION_EXTERNA_CONGELADO.json). Aqui se comprueba, para cada
variante del conjunto congelado, si el prior oficial y ESM-2 discrepan en
direccion, y si es asi, a cual de los dos le da la razon el ensayo funcional
real -- exactamente la pregunta que Claude queria resolver con BRCA1/SGE,
aplicada a los datos que el proyecto ya tiene, sin gen nuevo.
"""
import json
import re

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}


def f_uni(s):
    return float(s.replace("−", "-").strip())


def parse_protein_change(pc):
    m = re.match(r"^p\.([A-Za-z]{1,3})(\d+)([A-Za-z]{1,3})$", pc)
    if not m:
        return None
    aa1, pos, aa2 = m.groups()
    if len(aa1) == 3:
        aa1 = AA3_TO_1.get(aa1.capitalize())
    if len(aa2) == 3:
        aa2 = AA3_TO_1.get(aa2.capitalize())
    if aa1 is None or aa2 is None:
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


def cargar_esm(gene):
    path = f"/home/jesus/paper_msh6/datos/esm2_zeroshot/{gene}_esm2_650M_zeroshot.json"
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        pos_str, aa = k.rsplit("_", 1)
        out[(int(pos_str), aa)] = -v  # convencion del proyecto: alto = mas danino
    return out


def evidencia_oficial(prior_p):
    if prior_p > 0.81:
        return "patogenico"
    elif prior_p > 0.68:
        return "patogenico"
    elif prior_p < 0.11:
        return "benigno"
    else:
        return "sin_evidencia"


def procesar(gene, select_db, frozen_entries, es_msh6, umbral_esm_percentil):
    oficial = cargar_oficial(select_db)
    esm = cargar_esm(gene)
    esm_vals_todos = sorted(esm.values())
    n = len(esm_vals_todos)
    # umbrales ESM-2 por percentil (proxy simple de direccion, ya que no hay
    # umbral oficial propio para ESM-2): terciles superior/inferior.
    p33 = esm_vals_todos[int(n * 0.33)]
    p67 = esm_vals_todos[int(n * 0.67)]

    print(f"\n{'='*70}\n{gene}: arbitraje con verdad funcional real (frozen set)\n{'='*70}")
    n_discordantes = 0
    n_gana_oficial = 0
    n_gana_esm = 0
    n_empate_o_no_concluyente = 0
    detalle = []
    for entrada in frozen_entries:
        if es_msh6:
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

        key = (pos, mut_aa)
        if key not in oficial or key not in esm:
            continue
        ev_of = evidencia_oficial(oficial[key])
        esm_v = esm[key]
        ev_esm = "patogenico" if esm_v >= p67 else ("benigno" if esm_v <= p33 else "sin_evidencia")

        # verdad funcional real: oddspath > 1 => patogenico (PS3-like), < 1 => benigno (BS3-like)
        verdad = "patogenico" if oddspath > 1 else "benigno"

        discordan = (ev_of == "patogenico" and ev_esm == "benigno") or (ev_of == "benigno" and ev_esm == "patogenico")
        fila = {"variante": f"{pos}{mut_aa}", "oficial": ev_of, "esm2": ev_esm,
                "oddspath_funcional": oddspath, "verdad_funcional": verdad, "discordan": discordan}
        detalle.append(fila)
        if discordan:
            n_discordantes += 1
            acierta_of = ev_of == verdad
            acierta_esm = ev_esm == verdad
            if acierta_of and not acierta_esm:
                n_gana_oficial += 1
            elif acierta_esm and not acierta_of:
                n_gana_esm += 1
            else:
                n_empate_o_no_concluyente += 1
            print(f"  DISCORDANTE {fila['variante']}: oficial={ev_of}, ESM-2={ev_esm}, "
                  f"verdad funcional (OddsPath={oddspath:.3g})={verdad} "
                  f"-> {'gana oficial' if acierta_of and not acierta_esm else 'gana ESM-2' if acierta_esm and not acierta_of else 'ninguno/ambos'}")

    print(f"\n  Total variantes evaluables (con oficial+ESM-2+funcional real): {len(detalle)}")
    print(f"  Discordantes oficial vs ESM-2: {n_discordantes}")
    print(f"  De esas: gana oficial={n_gana_oficial}, gana ESM-2={n_gana_esm}, "
          f"ninguno concluyente={n_empate_o_no_concluyente}")

    return {"gene": gene, "n_evaluables": len(detalle), "n_discordantes": n_discordantes,
            "n_gana_oficial": n_gana_oficial, "n_gana_esm": n_gana_esm,
            "n_no_concluyente": n_empate_o_no_concluyente, "detalle": detalle}


def main():
    with open("/home/jesus/paper_msh6/datos/CONJUNTO_VALIDACION_EXTERNA_CONGELADO.json") as f:
        frozen = json.load(f)

    resultado = {}
    resultado["MSH6"] = procesar("MSH6", "MSH6_priors", frozen["msh6"], True, None)
    resultado["PMS2"] = procesar("PMS2", "PMS2_priors", frozen["pms2"], False, None)

    with open("/home/jesus/paper_msh6/datos/resultado_arbitraje_frozen_set.json", "w") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    print("\n\nGuardado: datos/resultado_arbitraje_frozen_set.json")


if __name__ == "__main__":
    main()
