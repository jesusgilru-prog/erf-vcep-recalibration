"""
Features estructurales reales, dos niveles de completitud segun lo que existe:

1. pLDDT por residuo (confianza de AlphaFold, B-factor en el PDB monomerico v6),
   disponible para los 4 genes (MSH2, MSH6, MLH1, PMS2).

2. Distancia minima al ADN y distancia minima a la cadena pareja, calculadas sobre
   el co-cristal real MSH2-MSH6-ADN (PDB 2O8C, cadenas A=MSH2, B=MSH6, E/F=ADN) --
   SOLO disponible para MutSalpha.

CORRECCION IMPORTANTE (6-ago-2026): las 9 estructuras que el PREREGISTRO.md listaba
para MutLalpha (6RMN, 6SHX, 6SNS, 4E4W, 6SNV, 4FMN, 4FMO, 5U5P, 5U5R) NO son un
co-cristal humano MLH1-PMS2. Verificado con el filtro de organismo de RCSB: 6RMN/
6SHX/6SNS/6SNV son MLH1-MLH3; 4E4W/4FMN/4FMO son MLH1-PMS1 (parece ser el homologo
de levadura o el parologo humano PMS1, no PMS2); 5U5P es MLH1+importina; 5U5R es
PMS2+importina, cada uno por separado, nunca juntos. NO EXISTE un co-cristal humano
MLH1-PMS2 en el PDB a fecha de hoy. Para MutLalpha solo se calcula pLDDT monomerico;
no hay distancia-a-pareja ni distancia-a-ADN disponibles. Se declara como limitacion,
no se rellena con un proxy.
"""
import json

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_index, index_to_one

ESTR_DIR = "/home/jesus/paper_msh6/datos/estructuras"
parser = PDBParser(QUIET=True)

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def plddt_por_posicion(gene, acc):
    structure = parser.get_structure(gene, f"{ESTR_DIR}/AF_{gene}_{acc}.pdb")
    out = {}
    for residue in structure[0]["A"]:
        if residue.id[0] != " ":
            continue
        pos = residue.id[1]
        # pLDDT esta en el B-factor de cada atomo del residuo (AlphaFold), tomamos el del CA
        if "CA" in residue:
            out[pos] = float(residue["CA"].get_bfactor())
    return out


def distancias_2o8c():
    """Devuelve {'MSH2': {pos: {'dist_adn':..., 'dist_pareja':...}}, 'MSH6': {...}}"""
    structure = parser.get_structure("2o8c", f"{ESTR_DIR}/2O8C.pdb")
    model = structure[0]
    chain_msh2 = model["A"]
    chain_msh6 = model["B"]
    dna_atoms = [atom for chain_id in ("E", "F") for atom in model[chain_id].get_atoms()]

    def min_dist_a_atomos(residue, atomos):
        if "CA" not in residue:
            return None
        ca = residue["CA"].coord
        return min(((ca - a.coord) ** 2).sum() ** 0.5 for a in atomos)

    out = {"MSH2": {}, "MSH6": {}}
    for nombre, chain, chain_pareja in [("MSH2", chain_msh2, chain_msh6), ("MSH6", chain_msh6, chain_msh2)]:
        atomos_pareja = list(chain_pareja.get_atoms())
        for residue in chain:
            if residue.id[0] != " ":
                continue
            pos = residue.id[1]
            d_dna = min_dist_a_atomos(residue, dna_atoms)
            d_pareja = min_dist_a_atomos(residue, atomos_pareja)
            if d_dna is not None:
                out[nombre][pos] = {"dist_adn": float(d_dna), "dist_pareja": float(d_pareja)}
    return out


def main():
    acc = {"MSH2": "P43246", "MSH6": "P52701", "MLH1": "P40692", "PMS2": "P54278"}
    plddt = {}
    for gene, a in acc.items():
        plddt[gene] = plddt_por_posicion(gene, a)
        print(f"{gene}: pLDDT extraido para {len(plddt[gene])} posiciones "
              f"(media={sum(plddt[gene].values())/len(plddt[gene]):.1f})")

    dist_mutsalpha = distancias_2o8c()
    for gene in ("MSH2", "MSH6"):
        n = len(dist_mutsalpha[gene])
        print(f"{gene} (2O8C): distancias calculadas para {n} posiciones")

    out = {
        "plddt": plddt,
        "distancias_mutsalpha": dist_mutsalpha,
        "nota_mutlalpha": "Sin co-cristal humano MLH1-PMS2 disponible (verificado 6-ago-2026). "
                           "Solo pLDDT monomerico para MLH1 y PMS2.",
    }
    with open("/home/jesus/paper_msh6/datos/features_estructurales.json", "w") as f:
        json.dump(out, f)
    print("\nGuardado: datos/features_estructurales.json")


if __name__ == "__main__":
    main()
