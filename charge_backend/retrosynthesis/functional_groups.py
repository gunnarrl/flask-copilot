################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

try:
    from rdkit import Chem
except ImportError:
    Chem = None

FUNCTIONAL_GROUP_SMARTS = {
    # Carbonyl compounds
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "carboxylic acid": "C(=O)[OX2H1]",
    "ester": "C(=O)O[#6]",
    "amide": "C(=O)N",
    "acyl halide": "C(=O)[F,Cl,Br,I]",
    "anhydride": "C(=O)OC(=O)",
    # Alcohols / ethers
    "alcohol": "[OX2H][CX4]",
    "phenol": "[OX2H][c]",
    "ether": "[#6]-O-[#6]",
    "epoxide": "[OX2r3]1CC1",
    # Nitrogen
    "amine": "[NX3;H2,H1,H0;!$(NC=O)]",
    "imine": "[CX3]=[NX2]",
    "nitrile": "C#N",
    "nitro": "[N+](=O)[O-]",
    "azo": "[N]=[N]",
    "isocyanate": "N=C=O",
    "isothiocyanate": "N=C=S",
    "urea": "N-C(=O)-N",
    "carbamate": "O-C(=O)-N",
    # Sulfur
    "thiol": "[SX2H]",
    "thioether": "[#6]-S-[#6]",
    "sulfoxide": "[SX3](=O)",
    "sulfone": "[SX4](=O)(=O)",
    "sulfonamide": "S(=O)(=O)N",
    "sulfonic acid": "S(=O)(=O)[OX2H]",
    # Phosphorus
    "phosphate": "P(=O)(O)(O)",
    # Unsaturation
    "alkene": "C=C",
    "alkyne": "C#C",
    # Rings
    "aromatic ring": "a1aaaaa1",
    # Halogens
    "organohalide": "[#6][F,Cl,Br,I]",
    # Hydrocarbon
    "alkyl substituent": "[CX4][a]",
}


def functional_groups_from_smiles(smiles):
    """Return simple SMARTS-matched functional group names."""
    if Chem is None:
        return []

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    groups = []
    for name, smarts in FUNCTIONAL_GROUP_SMARTS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and mol.HasSubstructMatch(pattern):
            groups.append(name)
    return groups
