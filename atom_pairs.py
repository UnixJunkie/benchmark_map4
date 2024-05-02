#!/usr/bin/env python3

import typing
import rdkit
from rdkit import Chem

# counted atom pairs; maybe folded using a counted-bloom filter

def nb_heavy_atom_neighbors(a: rdkit.Chem.rdchem.Atom) -> int:
    res = 0
    for neighb in a.GetNeighbors():
        if neighb.GetAtomicNum() > 1:
            res += 1
    return res

# return (#HA, #H)
def count_neighbors(a: rdkit.Chem.rdchem.Atom) -> tuple[int, int]:
    nb_heavy = nb_heavy_atom_neighbors(a)
    nb_H = a.GetTotalNumHs()
    return (nb_heavy, nb_H)

# encode by an integer what kind of ring this atom is involved in
def ring_membership(a: rdkit.Chem.rdchem.Atom) -> int:
    if a.IsInRing():
        if a.GetIsAromatic():
            return 2 # in aromatic ring
        else:
            return 1 # in aliphatic ring
    else:
        return 0 # not in ring

# a simple atom typing scheme
def type_atom_simple(a) -> list[int]:
    anum = a.GetAtomicNum()
    fc = a.GetFormalCharge()
    aro = ring_membership(a)
    heavies, hydrogens = count_neighbors(a)
    return [anum, fc, aro, heavies, hydrogens]
