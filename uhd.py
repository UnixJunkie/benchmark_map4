#!/usr/bin/env python3

import rdkit
import sys
import typing
from typing import Dict
from rdkit import Chem

# chemical element of each atom in the molecule
def get_elements(mol) -> list[str]:
    res = []
    for a in mol.GetAtoms():
        res.append(a.GetSymbol())
    return res

def formula_at_radius(elements, center_i, dists, radius):
    symbols = set()
    elt2count = {}
    distances = dists[center_i]
    assert(elements[center_i] != "H") # not supposed to encode H
    for j, dist in enumerate(distances):
        if dist == radius:
            neighbor = elements[j]
            symbols.add(neighbor)
            try:
                elt2count[neighbor] += 1
            except KeyError:
                elt2count[neighbor] = 1
    res = ""
    for elt in symbols:
        count = elt2count[elt]
        if count > 1:
            res += "%s%d" % (elt, count)
        else:
            res += elt
    return res

# unfolded-counted UHD fp
def encode(mol_noH, radius, dico):
    mol = Chem.AddHs(mol_noH)
    dists = Chem.GetDistanceMatrix(mol)
    elements = get_elements(mol)
    # only consider heavy atoms !!!
    feat2count = {}
    # count UHD features in mol
    # ...
    # translate features to their integer code (index)
    for feat, count in feat2count.items():
        feat_idx = dico[feat]
        res[feat_idx] = count
    return res

# if we don't want to fold the fp
def to_dense(fp, dest_size):
    vect_shape = (dest_size,)
    res = np.zeros(vect_shape, int)
    for feat, count in fp.items():
        res[feat] = count
    return res

# FBR: max_dim is outdated
def encode_molecules_unfolded(mols, max_dim=14087):
    d = {}
    res = []
    for mol in mols:
        sparse = encode(mol, d)
        unfolded = to_dense(sparse, max_dim)
        res.append(unfolded)
    print('atom_pairs: %d features' % len(d), file=sys.stderr)
    return res

# # tests
# m = Chem.MolFromSmiles('Cn1c(=O)c2c(ncn2C)n(C)c1=O')
# dico: Dict[str,int] = {}
# fp = encode(m, dico)
# fp
# folded_fp = counting_bloom_fold(fp, 2048, 3)
# folded_fp
# inspect_vector(folded_fp)
# encode_molecules([m, m], 2048)
