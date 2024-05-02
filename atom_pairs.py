#!/usr/bin/env python3

import numpy as np
import random
import rdkit
import sys
import typing
from typing import Dict
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

def fst(a_b):
    a, _b = a_b
    return a

def snd(a_b):
    _a, b = a_b
    return b

def dict_contains(d, k):
    try:
        _v = d[k]
        return True
    except KeyError:
        return False

# counted atom pairs encoding
def encode(mol, dico):
    dists = Chem.GetDistanceMatrix(mol)
    type_atoms = []
    for a in mol.GetAtoms():
        # only consider heavy atoms
        if a.GetAtomicNum() > 1:
            t_a = (type_atom_simple(a), a)
            type_atoms.append(t_a)
    # sort by atom types (canonicalization of pairs)
    type_atoms.sort(key=fst)
    n = len(type_atoms)
    feat2count = {}
    # count AP features in mol
    for i in range(n - 1):
        a_t, a = type_atoms[i]
        a_i = a.GetIdx()
        for j in range(i + 1, n):
            b_t, b = type_atoms[j]
            b_i = b.GetIdx()
            dist = dists[a_i][b_i]
            feat = (a_t, dist, b_t)
            # python cannot create a dictionary w/ list keys!
            feat = str(feat)
            # maintain global feature dictionary
            if not dict_contains(dico, feat):
                # reserve index 0 for unknown features
                feat_idx = len(dico) + 1
                dico[feat] = feat_idx
            if dict_contains(feat2count, feat):
                prev_count = feat2count[feat]
                feat2count[feat] = prev_count + 1
            else:
                feat2count[feat] = 1
    res = {}
    # translate features to their integer code (index)
    for feat, count in feat2count.items():
        feat_idx = dico[feat]
        res[feat_idx] = count
    return res

# fold fp using a counting Bloom filter
def counting_bloom_fold(fp, dest_size, k):
    vect_shape = (dest_size,)
    res = np.zeros(vect_shape, int)
    for feat, count in fp.items():
        # belt & shoulder straps
        assert(type(feat) == int and type(count) == int)
        random.seed(feat)
        for _i in range(k):
            key = random.randrange(dest_size)
            res[key] = res[key] + count
    return res

def inspect_vector(v):
    n = len(v)
    for i in range(n):
        count = v[i]
        if count > 0:
            print('%d: %d' % (i, count), file=sys.stderr)

def encode_molecules(mols, dest_size):
    d = {}
    res = []
    for mol in mols:
        unfolded = encode(mol, d)
        # this 3 is COMPLETELY arbitrary; could be 2; could be 5...
        folded = counting_bloom_fold(unfolded, dest_size, 3)
        res.append(folded)
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
