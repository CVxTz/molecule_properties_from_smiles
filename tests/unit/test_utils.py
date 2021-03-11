import numpy as np

from smiles_mol.utils import map_and_cap, oversample


def test_map_and_cap():
    smile = "OC1CN=C2C3:C:C:C:C:C:3C(O)(C3:C:C:C(Cl):C:C:3)N2C1,CI"
    smile_int = map_and_cap(smile, cap_len=25)
    print(smile_int)
    assert smile_int == [51, 39, 2, 39, 50, 81, 39, 3, 39, 4, 78, 39, 78, 39, 78, 39, 78, 39, 78, 39, 78, 4, 39, 70, 51]
    assert len(smile_int) == 25


def test_oversample():
    sampled = oversample([0] * 50, [1] * 60)

    assert len(sampled) == 120
    assert sorted(sampled) == [0] * 60 + [1] * 60
