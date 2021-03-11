import torch
from smiles_mol.data_loaders import SmileDataset, Encoding


def test_smiles_dataset():
    smile = "OC1CN=C2C3:C:C:C:C:C:3C(O)(C3:C:C:C(Cl):C:C:3)N2C1,CI"
    data = SmileDataset(data=[(smile, 1)], encoding=Encoding.INTEGER)
    assert type(data[0][0]) is torch.Tensor
    assert data[0][0].numel() == 74
    assert data[0][0].sum() == 2550
