import torch

from smiles_mol.models import accuracy


def test_accuracy_1d_1():
    t1 = torch.FloatTensor([0.4, 0.45, 1.0, 0.2])
    t2 = torch.FloatTensor([0.0, 1.0, 1.0, 0.0])

    acc = accuracy(t2, t1)

    assert acc == 0.75


def test_accuracy_1d_2():
    t1 = torch.FloatTensor([0.4, 0.45, 1.0, 0.2])
    t2 = torch.IntTensor([0, 1, 1, 0])

    acc = accuracy(t2, t1)

    assert acc == 0.75


def test_accuracy_2d_1():
    t1 = torch.FloatTensor([[0.4, 0.45, 1.0, 0.2], [0.4, 0.55, 1.0, 0.2]])
    t2 = torch.IntTensor([[0, 1, 1, 0], [0, 1, 1, 0]])

    acc = accuracy(t2, t1)

    assert acc == 7 / 8


if __name__ == "__main__":
    test_accuracy_2d_1()
