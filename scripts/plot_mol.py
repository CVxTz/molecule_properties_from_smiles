from rdkit import Chem
from rdkit.Chem import Draw

import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--smiles", default="CCC1=[O+][Cu-3]2([O+]=C(CC)C1)[O+]=C(CC)CC(CC)=[O+]2"
    )
    parser.add_argument("--out_path", default="../data/mol.png")

    args = parser.parse_args()

    mol = Chem.MolFromSmiles(args.smiles)

    img = Draw.MolToImage(mol)

    img.save(open(args.out_path, "wb"))

    imgplot = plt.imshow(img)
    plt.show()
