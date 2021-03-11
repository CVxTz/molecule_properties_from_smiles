import enum
from typing import List, Tuple, Optional

import torch

from smiles_mol.utils import map_and_cap


class Encoding(enum.Enum):
    INTEGER = "integer_sequence"


class SmileDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: List[Tuple[str, int]],
        max_len: Optional[int] = 74,
        encoding: Encoding = Encoding.INTEGER,
    ):
        self.data = data
        self.max_len = max_len
        self.encoding = encoding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smile, labels = self.data[idx]

        if self.encoding == Encoding.INTEGER:
            smile = map_and_cap(smile, self.max_len)
            x = torch.tensor(smile, dtype=torch.long)
        else:
            raise NotImplementedError

        y = torch.tensor(labels, dtype=torch.float)

        return x, y
