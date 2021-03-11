from typing import List

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from smiles_mol.data_loaders import Encoding, SmileDataset
from smiles_mol.eval import eval_model
from smiles_mol.models import RNN
from smiles_mol.utils import oversample


def train_model(
    train_path: str,
    test_path: str,
    text_col: str,
    labels: List,
    model_dir_path: str = "../models/",
    model_name: str = "baseline",
    logs_path: str = "../logs",
    epochs: int = 40,
    batch_size: int = 32,
    gpus=1,
    oversample_minority_class: bool = True,
    encoding: Encoding = Encoding.INTEGER,
):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    train_val = list(
        zip(train_data[text_col].tolist(), train_data[labels].values.tolist())
    )
    test = list(zip(test_data[text_col].tolist(), test_data[labels].values.tolist()))

    for label in labels:
        print(f"Label {label} : proportion of ones = {train_data[label].mean()}")

    test_original = test

    train, val = train_test_split(
        train_val,
        test_size=0.1,
        random_state=1337,
        stratify=[a[1][0] for a in train_val],
    )
    if oversample_minority_class:
        train = oversample(
            [a for a in train if a[1][0] == 1], [a for a in train if a[1][0] == 0]
        )
        val = oversample(
            [a for a in val if a[1][0] == 1], [a for a in val if a[1][0] == 0]
        )
        test = oversample(
            [a for a in test if a[1][0] == 1], [a for a in test if a[1][0] == 0]
        )

    train_data = SmileDataset(train, encoding=encoding)
    val_data = SmileDataset(val, encoding=encoding)
    test_data = SmileDataset(test, encoding=encoding)
    test_original_data = SmileDataset(test_original, encoding=encoding)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, num_workers=8, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, num_workers=8, shuffle=False
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=8
    )
    test_original_loader = DataLoader(
        test_original_data, batch_size=batch_size, shuffle=False, num_workers=8
    )

    if encoding == Encoding.INTEGER:
        model = RNN()
    else:
        raise NotImplementedError

    logger = TensorBoardLogger(
        save_dir=logs_path,
        name=model_name,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=model_dir_path,
        filename=model_name,
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=gpus,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_loader, val_loader)

    test_result = trainer.test(test_dataloaders=test_loader)

    print(f"Model saved in {model_dir_path}/{model_name}.ckpt")

    acc = test_result[0]["test_acc"]

    test_eval = eval_model(
        f"{model_dir_path}/{model_name}.ckpt", test_original_loader, encoding=encoding
    )

    test_eval["balanced_accuracy"] = acc

    return test_eval


if __name__ == "__main__":

    for model_name, encoding in [("rnn", Encoding.INTEGER)]:
        test_eval = train_model(
            train_path="../data/HIV_train.csv",
            test_path="../data/HIV_test.csv",
            text_col="smiles",
            labels=["HIV_active"],
            encoding=encoding,
            model_name=model_name,
            epochs=6,
            oversample_minority_class=True,
        )
        print(model_name, test_eval)
