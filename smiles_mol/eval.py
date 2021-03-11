from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from smiles_mol.data_loaders import Encoding, SmileDataset
from smiles_mol.models import RNN, accuracy


def load_model(model_path: str, encoding: Encoding = Encoding.INTEGER):
    if encoding == Encoding.INTEGER:
        model = RNN()
    else:
        raise NotImplementedError

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path)["state_dict"])
    else:
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))["state_dict"]
        )

    model.eval()

    return model


def eval_model(
    model_path: str,
    test_loader: torch.utils.data.DataLoader,
    cuda: bool = False,
    encoding: Encoding = Encoding.INTEGER,
):
    """
    Eval model and compare f1 score to random baseline.
    :param model_path:
    :param test_loader:
    :param cuda:
    :param encoding:
    :return:
    """
    model = load_model(model_path, encoding)

    y_full = []
    y_hat_full = []

    for x, y in tqdm(test_loader):
        x = x.cuda() if cuda else x
        y = y.cuda() if cuda else y

        with torch.no_grad():
            y_pred = model(x)

        y_full.append(y)
        y_hat_full.append(y_pred)

    y_full = torch.cat(y_full, dim=0)
    y_hat_full = torch.cat(y_hat_full, dim=0)

    acc = accuracy(y_full, y_hat_full)

    y_full = y_full.cpu().detach().numpy().ravel()
    y_hat_full = y_hat_full.cpu().numpy().ravel()

    f1_c_1 = f1_score(y_full.astype(np.int), (y_hat_full > 0.5).astype(np.int))
    f1_c_0 = f1_score(
        y_full.astype(np.int), (y_hat_full > 0.5).astype(np.int), pos_label=0
    )

    auc = roc_auc_score(y_full.astype(np.int), y_hat_full)

    f1_random_c_1 = f1_score(
        y_full.astype(np.int),
        (np.random.uniform(size=y_full.shape) > 0.5).astype(np.int),
    )
    f1_random_c_0 = f1_score(
        y_full.astype(np.int),
        (np.random.uniform(size=y_full.shape) > 0.5).astype(np.int),
        pos_label=0,
    )

    print(
        classification_report(y_full.astype(np.int), (y_hat_full > 0.5).astype(np.int))
    )

    return {
        "auc": auc,
        "accuracy": acc,
        "f1_class_1": f1_c_1,
        "f1_class_0": f1_c_0,
        "f1_random_class_1": f1_random_c_1,
        "f1_random_class_0": f1_random_c_0,
    }


def eval_from_csv(
    model_path: str,
    text_col: str,
    csv_path: str,
    labels: List,
    cuda: bool = False,
    encoding: Encoding = Encoding.INTEGER,
):
    """
    Eval model on csv and compare f1 score to random baseline.
    :param model_path:
    :param text_col:
    :param csv_path:
    :param labels:
    :param cuda:
    :param encoding:
    :return:
    """
    data = pd.read_csv(csv_path)
    list_tuples = list(zip(data[text_col].tolist(), data[labels].values.tolist()))

    test_data = SmileDataset(list_tuples, encoding=encoding)

    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=8)

    return eval_model(
        model_path,
        test_loader,
        cuda=cuda,
        encoding=encoding,
    )


def predict_single_smile(model, smile: str, encoding: Encoding = Encoding.INTEGER):
    """
    Predict P1 score on single smile
    :param model:
    :param smile:
    :param encoding:
    :return:
    """
    data = SmileDataset(data=[(smile, -1)], encoding=encoding)
    X, _ = data[0]
    X = X.unsqueeze(0)

    with torch.no_grad():
        y_hat = model(X)

    return y_hat.item()


if __name__ == "__main__":

    model = load_model(model_path="../models/rnn.ckpt", encoding=Encoding.INTEGER)

    print(
        predict_single_smile(
            model, smile="CC(=O)N1C2:C:C:C:C:C:2SC2:C1:C:C:C1:C:C:C:C:C:2:1"
        )
    )

    print(
        eval_from_csv(
            model_path="../models/rnn.ckpt",
            csv_path="../data/HIV_test.csv",
            encoding=Encoding.INTEGER,
            text_col="smiles",
            labels=["HIV_active"],
        )
    )
