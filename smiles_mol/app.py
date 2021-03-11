import os
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from smiles_mol.data_loaders import Encoding
from smiles_mol.eval import load_model, predict_single_smile

model = load_model(os.environ.get("MODEL_PATH"), encoding=Encoding.INTEGER)
model.eval()


class Smiles(BaseModel):
    smiles: str


app = FastAPI()


@app.post("/predict/")
def create_item(in_query: Smiles):
    score = predict_single_smile(model, in_query.smiles, encoding=Encoding.INTEGER)

    return {"score": score, "smiles": in_query.smiles}
