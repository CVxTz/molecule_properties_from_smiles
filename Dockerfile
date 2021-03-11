FROM python:3.6-slim

RUN pip install torch==1.7.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
COPY requirements.txt /deploy/
COPY smiles_mol /deploy/smiles_mol
COPY setup.py /deploy/
COPY models/rnn.ckpt /models/rnn.ckpt

ENV MODEL_PATH="/models/rnn.ckpt"

WORKDIR /deploy/

RUN pip install .
RUN pip install -r requirements.txt

CMD uvicorn --app-dir smiles_mol app:app --host 0.0.0.0 --port 8080 --workers 2