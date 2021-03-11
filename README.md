## Predicting Molecular Properties From Smiles

##### Dataset
http://moleculenet.ai/datasets-1 + https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv


##### Local Setup (GPU)

```
wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda
export PATH=~/miniconda/bin:$PATH
conda update -n base conda
conda create -y --name smiles_mol python=3.6
conda activate smiles_mol
conda install -c conda-forge rdkit
pip install torch==1.7.1+gpu -f https://download.pytorch.org/whl/torch_stable.html
git clone https://github.com/CVxTz/smile
cd smile
pip install -e .
```
##### Run flask app
```
export MODEL_PATH="absolute_path_to_your_model/models/rnn.ckpt" && uvicorn app:app --host 0.0.0.0 --port 8080 --workers 2
```

##### Using docker (CPU)

Build:
```
sudo docker build . -t smiles_mol
```
Run FastAPI App:
```
sudo docker run -p 8080:8080 -i -t smiles_mol
```

Then to query do:

```
curl -X POST "http://0.0.0.0:8080/predict/" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{\"smiles\":\"O=C(CN1CCCC1)Nc1ccc(Oc2ccc(NC(=O)CN3CCCC3)cc2)cc1\"}"```
```
#### CLI

##### Predict property for a single Smile:
```
smiles_mol predict --model_path ~/PycharmProjects/molecule_properties_from_smiles/models/rnn.ckpt \
                   --encoding integer_sequence \
                   --smile "O=C(CN1CCCC1)Nc1ccc(Oc2ccc(NC(=O)CN3CCCC3)cc2)cc1"
```

##### Evaluate the model on a CSV

```
smiles_mol evaluate --model_path ~/PycharmProjects/molecule_properties_from_smiles/models/rnn.ckpt\
                    --encoding integer_sequence \
                    --csv_path ~/PycharmProjects/molecule_properties_from_smiles/data/HIV_test.csv

```

##### Train model

```
smiles_mol train --model_dir_path ~/PycharmProjects/molecule_properties_from_smiles/models/ \
                 --encoding integer_sequence \
                 --train_path ~/PycharmProjects/molecule_properties_from_smiles/data/HIV_train.csv \
                 --test_path ~/PycharmProjects/molecule_properties_from_smiles/data/HIV_test.csv \
                 --model_name rnn \
                 --epochs 20
```

### Description

In this project we try to predict some binary molecular properties given their smile representation.

