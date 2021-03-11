import click
from smiles_mol.eval import load_model, predict_single_smile, eval_from_csv
from smiles_mol.data_loaders import Encoding
from smiles_mol.train import train_model


@click.group()
def cli():
    pass


@click.command()
@click.option(
    "--model_path",
    required=True,
    help="Path to the model",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option(
    "--encoding",
    required=True,
    help="Encoding",
    type=click.Choice(["morgan_fingerprint", "integer_sequence"], case_sensitive=False),
)
@click.option(
    "--smile",
    required=True,
    help="Smile string",
    type=str,
)
def predict(model_path: str, encoding: str, smile: str):

    encoding = Encoding(encoding)
    model = load_model(model_path=model_path, encoding=encoding)

    score = predict_single_smile(model=model, smile=smile, encoding=encoding)

    click.echo(f"Score : {score}")


@click.command()
@click.option(
    "--model_path",
    required=True,
    help="Path to the model",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option(
    "--encoding",
    required=True,
    help="Encoding",
    type=click.Choice(["integer_sequence"], case_sensitive=False),
)
@click.option(
    "--csv_path",
    required=True,
    help="CSV path",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option("--text_col", required=False, help="text_col", type=str, default="smiles")
@click.option("--propr", required=False, help="propr", type=str, default="HIV_active")
def evaluate(model_path: str, encoding: str, csv_path: str, text_col: str, propr: str):
    encoding = Encoding(encoding)
    evaluation = eval_from_csv(
        model_path=model_path,
        csv_path=csv_path,
        text_col=text_col,
        labels=[propr],
        encoding=encoding,
    )
    click.echo(evaluation)


@click.command()
@click.option(
    "--model_dir_path",
    required=True,
    help="Path to the model base folder",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, readable=True),
)
@click.option(
    "--model_name",
    required=True,
    help="Model Name",
    type=str,
)
@click.option(
    "--encoding",
    required=True,
    help="Encoding",
    type=click.Choice(["integer_sequence"], case_sensitive=False),
)
@click.option(
    "--train_path",
    required=True,
    help="Train CSV path",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option(
    "--test_path",
    required=True,
    help="Test CSV path",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option(
    "--epochs",
    required=True,
    help="Epochs",
    type=int,
)
@click.option("--text_col", required=False, help="text_col", type=str, default="smiles")
@click.option("--propr", required=False, help="propr", type=str, default="HIV_active")
def train(
    model_dir_path: str,
    model_name: str,
    encoding: str,
    train_path: str,
    test_path: str,
    text_col: str,
    propr: str,
    epochs: int,
):
    encoding = Encoding(encoding)
    evaluation = train_model(
        model_dir_path=model_dir_path,
        train_path=train_path,
        test_path=test_path,
        text_col=text_col,
        labels=[propr],
        encoding=encoding,
        model_name=model_name,
        epochs=epochs,
    )
    click.echo(evaluation)


cli.add_command(predict)
cli.add_command(evaluate)
cli.add_command(train)
