import typer
from dataset.cli import app as dataset_app

app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(dataset_app, name="dataset")


def main():
    app()
