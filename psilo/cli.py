import typer
from dataset.cli import app as dataset_app
from methods.cli import app as methods_app

app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(dataset_app, name="dataset")
app.add_typer(methods_app, name="methods")

def main():
    app()
