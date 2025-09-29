import typer
from loguru import logger

app = typer.Typer(help="PsiloQA Methods Pipeline")

@app.command("uncertainty")
def uncertainty():
    logger.info("uncertainty cli")

@app.command("encoder_train")
def encoder_train():
    ...

@app.command("encoder_eval")
def encoder_eval():
    ...