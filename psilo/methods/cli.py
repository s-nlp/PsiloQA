import typer
from loguru import logger

from datasets import load_dataset

from encoder import run_encoder_training, run_encoder_evaluation

app = typer.Typer(help="PsiloQA Methods Pipeline")

@app.command("uncertainty")
def uncertainty():
    logger.info("uncertainty cli")

@app.command("encoder_train")
def encoder_train(
    model_name: str = typer.Option("answerdotai/ModernBERT-base", help="Name of the model to train"),
    output_dir: str = typer.Option("output/hallucination_detector", help="Directory to save outputs"),
    batch_size: int = typer.Option(4, help="Training batch size"),
    epochs: int = typer.Option(6, help="Number of training epochs"),
    learning_rate: float = typer.Option(1e-5, help="Learning rate for training"),
):
    logger.info("encoder train cli")
    
    dataset = load_dataset("s-nlp/PsiloQA")
    logger.info("loaded dataset")
    
    logger.info("starting training")
    run_encoder_training(
        model_name=model_name,
        output_dir=output_dir,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        dataset=dataset,
    )
    logger.info("finished training")

@app.command("encoder_eval")
def encoder_eval(
    model_path: str = typer.Argument(..., help="Path to the saved model"),
    language: str = typer.Option(None, help="If provided, evaluate only this language (e.g., 'cs', 'es')"),
    evaluation_type: str = typer.Option("token_level", help="Evaluation type (token_level, example_level, or char_level)"),
    batch_size: int = typer.Option(8, help="Evaluation batch size"),
):
    logger.info("encoder evaluation cli")
    
    dataset = load_dataset("s-nlp/PsiloQA", split="test")
    logger.info("loaded dataset")
    
    logger.info("starting evaluation")
    run_encoder_evaluation(
        model_path=model_path,
        language=language,
        evaluation_type=evaluation_type,
        batch_size=batch_size,
        dataset=dataset
    )
    logger.info("finished evaluation")


if __name__ == "__main__":
    app()
