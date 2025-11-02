import typer
from loguru import logger
from .uncertainty.recompute_logits_with_uq import run_uncertainty_evaluation
from .uncertainty.evaluation import evaluate_uncertainty
from datasets import load_dataset

app = typer.Typer(help="PsiloQA Methods Pipeline")

@app.command("uncertainty")
def uncertainty():
    logger.info("uncertainty evaluation cli")
    
    dataset_val = load_dataset("s-nlp/PsiloQA", split="validation").to_pandas()
    dataset_test = load_dataset("s-nlp/PsiloQA", split="test").to_pandas()
    
    logger.info("loaded datasets")
    
    logger.info("starting validation evaluation")
    result_val = run_uncertainty_evaluation(dataset=dataset_val)
    logger.info("finished validation evaluation")
    
    logger.info("starting test evaluation")
    result_test = run_uncertainty_evaluation(dataset=dataset_test)
    logger.info("finished test evaluation")
    
    logger.info("starting results aggregation")
    evaluate_uncertainty(result_val, result_test)
    logger.info("finished evaluation")

@app.command("encoder_train")
def encoder_train():
    logger.info("encoder train cli")

@app.command("encoder_eval")
def encoder_eval():
    logger.info("encoder evaluation cli")
    
if __name__ == "__main__":
    app()