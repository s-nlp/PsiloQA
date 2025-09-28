import torch

LONG_ANSWER_CONSTRAINT = "You must generate long answers."
SHORT_ANSWER_CONSTRAINT = "You must generate short answers."
WIKI_API = "https://{lang}.wikipedia.org/w/api.php"
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
