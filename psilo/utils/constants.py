import uuid

import torch

LONG_ANSWER_CONSTRAINT = "You must generate long answers."
SHORT_ANSWER_CONSTRAINT = "You must generate short answers."
WIKI_API = "https://{lang}.wikipedia.org/w/api.php"
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NAMESPACE = uuid.UUID("6e8d8a3b-0d7a-4f59-8f4d-3e2c1a9c5e01")

# fmt: off
AVAILABLE_LANGUAGES = {'ru', 'ar', 'ca', 'cs', 'de', 'en', 'es', 'eu', 'fa', 'fi', 'fr', 'hi', 'it', 'sv', 'zh'}
# fmt: on
