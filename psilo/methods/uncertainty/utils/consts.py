from .model_templates import *

LANG_MODEL2CHAT_TEMPLATE = {
 'en': {
   'togethercomputer/Pythia-Chat-Base-7B': HumanBotChatTemplate(),
   "tiiuae/falcon-7b-instruct": QAChatTemplate(),
   "mistralai/Mistral-7B-Instruct-v0.1": RagTruthChatTemplate(),
   "meta-llama/Llama-2-7b-chat-hf": RagTruthChatTemplate(),
   "meta-llama/Llama-2-13b-chat-hf": RagTruthChatTemplate(),
   "HuggingFaceH4/zephyr-7b-beta": ZephyrChatTemplate(),
   "TinyLlama/TinyLlama-1.1B-Chat-v1.0": TransformersChatTemplate(),
   "HuggingFaceTB/SmolLM2-135M-Instruct": TransformersChatTemplate(),
   "HuggingFaceTB/SmolLM2-360M-Instruct": TransformersChatTemplate(),
   "HuggingFaceTB/SmolLM2-1.7B-Instruct": TransformersChatTemplate(),
   "ServiceNow-AI/Apriel-5B-Instruct": AprielChatTemplate(),
   "togethercomputer/Pythia-Chat-Base-7B-v0.16": HumanBotChatTemplate(),
   "microsoft/Phi-4-mini-instruct": Phi_4_mini_instruct_ChatTemplate(),
   "NousResearch/Nous-Hermes-2-Mistral-7B-DPO": Mistral_7B_DPO_ChatTemplate(),
   "utter-project/EuroLLM-9B-Instruct": EuroLLM_9B_Instruct_ChatTemplate(),
 },
 'hi': {
    'nickmalhotra/ProjectIndus': TransformersChatTemplate(),
    'sarvamai/sarvam-1': TransformersChatTemplate(),
    'google/gemma-7b-it': TransformersChatTemplate(),
 },
 'sv': {
    'utter-project/EuroLLM-9B-Instruct': TransformersChatTemplate(),
 },
 'it': {
    'sapienzanlp/modello-italia-9b': TransformersChatTemplate(),
 },
 'de': {
    'malteos/bloom-6b4-clp-german-oasst-v0.1': NewlineChatTemplate(),
 },
 'eu': {
    'google/gemma-7b-it': TransformersChatTemplate(),
 },
 'zh': {
    'Qwen/Qwen2.5-3B-Instruct': TransformersChatTemplate(),
    "Qwen/Qwen2-7B-Instruct": Qwen2_7BChatTemplate(),
    "ikala/bloom-zh-3b-chat": BloomChatTemplate(),
 },
 'fr': {
    'croissantllm/CroissantLLMChat-v0.1': TransformersChatTemplate(),
 },
 'ar': {
    'SeaLLMs/SeaLLM-7B-v2.5': TransformersChatTemplate(),
 },
 'fa': {
    'Qwen/Qwen2-7B-Instruct': TransformersChatTemplate(),
 },
 'fi': {
    'Finnish-NLP/llama-7b-finnish-instruct-v0.2': AlpacaChatTemplate(),
    'BSC-LT/salamandra-7b': SalamandraChatTemplate(),
 },
 'cs': {
    'mistralai/Mistral-7B-Instruct-v0.3': TransformersChatTemplate(),
 },
 'es': {
    'Iker/Llama-3-Instruct-Neurona-8b-v2': TransformersChatTemplate()
 },
 'ca': {
    "occiglot/occiglot-7b-es-en-instruct": CatalanTemplate(),
 },
}


LARGE_MODELS_LIST = [
    'mistralai/Mistral-Nemo-Instruct-2407',
    'bofenghuang/vigogne-2-13b-chat',
    'LumiOpen/Poro-34B-chat',
    'Qwen/Qwen1.5-14B-Chat',
    'LumiOpen/Viking-33B',
    'baichuan-inc/Baichuan2-13B-Chat',
    'CohereForAI/aya-23-35B',
]

RAGTRUTH_MODELS = [
   "mistralai/Mistral-7B-Instruct-v0.1",
   "meta-llama/Llama-2-7b-chat-hf",
   # "meta-llama/Llama-2-13b-chat-hf",
]

GGUF2FILENAME = {
    "TheBloke/Mistral-7B-Instruct-v0.2-GGUF": "mistral-7b-instruct-v0.2.Q6_K.gguf",
    "TheBloke/SauerkrautLM-7B-v1-GGUF": "sauerkrautlm-7b-v1.Q4_K_M.gguf",
    "AI-Sweden-Models/gpt-sw3-6.7b-v2-instruct-gguf": "gpt-sw3-6.7b-v2-instruct-Q4_K_M.gguf"
}

GGUF_MODELS_LIST = list(GGUF2FILENAME.keys())
