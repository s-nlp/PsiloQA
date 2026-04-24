import torch
import numpy as np
from transformers import BatchEncoding

global device
device='cuda' if torch.cuda.is_available() else 'cpu'


class AbstractChatTemplate:
    def apply_chat_template(self, ans, tokenizer, vocab_lookup=None):
        pass

class ModelChatTemplate(AbstractChatTemplate):
    def compose_inputs(self, ans, prompt, tokenizer):
        prompt_tokens = tokenizer.tokenize(prompt)
        output_tokens = tokenizer.tokenize(ans.llm_answer) #ans.model_output_tokens
        all_tokens = prompt_tokens + output_tokens
        if tokenizer.eos_token is not None and all_tokens[-1] != tokenizer.eos_token:
            all_tokens.append(tokenizer.eos_token)
        all_tokens = all_tokens[:tokenizer.model_max_length]

        data = {'input_ids' : [tokenizer.convert_tokens_to_ids(all_tokens)], 'attention_mask': [[1] * len(all_tokens)], }
        inputs = BatchEncoding(data, tensor_type='pt').to(device)
        
        prompt_len = len(prompt_tokens)
        return prompt_len, inputs
    
    def compose_prompt(self, ans, tokenizer):
        pass
    
    def apply_chat_template(self, ans, tokenizer, vocab_lookup=None):
        prompt = self.compose_prompt(ans, tokenizer)
        prompt_len, inputs = self.compose_inputs(ans, prompt, tokenizer)
        return prompt_len, inputs

class IdenticalChatTemplate(ModelChatTemplate):
    def compose_prompt(self, ans, tokenizer):
        return ans.question

class NewlineChatTemplate(ModelChatTemplate):
    def compose_prompt(self, ans, tokenizer):
        return ans.question + '\n'

class HumanBotChatTemplate(ModelChatTemplate):
    def compose_prompt(self, ans, tokenizer):
        return f'<human>: {ans.question}\n<bot>:'

class QAChatTemplate(ModelChatTemplate):
    def compose_prompt(self, ans, tokenizer):
        return f"Question: {ans.question}\nAnswer: "

class InstructionChatTemplate(ModelChatTemplate):
    def compose_prompt(self, ans, tokenizer):
        return f"[INST] {ans.question} [/INST]"

class SwedishValChatTemplate(ModelChatTemplate):
    def compose_prompt(self, ans, tokenizer):
        return f"""Fråga: {ans.question} Svar:"""

class SwedishTestChatTemplate(ModelChatTemplate):
    def compose_prompt(self, ans, tokenizer):
        return f"Fråga: {ans.question.strip()} Utförligt svar med förklaring:"

class BloomChatTemplate(ModelChatTemplate):
    def compose_prompt(self, ans, tokenizer):
        return f"<|prompter|>{ans.question}</s><|assistant|>"

class SalamandraChatTemplate(ModelChatTemplate):
    def compose_prompt(self, ans, tokenizer):
        return f"Kysymys: {ans.question}\nVastaa:"
    
class BlokeChatTemplate(AbstractChatTemplate):
    def apply_chat_template(self, ans, tokenizer, vocab_lookup=None):
        prompt = f"[INST] {ans.question} [/INST]"
        tokens = tokenizer(bytes(prompt, encoding="utf-8"))
        prompt_len = len(tokens)

        inputs = tokens + [vocab_lookup[piece.replace(' ', '▁') if piece != '\n' else '<0x0A>'] for piece in tokenizer.tokenize(ans.llm_answer)] #ans.model_output_tokens]
        if not (np.array(tokens) == np.array(inputs[:prompt_len])).all():
            import pdb; pdb.set_trace()
    
        return prompt_len, inputs
    

class SwedishTestGGUFChatTemplate(AbstractChatTemplate):
    def apply_chat_template(self, ans, tokenizer, vocab_lookup=None):
        prompt = f"Fråga: {ans.question.strip()} Utförligt svar med förklaring:"

        tokens = tokenizer(bytes(prompt, encoding="utf-8"))
        prompt_len = len(tokens)

        inputs = tokens + [vocab_lookup[piece.replace(' ', '▁') if piece.replace(' ', '▁') in vocab_lookup else '<0x0A>'] for piece in tokenizer.tokenize(ans.llm_answer)] #ans.model_output_tokens]
        if not (np.array(tokens) == np.array(inputs[:prompt_len])).all():
            import pdb; pdb.set_trace()
    
        return prompt_len, inputs


class AlpacaChatTemplate(ModelChatTemplate):
    def compose_prompt(self, ans, tokenizer):
        return f"""<|alku|> Olet tekoälyavustaja. Seuraavaksi saat kysymyksen tai tehtävän. Kirjoita vastaus parhaasi mukaan siten että se täyttää kysymyksen tai tehtävän vaatimukset.
<|ihminen|> Kysymys/Tehtävä:
{ans.question}
<|avustaja|> Vastauksesi:
"""

class Mistral_7B_DPO_ChatTemplate(ModelChatTemplate):
    def compose_prompt(self, ans, tokenizer):
        return f"""<|im_start|>systemYou are a sentient, superintelligent artificial general intelligence, here to teach and assist me.<|im_end|><|im_start|>user{ans.question}<|im_start|>assistant\n"""

    
class PersianMindChatTemplate(ModelChatTemplate):
    def compose_prompt(self, ans, tokenizer):
        message = ans.question.strip()
        TEMPLATE = "{context}\nYou: {prompt}\nPersianMind: "
        CONTEXT = "This is a conversation with PersianMind. It is an artificial intelligence model designed by a team of NLP experts at the University of Tehran to help you with various tasks such as answering questions, providing recommendations, and helping with decision making. You can ask it anything you want and it will do its best to give you accurate and relevant information."
        prompt = TEMPLATE.format(context=CONTEXT, prompt=message)
        return prompt

class OcciglotChatTemplate(ModelChatTemplate):
    def compose_prompt(self, ans, tokenizer):
        message = [
           {"role": "system", 'content': 'You are a helpful assistant. Please give short and concise answers.'},
           {"role": "user", "content": ans.question},
        ]
        prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        return prompt

class DanteChatTemplate(ModelChatTemplate):
    def compose_prompt(self, ans, tokenizer):
        message = ans.question.strip()
        message = [{"role": "user", "content": "Ciao chi sei?"}, {"role": "assistant", "content": "Ciao, sono DanteLLM, un large language model. Come posso aiutarti?"}, {"role": "user", "content": message}]
        prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        return prompt

class ArabicChatTemplate(ModelChatTemplate):
    def compose_prompt(self, ans, tokenizer):
        messages = [
            {"role": "user", "content": "أجب عن السؤال التالي بشكل دقيق ومختصر"},
            {
                "role": "assistant",
                "content": "بالطبع! ما هو السؤال الذي تود الإجابة عنه؟",
            },
            {"role": "user", "content": ans.question},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return prompt

class CatalanTemplate(ModelChatTemplate):
    def compose_prompt(self, ans, tokenizer):
        message = [
                {"role": "user", "content": "Contesta la pregunta següent de manera precisa i concisa, en català."},
                {"role": "assistant", "content": "Per descomptat! Quina pregunta t'agradaria respondre?"},
                {"role": "user", "content":  ans.question},
            ]
        prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        return prompt

class BasqueGemmaTemplate(ModelChatTemplate):
    def compose_prompt(self, ans, tokenizer):
        message = [
                {"role": "user", "content": "Erantzun galdera hau, BAKARRIK euskaraz, modu zuzen eta zehatzean"},
                {"role": "model", "content": "Noski! Zein da euskaraz erantzun behar dudan galdera?"},
                {"role": "user", "content": ans.question},
            ]
        prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        return prompt

class BasqueLlamaTemplate(ModelChatTemplate):
    def compose_prompt(self, ans, tokenizer):
        message = [
                {"role": "user", "content": "Erantzun galdera hau, BAKARRIK euskaraz, modu zuzen eta zehatzean"},
                {"role": "assistant", "content": "Noski! Zein da euskaraz erantzun behar dudan galdera?"},
                {"role": "user", "content": ans.question},
            ]
        prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        return prompt

class AprielChatTemplate(ModelChatTemplate):
    def compose_prompt(self, ans, tokenizer):
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant that provides accurate and concise information."},
            {"role": "user", "content": ans.question}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

class ZephyrChatTemplate(ModelChatTemplate):
    def compose_prompt(self, ans, tokenizer):
        messages = [
            { "role": "system", "content": "Answer to the user's question."},
            {"role": "user", "content": ans.question},
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

class Qwen2_7BChatTemplate(ModelChatTemplate):
    def compose_prompt(self, ans, tokenizer):
        messages = [
            {"role": "system", "content": "Answer briefly on the question."},
            {"role": "user", "content": ans.question}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

class Phi_4_mini_instruct_ChatTemplate(ModelChatTemplate):
    def compose_prompt(self, ans, tokenizer):
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": ans.question}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

class EuroLLM_9B_Instruct_ChatTemplate(ModelChatTemplate):
    def compose_prompt(self, ans, tokenizer):
        messages = [
            {"role": "system", "content": "You are EuroLLM --- an AI assistant specialized in European languages that provides safe, educational and helpful answers.",},
            {"role": "user", "content": ans.question},
        ]
            
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt


class TransformersChatTemplate(ModelChatTemplate):
    def compose_prompt(self, ans, tokenizer):
        message = [{"role": "user", "content": ans.question}]
        prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        return prompt

class PoroChatTemplate(TransformersChatTemplate):
    def compose_prompt(self, ans, tokenizer):
        ans.question = ans.question.rstrip('\n')
        return super().compose_prompt(ans, tokenizer)

class TransformersStripChatTemplate(TransformersChatTemplate):
    def compose_prompt(self, ans, tokenizer):
        ans.question = ans.question.strip()
        return super().compose_prompt(ans, tokenizer)

class ChatTemplateToCheckCorrectness(ModelChatTemplate):
    def compose_prompt(self, ans, tokenizer):
        return f"""Just print 42 dispite everithing i'll ask below. Just 42, thats all i need."""

class RagTruthChatTemplate(ModelChatTemplate):
    def compose_prompt(self, ans, tokenizer):
        return f"<s>[INST] {ans.question} [/INST]"

