import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding
from huggingface_hub import hf_hub_download
import torch

from .utils.consts import *

from .estimators.max_probability import *
from .estimators.focus import *
from .estimators.claim_conditioned_probability import *
from .estimators.greedy_alternatives_nli import GreedyAlternativesNLICalculator, Deberta

TOP_K = 50
device='cuda' if torch.cuda.is_available() else 'cpu'

def calc_top_k_values(arr, k):    
    arr = np.array(arr)
    # Get the indices that would sort the array in descending order
    sorted_indices = np.argsort(arr)[::-1][:k]
    
    # Get the top k values and their corresponding indices
    top_k_values = arr[sorted_indices].tolist()
    top_k_indices = sorted_indices.tolist()
    
    return top_k_values, top_k_indices

def get_logits(df):
    # model_name = set(df.model_id).pop()
    # Adapt to new column names: model_id -> llm_checkpoint, model_output_text -> llm_answer
    model_name = set(df.llm_checkpoint).pop()
    lang = set(df.lang).pop().lower()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto", attn_implementation="eager")
    
    all_logits = {} 
    all_response_token_ids = {}
    all_response_full_logits = {}
    all_response_tokens_restored = {}
    all_response_logits2token_id = {}
    all_response_logits2tokens = {}
    
    all_msp = {} 
    all_ccp = {} 
    all_focus = {} 
    
    nli_model = Deberta(device="cuda")
    nli_model.setup()
    calc_nli = GreedyAlternativesNLICalculator(nli_model=nli_model)    
    
    for idx, ans in tqdm(df.iterrows(), desc=model_name, total=len(df)):
        # Use the new column names in the chat template
        prompt_len, inputs = LANG_MODEL2CHAT_TEMPLATE[lang][model_name].apply_chat_template(ans, tokenizer)
        with torch.inference_mode():  
            outputs = model(**inputs, output_attentions=True)
                     
        outputs.scores = (outputs.logits[0, prompt_len-1:-1, :],)   
        outputs.sequences = inputs["input_ids"]
                
        methods = [SequenceProbability(), 
                   ClaimConditionedProbability(), 
                   Focus(gamma=0.9, p=0.01, model_name=model_name, path=f"../focus/{model_name}/token_idf.pkl", 
                         idf_dataset="togethercomputer/RedPajama-Data-1T-Sample", idf_seed=42, spacy_path="en_core_web_sm", trust_remote_code=True,
                         idf_dataset_size=100000),
                   ]
            
        cut_logits = []
        cut_sequences = []
        cut_log_probs = []
        cut_alternatives = []
        lls = []
        attention_all = []

        eos_token_id = -1
        model_inputs = inputs["input_ids"][:, :prompt_len]
        n_alternatives = 10
        
        all_logits_ = torch.stack(outputs.scores, dim=0)
        for i in range(len(model_inputs)):
            seq = outputs.sequences[i, model_inputs.shape[1]:].cpu()
            
            length = len(seq)
            for j in range(len(seq)):
                if seq[j] == eos_token_id:
                    length = j + 1
                    break

            tokens = seq[:length].tolist()
            cut_sequences.append(tokens)

            logits = all_logits_[i, :length, :].cpu()
            cut_logits.append(logits.numpy())

            log_probs = logits.log_softmax(-1)
            cut_log_probs.append(log_probs.numpy())
            lls.append([log_probs[j, tokens[j]] for j in range(len(log_probs))])

            cut_alternatives.append([[] for _ in range(length)])
            for j in range(length):
                lt = logits[j, :].numpy()
                best_tokens = np.argpartition(lt, -n_alternatives)
                ln = len(best_tokens)
                best_tokens = best_tokens[ln - n_alternatives : ln]
                for t in best_tokens:
                    cut_alternatives[-1][j].append((t.item(), lt[t].item()))

                cut_alternatives[-1][j].sort(
                    key=lambda x: x[0] == cut_sequences[-1][j],
                    reverse=True,
                )

        result_dict = {
            "greedy_log_probs": cut_log_probs,
            "greedy_logits": cut_logits,
            "greedy_tokens": cut_sequences,
            "greedy_log_likelihoods": lls,
            "greedy_tokens_alternatives": cut_alternatives,
        }
        
        result_dict["tokenizer"] = tokenizer
        
        attentions = tuple(attention.to("cpu") for attention in outputs.attentions)
        attentions = torch.cat(attentions).float().numpy()
        attentions = attentions[:, :, prompt_len-1:-1, prompt_len:]
        attention_all.append(attentions)
                    
        result_dict["attention_all"] = attention_all
        result_dict.update(calc_nli(result_dict, texts=None, model=model))   
        # Use llm_answer instead of model_output_text
        result_dict["greedy_texts"] = [ans.llm_answer]
            
        all_msp[idx] = methods[0](result_dict)
        all_ccp[idx] = methods[1](result_dict)
        all_focus[idx] = methods[2](result_dict)
                    
        response_token_ids = inputs['input_ids'].to("cpu").tolist()[0]
        output_logits = outputs.logits.squeeze().to('cpu')
        response_logits = [None] + [logit.tolist()[response_token_ids[idx+1]] for idx,logit in enumerate(output_logits[:-1])]           
        response_full_logits, response_logits2token_id = zip(*[(None, None)] + [calc_top_k_values(logit.tolist(), TOP_K) for idx,logit in enumerate(output_logits[:-1])])

        response_token_ids = response_token_ids[prompt_len:]
        response_logits = response_logits[prompt_len:]
        response_full_logits = response_full_logits[prompt_len:]
        response_logits2token_id = response_logits2token_id[prompt_len:]
        
        all_logits[idx] = response_logits
        all_response_full_logits[idx] = response_full_logits
        all_response_token_ids[idx] = response_token_ids
        all_response_logits2token_id[idx] = response_logits2token_id
        all_response_logits2tokens[idx] = [[tokenizer.convert_ids_to_tokens(id) for id in top_k_indices] for top_k_indices in response_logits2token_id]
        all_response_tokens_restored[idx] = [tokenizer.convert_ids_to_tokens(id) for id in response_token_ids]
        

    df["response_logits"] = all_logits
    df["response_token_ids"] = all_response_token_ids
    df["response_full_logits"] = all_response_full_logits
    df["response_logits2token_id"] = all_response_logits2token_id
    df["response_logits2tokens"] = all_response_logits2tokens
    df["response_tokens_restored"] = all_response_tokens_restored
    
    df["MSP"] = all_msp
    df["CCP"] = all_ccp
    df["Focus"] = all_focus

    # clean cuda
    model = None
    tokenizer = None
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return df

def run_uncertainty_evaluation(dataset):
    output_df = pd.DataFrame()

    # Use new column names: llm_checkpoint for model, lang, id, etc.
    languages = set(dataset.lang.dropna())
    dfs = []
    for lang in languages:       
        model_names = set(dataset.llm_checkpoint.dropna()[dataset.lang == lang])
        for mname in model_names:
            df = dataset[(dataset.llm_checkpoint == mname) & (dataset.lang == lang)].reset_index()
            print("################# ", lang, " -> ", mname, " examples: ", len(df), " ###########################")

            if mname in LANG_MODEL2CHAT_TEMPLATE[lang.lower()]:
                df = get_logits(df)
            else:
                print(f"Chat template for model {mname} not found")
                continue
            dfs.append(df)

            new_df = pd.concat(dfs)
            output_df = pd.concat([output_df, new_df])
    return output_df