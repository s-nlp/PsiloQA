import numpy as np
from sklearn.metrics import average_precision_score

import torch
from tqdm import tqdm

import re
from transformers import AutoTokenizer

tokenizers_cache = {}

def row_to_psiloqa_record(row):
    question = row['question']
    passage = row['wiki_passage']
    llm_answer = row['llm_answer']
    annotated = row.get('annotated_span') or ""
    split = row.get('split', 'train')
    language = row.get('language', row.get('lang'))

    # 1) prompt из заданного темплейта
    prompt = (
        "Briefly answer the following question:\n"
        f"{question}\n"
        "Bear in mind that your response should be strictly based on the following 1 passages:\n"
        f"{passage}\n"
        "In case the passages do not contain the necessary information to answer the question, please reply with: "
        "\"Unable to answer based on given passages.\"\n"
        "output:"
    )

    hal_chunks = [m.group(1) for m in re.finditer(r"\[HAL\](.*?)\[/HAL\]", annotated, flags=re.DOTALL)]
    labels = []
    cursor = 0
    for chunk in hal_chunks:
        if not chunk:
            continue
        start = llm_answer.find(chunk, cursor)
        if start == -1:
            start = llm_answer.find(chunk)
            if start == -1:
                continue
        end = start + len(chunk)
        labels.append({"start": start, "end": end, "label": "HAL"})
        cursor = end

    return {
        "prompt": prompt,
        "answer": llm_answer,
        "labels": labels,
        "split": split,
        "task_type": "QA",
        "dataset": "psiloqa",
        "language": language,
    }
    
def tokens_to_char_mask(text, token_values, model_name):
    tokenizer = tokenizers_cache.get(model_name, None)
    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizers_cache[model_name] = tokenizer
    
    # tokenize with offsets
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    
    assert len(offsets) == len(token_values) - 1, "Mismatch: token_values must match token count"
    
    mask = np.zeros(len(text))

    for (start, end), val in zip(offsets, token_values):
        mask[start: end] = val
    
    return mask


def evaluate_uncertainty(df_val, df_test):
    
    uq_methods = ["MSP", "CCP", "Focus"]
    scores_val = {lang: {uq: [] for uq in uq_methods} for lang in df_test.lang.unique()}

    for lang in tqdm(df_val.lang.unique()):
        for i, row in df_val.iterrows():
            if row.lang != lang:
                continue        
            for uq_method in uq_methods:
                token_uq_scores = row.get(uq_method)[0]
                if isinstance(token_uq_scores, float):
                    token_uq_scores = np.array(row.get("MSP")[0])
                    
                scores = tokens_to_char_mask(row.llm_answer, token_uq_scores, row.llm_checkpoint)
                scores_val[lang][uq_method].extend(scores.tolist())
                
                
    thr_val = {lang: {uq: [] for uq in uq_methods} for lang in df_val.lang.unique()}
    for lang in df_val.lang.unique():            
        thr_val_lang = {uq: [] for uq in uq_methods}
        bounds = {
            uq: {"min": np.min(scores_val[lang][uq]), "max":np.max(scores_val[lang][uq])} for uq in uq_methods
        }
        
        for i, row in tqdm(df_val.iterrows(), total=df_val.shape[0]):
            if row.lang != lang:
                continue
            span_labels = row_to_psiloqa_record(row)["labels"]
            binary_labels = np.zeros(len(row.llm_answer))
            for span in span_labels:
                binary_labels[span["start"] : span["end"]] = 1
            
            for uq in uq_methods:
                step = (bounds[uq]["max"] - bounds[uq]["min"]) / 2000
                token_uq_scores = row.get(uq)[0]
                if isinstance(token_uq_scores, float):
                    token_uq_scores = np.array(row.get("MSP")[0])
                    
                scores = tokens_to_char_mask(row.llm_answer, token_uq_scores, row.llm_checkpoint)
                ids_labels = set(torch.where(torch.tensor(binary_labels))[0].tolist())
                
                thrs_row_uq = []
                for thr in np.arange(bounds[uq]["min"], bounds[uq]["max"], step):
                    preds = (scores >= thr).astype(int)
                    ids_pred = set(torch.where(torch.tensor(preds))[0].tolist())
                    
                    if not ids_pred and not ids_labels:
                        iou = 1.
                    else:
                        iou = len(ids_labels & ids_pred) / len(ids_labels | ids_pred)
                        
                    thrs_row_uq.append(iou)    
                thr_val_lang[uq].append(thrs_row_uq)

        for uq in uq_methods:
            step = (bounds[uq]["max"] - bounds[uq]["min"]) / 2000
            thrs = np.arange(bounds[uq]["min"], bounds[uq]["max"], step)
            thr_val[lang][uq] = thrs[np.array(thr_val_lang[uq]).mean(0).argmax()]
        
    test_iou = {lang: {uq: [] for uq in uq_methods} for lang in df_test.lang.unique()}
    test_ap = {lang: {uq: [] for uq in uq_methods} for lang in df_test.lang.unique()}

    for i, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):
        
        span_labels = row_to_psiloqa_record(row)["labels"]
        binary_labels = np.zeros(len(row.llm_answer))
        for span in span_labels:
            binary_labels[span["start"] : span["end"]] = 1
        
        for uq in uq_methods:
            token_uq_scores = row.get(uq)[0]
            if isinstance(token_uq_scores, float):
                token_uq_scores = np.array(row.get("MSP")[0])
                
            scores = tokens_to_char_mask(row.llm_answer, token_uq_scores, row.llm_checkpoint)
            ids_labels = set(torch.where(torch.tensor(binary_labels))[0].tolist())
            
            preds = (scores >= thr_val[row.lang][uq]).astype(int)
            ids_pred = set(torch.where(torch.tensor(preds))[0].tolist())
            
            if not ids_pred and not ids_labels:
                iou = 1.
            else:
                iou = len(ids_labels & ids_pred) / len(ids_labels | ids_pred)

            test_iou[row.lang][uq].append(iou)
            ap = average_precision_score(binary_labels, scores)
            test_ap[row.lang][uq].append(ap)

    for lang in df_test.lang.unique():
        print(f"##### {lang} #####")
        print("IOU:")
        for uq in uq_methods:
            print(f"{uq}: {(np.array(test_iou[lang][uq]).mean()*100):.2f}")
        print()
        print("AP:")
        for uq in uq_methods:
            print(f"{uq}: {(np.array(test_ap[lang][uq]).mean()*100):.2f}")
