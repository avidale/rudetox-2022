import torch
import numpy as np
from tqdm.auto import tqdm, trange


def prepare_target_label(model, target_label):
    if target_label in model.config.id2label:
        pass
    elif target_label in model.config.label2id:
        target_label = model.config.label2id.get(target_label)
    elif target_label.isnumeric() and int(target_label) in model.config.id2label:
        target_label = int(target_label)
    else:
        raise ValueError(f'target_label "{target_label}" is not in model labels or ids: {model.config.id2label}.')
    return target_label


def classify_texts(model, tokenizer, texts, second_texts=None, target_label=None, batch_size=32, verbose=False):
    target_label = prepare_target_label(model, target_label)
    res = []
    if verbose:
        tq = trange
    else:
        tq = range
    for i in tq(0, len(texts), batch_size):
        inputs = [texts[i:i+batch_size]]
        if second_texts is not None:
            inputs.append(second_texts[i:i+batch_size])
        inputs = tokenizer(*inputs, return_tensors='pt', padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            preds = torch.softmax(model(**inputs).logits, -1)[:, target_label].cpu().numpy()
        res.append(preds)
    return np.concatenate(res)


def evaluate_formality(
    model,
    tokenizer,
    texts,
    target_label=1,  # 1 is formal, 0 is informal
    batch_size=32, 
    verbose=False
):
    target_label = prepare_target_label(model, target_label)
    scores = classify_texts(
        model,
        tokenizer,
        texts, 
        batch_size=batch_size, verbose=verbose, target_label=target_label
    )
    return scores


def evaluate_meaning(
    model,
    tokenizer,
    original_texts, 
    rewritten_texts,
    target_label='entailment', 
    bidirectional=True, 
    batch_size=32, 
    verbose=False, 
    aggregation='prod'
):
    target_label = prepare_target_label(model, target_label)
    scores = classify_texts(
        model, tokenizer,
        original_texts, rewritten_texts, 
        batch_size=batch_size, verbose=verbose, target_label=target_label
    )
    if bidirectional:
        reverse_scores = classify_texts(
            model, tokenizer,
            rewritten_texts, original_texts,
            batch_size=batch_size, verbose=verbose, target_label=target_label
        )
        if aggregation == 'prod':
            scores = reverse_scores * scores
        elif aggregation == 'mean':
            scores = (reverse_scores + scores) / 2
        elif aggregation == 'f1':
            scores = 2 * reverse_scores * scores / (reverse_scores + scores)
        else:
            raise ValueError('aggregation should be one of "mean", "prod", "f1"')
    return scores


def evaluate_cola(
    model,
    tokenizer,
    texts,
    target_label=1,
    batch_size=32, 
    verbose=False
):
    target_label = prepare_target_label(model, target_label)
    scores = classify_texts(
        model, tokenizer,
        texts, 
        batch_size=batch_size, verbose=verbose, target_label=target_label
    )
    return scores


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def evaluate_formality_transfer(
    original_texts,
    rewritten_texts,
    style_model,
    style_tokenizer,
    meaning_model,
    meaning_tokenizer,
    cola_model,
    cola_tokenizer,
    style_target_label=1,
    meaning_target_label='entailment',
    cola_weight=0.5,
    batch_size=32,
    verbose=True,
    aggregate=False,
):
    if verbose: print('Style evaluation')
    accuracy = evaluate_formality(
        style_model,
        style_tokenizer,
        rewritten_texts,
        target_label=style_target_label, batch_size=batch_size, verbose=verbose
    )
    if verbose: print('Meaning evaluation')
    similarity = evaluate_meaning(
        meaning_model,
        meaning_tokenizer,
        original_texts, 
        rewritten_texts,
        target_label=meaning_target_label, batch_size=batch_size, verbose=verbose
    )
    if verbose: print('Fluency evaluation')
    cola = evaluate_cola(
        cola_model,
        cola_tokenizer,
        rewritten_texts, 
        batch_size=batch_size, verbose=verbose,
    )

    fluency = cola
    joint = accuracy * similarity * fluency
    if verbose:
        print(f'Style accuracy:       {np.mean(accuracy)}')
        print(f'Meaning preservation: {np.mean(similarity)}')
        print(f'Joint fluency:        {np.mean(fluency)}')
        print(f'Joint score:          {np.mean(joint)}')
    result = dict(
        accuracy=accuracy,
        similarity=similarity,
        fluency=fluency,
        joint=joint
    )
    if aggregate:
        return {k: float(np.mean(v)) for k, v in result.items()}
    return result
