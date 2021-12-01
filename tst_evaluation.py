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


def get_text_perplexity(text, model, tokenizer, eos='\n', bos='\n'):
    encodings = tokenizer(eos + text + bos, return_tensors='pt', truncation=True)
    input_ids = encodings.input_ids.to(model.device)
    n_tokens = max(0, len(input_ids[0]) - 1)
    if n_tokens > 0:
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
    else:
        loss = 0
    return loss, n_tokens


def get_corpus_perplexity(model, tokenizer, texts, unit='token', verbose=True, **kwargs):
    loss = []
    n_tokens = []
    pb = tqdm(texts) if verbose else  texts
    for text in pb:
        ll, w = get_text_perplexity(text, model, tokenizer, **kwargs)
        loss.append(ll)
        n_tokens.append(w)
    loss = np.array(loss)
    n_tokens = np.array(n_tokens)
    if unit == 'token': 
        return loss
    elif unit == 'text':
        return loss * n_tokens
    elif unit == 'char':
        return loss * n_tokens / [len(t) for t in texts]
    else:
        raise ValueError('unit should be one of ["token", "text", "char"]')


def evaluate_perplexity(
    model,
    tokenizer,
    texts,
    original_texts=None,
    unit='token',
    verbose=False,
    comparison='relative',
):
    scores = -get_corpus_perplexity(model, tokenizer, texts, unit=unit, verbose=verbose)
    if comparison and original_texts is not None:
        original_scores = -get_corpus_perplexity(original_texts, model, tokenizer, unit=unit, verbose=verbose)
        scores = scores - original_scores
        if 'cap' in comparison:
            scores = np.minimum(0, scores)
        elif 'abs' in comparison:
            scores = -np.abs(scores)
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
    gpt_model,
    gpt_tokenizer,
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
    perplexity = evaluate_perplexity(
        gpt_model,
        gpt_tokenizer,
        rewritten_texts,
        original_texts=original_texts, verbose=verbose, comparison='relative',
    )
    fluency = cola ** cola_weight * sigmoid(perplexity) ** (1-cola_weight)
    joint = accuracy * similarity * fluency
    if verbose:
        print(f'Style accuracy:       {np.mean(accuracy)}')
        print(f'Meaning preservation: {np.mean(similarity)}')
        print(f'CoLA fluency:         {np.mean(cola)}')
        print(f'GPT fluency:          {np.mean(perplexity)}')
        print(f'Joint fluency:        {np.mean(fluency)}')
        print(f'Joint score:          {np.mean(joint)}')
    result = dict(
        accuracy=accuracy,
        similarity=similarity,
        cola=cola,
        perplexity=perplexity,
        fluency=fluency,
        joint=joint
    )
    if aggregate:
        return {k: np.mean(v) for k, v in result.items()}
    return result
