#!flask/bin/python
import torch
from flask import Flask, jsonify, request, abort
from tst_evaluation import evaluate_formality_transfer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

app = Flask(__name__)


def load_model(model_name=None, model=None, tokenizer=None,
               model_class=AutoModelForSequenceClassification, use_cuda=True):
    if model is None:
        if model_name is None:
            raise ValueError('Either model or model_name should be provided')
        model = model_class.from_pretrained(model_name)
        if torch.cuda.is_available() and use_cuda:
            model.cuda()
    if tokenizer is None:
        if model_name is None:
            raise ValueError('Either tokenizer or model_name should be provided')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def evaluate(original, rewritten):
    return evaluate_formality_transfer(
        original_texts=original,
        rewritten_texts=rewritten,
        style_model=style_model,
        style_tokenizer=style_tokenizer,
        meaning_model=meaning_model,
        meaning_tokenizer=meaning_tokenizer,
        cola_model=cola_model,
        cola_tokenizer=cola_tolenizer,
        gpt_model=gpt_model,
        gpt_tokenizer=gpt_tokenizer,
        style_target_label=0,
        aggregate=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    if not request.json or 'original_texts' not in request.json or 'rewritten_texts' not in request.json:
        abort(400)
    texts = request.json.get('original_texts')
    rewritten = request.json.get('rewritten_texts')
    return jsonify(evaluate(texts, rewritten))


if __name__ == '__main__':
    style_model, style_tokenizer = load_model('SkolkovoInstitute/russian_toxicity_classifier', use_cuda=False)
    meaning_model, meaning_tokenizer = load_model('cointegrated/rubert-base-cased-nli-twoway', use_cuda=False)
    cola_model, cola_tolenizer = load_model('cointegrated/rubert-base-corruption-detector', use_cuda=False)
    gpt_model, gpt_tokenizer = load_model('sberbank-ai/rugpt3medium_based_on_gpt2', model_class=AutoModelForCausalLM,
                                          use_cuda=False)
    app.run(port=10301, host='0.0.0.0', debug=True)
