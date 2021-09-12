#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#Benny Longwill
#12/20/20
#For use as part of BERT GAN thesis project
#Methods used to carry out evaluation on generator objects (i.e., BERT or GPT)


###### Acknowledgments ####################################################
#In this file I am only responsible for formulating the wrapper methods that facilitate evaluation for the current study.
#-Evaluation procedure and code was copied directly for reproducibility of results from:
#  @inproceedings{wang2019bert,
    #title = "{BERT} has a Mouth, and It Must Speak: {BERT} as a {M}arkov Random Field Language Model",
    #author = "Wang, Alex  and  Cho, Kyunghyun", month = jun, year = "2019"
    #https://github.com/nyu-dl/bert-gen/blob/master/bert-babble.ipynb
############################################################################


#Dependencies
import torch
from collections import Counter
from nltk.util import ngrams
import numpy as np
import time
from nltk.translate import bleu_score as bleu
import math
from utility_models import tokenizer, perplexity_model
from configuration import settings
from gan_model import GAN
from model_pytorch import LMModel, load_openai_pretrained_model, DEFAULT_CONFIG
from text_utils import TextEncoder
import pickle

# Set the seed value all over the place to make this reproducible.
seed_val = settings.get_random_state()
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

def evaluate_bert(generator, output_file_path:str):
    settings.write_debug("Entering Bleu EVAL")
    with torch.no_grad():

        ### Using 1 eval topk here breaks computation because it is essentially argmax and non-differentiable
        # Choose the prefix context
        generated_input_ids = generator.generate(settings.get_num_eval_samples(),
                                                 seed_text=settings.get_eval_seed_text().split(),
                                                 batch_size=settings.get_eval_batch_size(),
                                                 max_len=settings.get_sample_size(),
                                                 generation_mode=settings.get_eval_gen_mode_key(),
                                                 sample=settings.get_eval_sample(),
                                                 top_k=settings.get_eval_top_k(),
                                                 temperature=settings.get_eval_temp(),
                                                 burnin=settings.get_eval_burnin(),
                                                 max_iter=settings.get_eval_max_iter())

        bert_sents = [generator.detokenize(tokenizer.convert_ids_to_tokens(sent.tolist())).split() for sent in generated_input_ids]
        with open(output_file_path, 'a+') as f:
            [f.write('[%d/%d]\t' % (index, len(bert_sents)) + " ".join(sent) + "\n") for index, sent in enumerate(bert_sents)]

        avg_p = np.average([(perplexity_model.score(" ".join(sent))['positional_scores'].mean().neg().exp()).item() for sent in bert_sents])

        settings.write_result("BERT Perplexity: %.2f" % avg_p)

        settings.write_result("BERT self-BLEU: %.2f" % (100 * self_bleu(bert_sents)))

        max_n = settings.get_bleu_max_n()

        with open(settings.get_proc_wiki_path(), 'rb') as proc_wiki_f, open(settings.get_proc_tbc_path(), 'rb') as proc_tbc_f:

            wiki_data = pickle.load(proc_wiki_f)
            tbc_data = pickle.load(proc_tbc_f)

            settings.write_result("BERT-TBC BLEU: %.2f" % (100 * corpus_bleu(bert_sents, tbc_data)))
            settings.write_result("BERT-Wiki103 BLEU: %.2f" % (100 * corpus_bleu(bert_sents, wiki_data)))
            settings.write_result("BERT-{TBC + Wiki103} BLEU: %.2f" % (100 * corpus_bleu(bert_sents, tbc_data[:2500] + wiki_data[:2500])))

            pct_uniques = ref_unique_ngrams(bert_sents, wiki_data, max_n)
            for i in range(1, max_n + 1):
                settings.write_result("BERT unique %d-grams relative to Wiki: %.2f" % (i, 100 * pct_uniques[i]))
            pct_uniques = ref_unique_ngrams(bert_sents, tbc_data, max_n)
            for i in range(1, max_n + 1):
                settings.write_result("BERT unique %d-grams relative to TBC: %.2f" % (i, 100 * pct_uniques[i]))
            pct_uniques = self_unique_ngrams(bert_sents, max_n)
            for i in range(1, max_n + 1):
                settings.write_result("BERT unique %d-grams relative to self: %.2f" % (i, 100 * pct_uniques[i]))

        settings.write_result("")


def evaluate_gpt(output_file_path:str):

        settings.write_debug("Loading and Generating GPT model")

        gpt_model, gpt_text_encoder = load_openai_gpt(n_special=1)

        openai_sents = generate_openai(gpt_model,
                                       gpt_text_encoder,
                                       seed_text="",
                                       n_samples=settings.get_num_eval_samples(),
                                       batch_size=settings.get_eval_batch_size(),
                                       gen_len=settings.get_sample_size(),
                                       topk=settings.get_eval_top_k(),
                                       temperature=settings.get_eval_temp(),
                                       sample=settings.get_eval_sample(),
                                       n_special=1,
                                       print_every=1)


        with open(output_file_path, 'a+') as f:
            [f.write('[%d/%d]\t' % (index, len(openai_sents)) + " ".join(sent) + "\n") for index, sent in enumerate(openai_sents)]


        max_n = settings.get_bleu_max_n()

        openai_avg_p = np.average([(perplexity_model.score(" ".join(sent))['positional_scores'].mean().neg().exp()).item() for sent in openai_sents])
        settings.write_result("GPT Perplexity: %.2f" % openai_avg_p)

        with open(settings.get_proc_wiki_path(), 'rb') as proc_wiki_f, open(settings.get_proc_tbc_path(), 'rb') as proc_tbc_f:

            wiki_data = pickle.load(proc_wiki_f)
            tbc_data = pickle.load(proc_tbc_f)

            settings.write_result("OpenAI self-BLEU: %.2f" % (100 * self_bleu(openai_sents)))
            settings.write_result("GPT-TBC BLEU: %.2f" % (100 * corpus_bleu(openai_sents, tbc_data)))
            settings.write_result("GPT-Wiki103 BLEU: %.2f" % (100 * corpus_bleu(openai_sents, wiki_data)))
            settings.write_result("GPT-{TBC + Wiki103} BLEU: %.2f" % (100 * corpus_bleu(openai_sents, tbc_data[:2500] + wiki_data[:2500])))

        pct_uniques = ref_unique_ngrams(openai_sents, wiki_data, max_n)
        for i in range(1, max_n + 1):
            settings.write_result("GPT unique %d-grams relative to Wiki: %.2f" % (i, 100 * pct_uniques[i]))
        pct_uniques = ref_unique_ngrams(openai_sents, tbc_data, max_n)
        for i in range(1, max_n + 1):
            settings.write_result("GPT unique %d-grams relative to TBC: %.2f" % (i, 100 * pct_uniques[i]))
        pct_uniques = self_unique_ngrams(openai_sents, max_n)
        for i in range(1, max_n + 1):
            settings.write_result("GPT unique %d-grams relative to self: %.2f" % (i, 100 * pct_uniques[i]))

        settings.write_result("")

def load_openai_gpt(n_special=1, n_ctx=512):
    text_encoder = TextEncoder("pytorch-openai-transformer-lm/model/encoder_bpe_40000.json",
                               "pytorch-openai-transformer-lm/model/vocab_40000.bpe")
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)
    vocab = n_vocab + n_special + n_ctx

    args = DEFAULT_CONFIG
    lm_model = LMModel(args, vocab, n_ctx, return_probs=True)
    load_openai_pretrained_model(lm_model.transformer, n_ctx=n_ctx, n_special=n_special,
                                 path="pytorch-openai-transformer-lm/model/",
                                 path_names="pytorch-openai-transformer-lm/")
    # lm_model.to(device)
    lm_model.return_probs = False
    lm_model.eval()
    return lm_model, text_encoder


def make_batch(X, n_vocab, n_special, batch_size):
    X = np.array(X)
    assert X.ndim in [1, 2]
    if X.ndim == 1:
        X = np.expand_dims(X, axis=0)
    pos_enc = np.arange(n_vocab + n_special, n_vocab + n_special + X.shape[-1])
    pos_enc = np.tile(pos_enc, (batch_size, pos_enc.shape[-1]))  # np.expand_dims(pos_enc, axis=0)
    batch = np.stack([X, pos_enc], axis=-1)
    batch = torch.tensor(batch, dtype=torch.long)  # .to(device)
    return batch


def append_batch(X, next_idx):
    next_pos = X[:, -1:, 1] + 1
    next_x = torch.cat((next_idx, next_pos), -1).unsqueeze(1)
    return torch.cat((X, next_x), 1)


def _generate_sentence_openai(model, text_encoder, seed_text, batch_size=10, gen_len=20,
                              topk=100, sample=True, n_special=0):
    n_vocab = len(text_encoder.encoder)
    # X = np.random.randint(n_vocab, size=(batch_size, 1)).tolist()
    # sents = [[text_encoder.decoder[X[i][0]]].replace('</w>', '') for i in range(batch_size)]
    X = [[n_vocab - 1] for _ in range(batch_size)]
    sents = [[] for _ in range(batch_size)]
    if seed_text:
        seed_ids = text_encoder.encode([seed_text, ])
        X = [X[i] + seed_ids[0] for i in range(batch_size)]
        sents = [[seed_text] for _ in range(batch_size)]
    XMB = make_batch(X, n_vocab, n_special, batch_size=batch_size)

    for step_n in range(gen_len):
        out = model(XMB) + model.pos_emb_mask
        next_idxs = GAN.Generator._eval_generate_step(out, gen_idx=step_n, top_k=topk, sample=sample, return_list=False)
        idxs = next_idxs.tolist()
        for i in range(batch_size):
            next_token = idxs[i]
            if next_token == n_vocab:
                next_token = "<EOS>"
            else:
                next_token = text_encoder.decoder[next_token].replace('</w>', '')
            sents[i].append(next_token)
        XMB = append_batch(XMB, next_idxs.unsqueeze(-1))

    return [[tok for tok in sent if tok != '\n'] for sent in sents]


def generate_openai(model, text_encoder, n_samples, seed_text, batch_size=10, gen_len=20, topk=100, temperature=1.0,
                    sample=True, n_special=0, print_every=1):
    sents = []
    start_time = time.time()
    n_batches = math.ceil(n_samples / batch_size)
    for batch_n in range(n_batches):
        batch_sents = _generate_sentence_openai(model, text_encoder, seed_text,
                                                batch_size=batch_size, gen_len=gen_len,
                                                topk=topk, sample=sample,
                                                n_special=n_special)
        sents += batch_sents
        settings.write_debug("Generated batch %d of %d in %.3fs" % (batch_n + 1, n_batches, time.time() - start_time))
        start_time = time.time()
    return sents


def corpus_bleu(generated, references):
    """ Compute similarity between two corpora as measured by
    comparing each sentence of `generated` against all sentences in `references`

    args:
        - generated (List[List[str]]): list of sentences (split into tokens)
        - references (List[List[str]]): list of sentences (split into tokens)

    returns:
        - bleu (float)
    """
    return bleu.corpus_bleu([references for _ in range(len(generated))], generated)


def self_bleu(sents):
    return bleu.corpus_bleu([[s for (j, s) in enumerate(sents) if j != i] for i in range(len(sents))], sents)


def get_ngram_counts(sents, max_n=4):
    size2count = {}
    for i in range(1, max_n + 1):
        size2count[i] = Counter([n for sent in sents for n in ngrams(sent, i)])
    return size2count


def ref_unique_ngrams(preds, refs, max_n=4):
    # get # of *distinct* pred ngrams that don't appear in ref
    pct_unique = {}
    pred_ngrams = get_ngram_counts(preds, max_n)
    ref_ngrams = get_ngram_counts(refs, max_n)
    for i in range(1, max_n + 1):
        pred_ngram_counts = set(pred_ngrams[i].keys())
        total = sum(pred_ngrams[i].values())
        ref_ngram_counts = set(ref_ngrams[i].keys())
        pct_unique[i] = len(pred_ngram_counts.difference(ref_ngram_counts)) / total
    return pct_unique


def self_unique_ngrams(preds, max_n=4):
    # get # of pred ngrams with count 1
    pct_unique = {}
    pred_ngrams = get_ngram_counts(preds, max_n)
    for i in range(1, max_n + 1):
        n_unique = len([k for k, v in pred_ngrams[i].items() if v == 1])
        total = sum(pred_ngrams[i].values())
        pct_unique[i] = n_unique / total
    return pct_unique
