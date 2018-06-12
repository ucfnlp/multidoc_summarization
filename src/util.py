# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains some utility functions"""

import tensorflow as tf
import time
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from absl import flags
FLAGS = flags.FLAGS
import itertools
from matplotlib import pyplot as plt
import data
from absl import logging

def get_config():
    """Returns config for tf.session"""
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    return config

def load_ckpt(saver, sess, ckpt_dir="train"):
    """Load checkpoint from the ckpt_dir (if unspecified, this is train dir) and restore it to saver and sess, waiting 10 secs in the case of failure. Also returns checkpoint name."""
    while True:
        try:
            latest_filename = "checkpoint_best" if ckpt_dir=="eval" else None
            if FLAGS.use_pretrained:
                ckpt_dir = FLAGS.pretrained_path
            else:
                ckpt_dir = os.path.join(FLAGS.log_root, ckpt_dir)
            ckpt_state = tf.train.get_checkpoint_state(ckpt_dir, latest_filename=latest_filename)
            logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            return ckpt_state.model_checkpoint_path
        except:
            logging.info("Failed to load checkpoint from %s. Sleeping for %i secs...", ckpt_dir, 10)
            time.sleep(10)

class InfinityValueError(ValueError): pass


def create_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def flatten_list_of_lists(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))

def chunk_tokens(tokens, chunk_size):
    chunk_size = max(1, chunk_size)
    return (tokens[i:i+chunk_size] for i in xrange(0, len(tokens), chunk_size))

def chunks(chunkable, n):
    """ Yield successive n-sized chunks from l.
    """
    chunk_list = []
    for i in xrange(0, len(chunkable), n):
        chunk_list.append( chunkable[i:i+n])
    return chunk_list

def plot_bar_graph(values):
    plt.figure()
    plt.bar(np.arange(len(values)), values)

def is_list_type(obj):
    return isinstance(obj, (list, tuple, np.ndarray))

def remove_period_ids(lst, vocab):
    if len(lst) == 0:
        return lst
    if is_list_type(lst[0]):
        return [[item for item in inner_list if item != vocab.word2id(data.PERIOD)] for inner_list in lst]
    else:
        return [item for item in lst if item != vocab.word2id(data.PERIOD)]

def to_unicode(text):
    try:
        text = unicode(text, errors='replace')
    except TypeError:
        return text
    return text

def reorder(l, ordering):
    return [l[i] for i in ordering]

def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings

    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if (len(string) < len(sub)):
        sub, string = string, sub

    lengths = [[0 for i in range(0, len(sub) + 1)] for j in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if (string[i - 1] == sub[j - 1]):
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    return lengths[len(string)][len(sub)]


def get_nGram(l, n=2):
    l = list(l)
    return list(set(zip(*[l[i:] for i in range(n)])))

def calc_ROUGE_L_score(candidate, reference, metric='f1'):
    """
    Compute ROUGE-L score given one candidate and references for an image
    :param candidate: str : candidate sentence to be evaluated
    :param refs: list of str : COCO reference sentences for the particular image to be evaluated
    :returns score: int (ROUGE-L score for the candidate evaluated against references)
    """
    # assert (len(candidate) == 1)
    # assert (len(refs) > 0)
    beta = 1.2
    prec = []
    rec = []

    if len(reference) == 0:
        return 0.

    if type(reference[0]) is not list:
        reference = [reference]

    for ref in reference:
        # compute the longest common subsequence
        lcs = my_lcs(ref, candidate)
        prec.append(lcs / float(len(candidate)))
        rec.append(lcs / float(len(ref)))


    prec_max = max(prec)
    rec_max = max(rec)

    if metric == 'f1':
        if (prec_max != 0 and rec_max != 0):
            score = ((1 + beta ** 2) * prec_max * rec_max) / float(rec_max + beta ** 2 * prec_max)
        else:
            score = 0.0
    elif metric == 'precision':
        score = prec_max
    elif metric == 'recall':
        score = rec_max
    else:
        raise Exception('Invalid metric argument: %s. Must be one of {f1,precision,recall}.' % metric)
    return score

class Similarity_Functions:
    '''
    Class for computing sentence similarity between a set of source sentences and a set of summary sentences
    
    '''
    @classmethod
    def get_similarity(cls, enc_sent_embs, enc_words_embs_list, enc_tokens, summ_embeddings, summ_words_embs_list,
                       summ_tokens, similarity_fn, vocab):
        if similarity_fn == 'cosine_similarity':
            similarity_matrix = cosine_similarity(enc_sent_embs, summ_embeddings)
            # Calculate amount of uncovered information for each sentence in the source
            importances_hat = np.sum(similarity_matrix, 1)
        elif similarity_fn == 'tokenwise_sentence_similarity':
            similarity_matrix = cls.tokenwise_sentence_similarity(enc_words_embs_list, summ_words_embs_list)
            # Calculate amount of uncovered information for each sentence in the source
            importances_hat = np.sum(similarity_matrix, 1)
        elif similarity_fn == 'ngram_similarity':
            importances_hat = cls.ngram_similarity(enc_tokens, summ_tokens)
        elif similarity_fn == 'rouge_l':
            metric = 'precision' if FLAGS.rouge_l_prec_rec else 'f1'
            # similarity_matrix = cls.rouge_l_similarity(enc_tokens, summ_tokens, metric=metric)
            # importances_hat = np.sum(similarity_matrix, 1)

            # importances_hat = cls.rouge_l_similarity(enc_tokens, summ_tokens, metric=metric)
            summ_tokens_combined = flatten_list_of_lists(summ_tokens)
            importances_hat = cls.rouge_l_similarity(enc_tokens, summ_tokens_combined, vocab, metric=metric)
        return importances_hat

    @classmethod
    def rouge_l_similarity(cls, article_sents, abstract_sents, vocab, metric='f1'):
        # sentence_similarity_matrix = np.zeros([len(article_sents), len(abstract_sents)], dtype=float)
        sentence_similarity = np.zeros([len(article_sents)], dtype=float)
        abstract_sents_removed_periods = remove_period_ids(abstract_sents, vocab)
        for article_sent_idx, article_sent in enumerate(article_sents):
            rouge_l = calc_ROUGE_L_score(article_sent, abstract_sents_removed_periods, metric=metric)
            # if eval_sents_indiv:
            #     abs_similarities = []
            #     for abstract_sent_idx, abstract_sent in enumerate(abstract_sents):
            #         rouge_l = calc_ROUGE_L_score(article_sent, abstract_sent, metric=metric)
            #         abs_similarities.append(rouge_l)
            #         sentence_similarity_matrix[article_sent_idx, abstract_sent_idx] = rouge_l
            # else:
            sentence_similarity[article_sent_idx] = rouge_l
        return sentence_similarity

    @classmethod
    def rouge_l_similarity_matrix(cls, article_sents, abstract_sents, vocab, metric='f1'):
        sentence_similarity_matrix = np.zeros([len(article_sents), len(abstract_sents)], dtype=float)
        abstract_sents_removed_periods = remove_period_ids(abstract_sents, vocab)
        for article_sent_idx, article_sent in enumerate(article_sents):
            rouge_l = calc_ROUGE_L_score(article_sent, abstract_sents_removed_periods, metric=metric)
            abs_similarities = []
            for abstract_sent_idx, abstract_sent in enumerate(abstract_sents):
                rouge_l = calc_ROUGE_L_score(article_sent, abstract_sent, metric=metric)
                abs_similarities.append(rouge_l)
                sentence_similarity_matrix[article_sent_idx, abstract_sent_idx] = rouge_l
        return sentence_similarity_matrix


    @classmethod
    def ngram_similarity(cls, article_sents, abstract_sents):
        abstracts_ngrams = [get_nGram(sent) for sent in abstract_sents]  # collect ngrams
        abstracts_ngrams = sum(abstracts_ngrams, [])  # combine ngrams in all sentences
        abstracts_ngrams = list(set(abstracts_ngrams))  # only unique ngrams
        res = np.zeros((len(article_sents)), dtype=float)
        for art_idx, art_sent in enumerate(article_sents):
            article_ngrams = get_nGram(art_sent)
            num_appear_in_source = sum([1 for ngram in article_ngrams if ngram in abstracts_ngrams])
            if len(article_ngrams) == 0:
                percent_appear_in_source = 0.
            else:
                percent_appear_in_source = num_appear_in_source * 1. / len(article_ngrams)
            res[art_idx] = percent_appear_in_source
        return res

    @classmethod
    def tokenwise_sentence_similarity(cls, enc_words_embs_list, summ_words_embs_list):
        sentence_similarity_matrix = np.zeros([len(enc_words_embs_list), len(summ_words_embs_list)], dtype=float)
        for enc_sent_idx, enc_sent in enumerate(enc_words_embs_list):
            for summ_sent_idx, summ_sent in enumerate(summ_words_embs_list):
                similarity_matrix = cosine_similarity(enc_sent, summ_sent)
                pair_similarity = np.sum(similarity_matrix) / np.size(similarity_matrix)
                pair_similarity = max(0, pair_similarity)  # don't allow negative values
                sentence_similarity_matrix[enc_sent_idx, summ_sent_idx] = pair_similarity
        return sentence_similarity_matrix


































