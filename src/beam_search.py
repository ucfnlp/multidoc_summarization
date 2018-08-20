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

"""This file contains code to run beam search decoding"""
import cPickle
import warnings

import numpy as np
import os
import data
# import nltk
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
if not "DISPLAY" in os.environ:
    matplotlib.use("Agg")
from matplotlib import pyplot as plt
import textwrap as tw
# import cv2
import PIL
import itertools
from sumy.nlp.tokenizers import Tokenizer
import util
from util import Similarity_Functions
import importance_features
import dill
import time
import random
from absl import flags

FLAGS = flags.FLAGS


class Hypothesis(object):
    """Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis."""

    def __init__(self, tokens, log_probs, state, attn_dists, p_gens, coverage, mmr):
        """Hypothesis constructor.

        Args:
            tokens: List of integers. The ids of the tokens that form the summary so far.
            log_probs: List, same length as tokens, of floats, giving the log probabilities of the tokens so far.
            state: Current state of the decoder, a LSTMStateTuple.
            attn_dists: List, same length as tokens, of numpy arrays with shape (attn_length). These are the attention distributions so far.
            p_gens: List, same length as tokens, of floats, or None if not using pointer-generator model. The values of the generation probability so far.
            coverage: Numpy array of shape (attn_length), or None if not using coverage. The current coverage vector.
        """
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.attn_dists = attn_dists
        self.p_gens = p_gens
        self.coverage = coverage
        self.mmr = mmr

    def extend(self, token, log_prob, state, attn_dist, p_gen, coverage, mmr):
        """Return a NEW hypothesis, extended with the information from the latest step of beam search.

        Args:
            token: Integer. Latest token produced by beam search.
            log_prob: Float. Log prob of the latest token.
            state: Current decoder state, a LSTMStateTuple.
            attn_dist: Attention distribution from latest step. Numpy array shape (attn_length).
            p_gen: Generation probability on latest step. Float.
            coverage: Latest coverage vector. Numpy array shape (attn_length), or None if not using coverage.
        Returns:
            New Hypothesis for next step.
        """
        return Hypothesis(tokens=self.tokens + [token],
                          log_probs=self.log_probs + [log_prob],
                          state=state,
                          attn_dists=self.attn_dists + [attn_dist],
                          p_gens=self.p_gens + [p_gen],
                          coverage=coverage,
                          mmr=mmr)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def log_prob(self):
        # the log probability of the hypothesis so far is the sum of the log probabilities of the tokens so far
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        # normalize log probability by number of tokens (otherwise longer sequences always have lower probability)
        return self.log_prob / len(self.tokens)


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.

    Rows are scores for each class.
    Columns are predictions (samples).
    """
    scoreMatExp = np.exp(np.asarray(x))
    return scoreMatExp / np.expand_dims(scoreMatExp.sum(-1), -1)

def plot_importances(article_sents, importances, abstracts_text, save_location=None, save_name=None):
    if save_location is not None:
        plt.ioff()

    sents_per_figure = 40
    num_sents = len(importances)
    max_importance = np.max(importances)
    chunked_sents = util.chunks(article_sents, sents_per_figure)
    chunked_importances = util.chunks(importances, sents_per_figure)

    for chunk_idx in range(len(chunked_sents)):
        my_article_sents = chunked_sents[chunk_idx]
        my_importances = chunked_importances[chunk_idx]

        if len(my_article_sents) < sents_per_figure:
            my_article_sents += [''] * (sents_per_figure-len(my_article_sents))
            my_importances = np.concatenate([my_importances, np.zeros([sents_per_figure-len(my_importances)])])

        y_pos = np.arange(len(my_article_sents))
        fig, ax1 = plt.subplots()
        fig.subplots_adjust(left=0.9, top=1.0, bottom=0.03, right=1.0)
        ax1.barh(y_pos, my_importances, align='center',
                 color='green', ecolor='black')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(my_article_sents)
        ax1.invert_yaxis()  # labels read top-to-bottom
        ax1.set_xlabel('Performance')
        ax1.set_title('How fast do you want to go today?')
        ax1.set_xlim(right=max_importance)

        # ax1.text(0.5,0.5,wrap(abstract_texts[0], 100))
        # plt.tight_layout()
        if save_location is not None:
            fig.set_size_inches(18.5, 10.5)
            plt.savefig(os.path.join(save_location, save_name + '_' + str(chunk_idx) + '.jpg'))
            plt.close(fig)
        else:
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.show()

    # fig, ax1 = plt.subplots()
    plt.figure()
    comment2_txt = '''\
        Notes: Sales for Product A have been flat through the year. We
        expect improvement after the new release in Q2.
        '''
    fig_txt = tw.fill(tw.dedent(abstracts_text), width=80)
    plt.figtext(0.5, 0.5, fig_txt, horizontalalignment='center',
                fontsize=9, multialignment='left',
                bbox=dict(boxstyle="round", facecolor='#D8D8D8',
                          ec="0.5", pad=0.5, alpha=1), fontweight='bold')
    # plt.tight_layout()
    if save_location is not None:
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.savefig(os.path.join(save_location, save_name + '_' + str(chunk_idx+1) + '.jpg'))
        plt.close(fig)
    else:
        plt.show()


def get_summ_sents_and_tokens(summ_tokens, tokenizer, batch, vocab, chunk_size=-1):
    summ_str = importance_features.tokens_to_continuous_text(summ_tokens, vocab, batch.art_oovs[0])
    if chunk_size == -1:
        sentences = tokenizer.to_sentences(summ_str)
        if data.PERIOD not in sentences[-1]:
            sentences = sentences[:len(sentences) - 1]  # Doesn't include the last sentence if incomplete (no period)
    else:
        all_tokens = summ_str.strip().split(' ')
        chunked_tokens = util.chunk_tokens(all_tokens, chunk_size)
        sentences = [' '.join(chunk) for chunk in chunked_tokens]
        if len(sentences[-1]) < chunk_size:
            sentences = sentences[:len(sentences) - 1]
    sent_words = []
    sent_tokens = []
    token_idx = 0
    for sentence in sentences:
        words = sentence.split(' ')
        sent_words.append(words)
        tokens = summ_tokens[token_idx:token_idx + len(words)]
        sent_tokens.append(tokens)
        token_idx += len(words)
    return sent_words, sent_tokens

def get_similarity_for_one_summ_sent(enc_sentences, enc_tokens, summ_tokens, vocab):
    if len(summ_tokens) == 0:
        return np.zeros([len(enc_sentences)], dtype=float) / len(enc_sentences)
    else:
        # normalization_fn = l1_normalize
        importances_hat = Similarity_Functions.get_similarity(enc_tokens, summ_tokens[-1:], FLAGS.similarity_fn, vocab)
        # importances = normalization_fn(importances_hat)
        importances = importances_hat
        return importances

def convert_to_word_level(mmr_for_sentences, batch, enc_tokens):
    mmr = np.ones([len(batch.enc_batch[0])], dtype=float) / len(batch.enc_batch[0])
    # Calculate how much for each word in source
    word_idx = 0
    for sent_idx in range(len(enc_tokens)):
        mmr_for_words = np.full([len(enc_tokens[sent_idx])], mmr_for_sentences[sent_idx])
        mmr[word_idx:word_idx + len(mmr_for_words)] = mmr_for_words
        word_idx += len(mmr_for_words)
    return mmr


def l1_normalize(importances):
    return importances / np.sum(importances)

def save_importances_and_coverages(importances, importances_hat, enc_sentences,
                                   enc_tokens, hyp, sess, batch, vocab, tokenizer, ex_index, sort=True):
    enc_sentences_str = [' '.join(sent) for sent in enc_sentences]
    summ_sents, summ_tokens = get_summ_sents_and_tokens(hyp.tokens, tokenizer, batch, vocab, FLAGS.chunk_size)
    prev_mmr = importances

    if sort:
        sort_order = np.argsort(importances, 0)[::-1]
        enc_sentences_str = util.reorder(enc_sentences_str, sort_order)

    for sent_idx in range(0, len(summ_sents)):
        cur_summ_sents = summ_sents[:sent_idx]
        cur_summ_tokens = summ_tokens[:sent_idx]
        summ_str = ' '.join([' '.join(sent) for sent in cur_summ_sents])
        similarity_amount = Similarity_Functions.get_similarity(enc_tokens, cur_summ_tokens, FLAGS.similarity_fn, vocab)

        if FLAGS.pg_mmr:
            mmr_for_sentences = calc_mmr_from_sim_and_imp(similarity_amount, importances, prev_mmr, batch, enc_tokens)
        else:
            mmr_for_sentences = None  # Don't use mmr if no sentence-level option is used

        distr_dir = os.path.join(FLAGS.log_root, 'mmr_distributions')
        if not os.path.exists(distr_dir):
            os.makedirs(distr_dir)
        save_name = os.path.join("%06d_decoded_%s_%d_sent" % (ex_index, '', sent_idx))
        file_path = os.path.join(distr_dir, save_name)
        np.savez(file_path, mmr=mmr_for_sentences, importances=importances, enc_sentences=enc_sentences, summ_str=summ_str)
        distributions = [('importance_hat', importances_hat),
                         ('similarity', similarity_amount),
                         ('importance', importances),
                         ('mmr', mmr_for_sentences)]
        for distr_str, distribution in distributions:
            if sort:
                distribution = distribution[sort_order]
            save_name = os.path.join("%06d_decoded_%s_%d_sent" % (ex_index, distr_str, sent_idx))
            plot_importances(enc_sentences_str, distribution, summ_str, save_location=distr_dir, save_name=save_name)

            img_file_names = sorted([file_name for file_name in os.listdir(distr_dir)
                                     if save_name in file_name and 'jpg' in file_name
                                     and 'combined' not in file_name])
            imgs = []
            for file_name in img_file_names:
                img = PIL.Image.open(os.path.join(distr_dir, file_name))
                imgs.append(img)
            max_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[-1][1]
            combined_img = np.vstack( (np.asarray( i.resize(max_shape) ) for i in imgs ) )
            combined_img = PIL.Image.fromarray(combined_img)
            combined_img.save(os.path.join(distr_dir, save_name+'_combined.jpg'))
            for file_name in img_file_names:
                os.remove(os.path.join(distr_dir, file_name))
        prev_mmr = mmr_for_sentences
    return mmr_for_sentences

def calc_mmr_from_sim_and_imp(similarity, importances, prev_mmr, batch, enc_tokens):
    new_mmr =  FLAGS.lambda_val*importances - (1-FLAGS.lambda_val)*similarity
    new_mmr = np.maximum(new_mmr, 0)
    return new_mmr

def mute_all_except_top_k(array, k):
    num_reservoirs_still_full = np.sum(array > 0)
    if num_reservoirs_still_full < k:
        selected_indices = np.nonzero(array)
    else:
        selected_indices = array.argsort()[::-1][:k]
    res = np.zeros_like(array, dtype=float)
    for selected_idx in selected_indices:
        if FLAGS.retain_mmr_values:
            res[selected_idx] = array[selected_idx]
        else:
            res[selected_idx] = 1.
    return res

def get_tokens_for_human_summaries(batch, vocab):
    art_oovs = batch.art_oovs[0]
    def get_all_summ_tokens(all_summs):
        return [get_summ_tokens(summ) for summ in all_summs]
    def get_summ_tokens(summ):
        summ_tokens = [get_sent_tokens(sent) for sent in summ]
        return list(itertools.chain.from_iterable(summ_tokens))     # combines all sentences into one list of tokens for summary
    def get_sent_tokens(sent):
        words = sent.split()
        return data.abstract2ids(words, vocab, art_oovs)
    human_summaries = batch.all_original_abstracts_sents[0]
    all_summ_tokens = get_all_summ_tokens(human_summaries)
    return all_summ_tokens

def get_svr_importances(enc_states, enc_sentences, enc_sent_indices, svr_model, tokenizer, sent_representations_separate):
    sent_indices = enc_sent_indices
    sent_reps = importance_features.get_importance_features_for_article(
        enc_states, enc_sentences, sent_indices, tokenizer, sent_representations_separate)
    features_list = importance_features.get_features_list(False)
    x = importance_features.features_to_array(sent_reps, features_list)
    if FLAGS.importance_fn == 'svr':
        importances = svr_model.predict(x)
    else:
        importances = svr_model.decision_function(x)
    return importances

def save_sorted_sentences(importances, raw_article_sents, ex_index):
    ranked_sent_indices = np.argsort(importances)[::-1]
    if len(ranked_sent_indices) != len(raw_article_sents):
        warnings.warn('len(ranked_sent_indices) (%d) != len(raw_article_sents) (%d)' % ( len(ranked_sent_indices), len(raw_article_sents)))
    ranked_sents = [raw_article_sents[idx] for idx in ranked_sent_indices]
    output_str = '\n'.join(ranked_sents)
    ranked_sent_path = os.path.join(FLAGS.log_root, 'ordered_sentences')
    if not os.path.exists(ranked_sent_path):
        os.makedirs(ranked_sent_path)
    with open(os.path.join(ranked_sent_path, 'test_%03d.txt' % ex_index), 'wb') as f:
        f.write(output_str)

def get_tfidf_importances(raw_article_sents):
    tfidf_model_path = os.path.join(FLAGS.actual_log_root, 'tfidf_vectorizer', FLAGS.dataset_name + '.dill')

    while True:
        try:
            with open(tfidf_model_path, 'rb') as f:
                tfidf_vectorizer = dill.load(f)
            break
        except (EOFError, KeyError):
            time.sleep(random.randint(3,6))
            continue
    sent_reps = tfidf_vectorizer.transform(raw_article_sents)
    cluster_rep = np.mean(sent_reps, axis=0)
    similarity_matrix = cosine_similarity(sent_reps, cluster_rep)
    return np.squeeze(similarity_matrix)


def run_beam_search(sess, model, vocab, batch, ex_index, hps):
    """Performs beam search decoding on the given example.

    Args:
        sess: a tf.Session
        model: a seq2seq model
        vocab: Vocabulary object
        batch: Batch object that is the same example repeated across the batch

    Returns:
        best_hyp: Hypothesis object; the best hypothesis found by beam search.
    """

    max_dec_steps = FLAGS.max_dec_steps
    # Run the encoder to get the encoder hidden states and decoder initial state
    enc_states, dec_in_state = model.run_encoder(sess, batch)
    # dec_in_state is a LSTMStateTuple
    # enc_states has shape [batch_size, <=max_enc_steps, 2*hidden_dim].


    if FLAGS.pg_mmr:

        if FLAGS.importance_fn == 'svr':
            with open(os.path.join(FLAGS.actual_log_root, FLAGS.importance_model_name + '_' + str(FLAGS.svr_num_documents) + '.pickle'), 'rb') as f:
                svr_model = cPickle.load(f)
        tokenizer = Tokenizer('english')

        enc_sentences, enc_tokens = batch.tokenized_sents[0], batch.word_ids_sents[0]
        enc_sent_indices = importance_features.get_sent_indices(enc_sentences, batch.doc_indices[0])
        sent_representations_separate = importance_features.get_separate_enc_states(model, sess, enc_sentences, vocab, hps)


        if FLAGS.importance_fn == 'oracle':
            human_tokens = get_tokens_for_human_summaries(batch, vocab)     # list (of 4 human summaries) of list of token ids
            metric = 'recall'
            importances_hat = Similarity_Functions.rouge_l_similarity(enc_tokens, human_tokens, vocab, metric=metric)
        elif FLAGS.importance_fn == 'svr':
            importances_hat = get_svr_importances(enc_states[0], enc_sentences, enc_sent_indices, svr_model, tokenizer, sent_representations_separate)
        elif FLAGS.importance_fn == 'tfidf':
            importances_hat = get_tfidf_importances(batch.raw_article_sents[0])
        importances = util.special_squash(importances_hat)

        save_sorted_sentences(importances, batch.raw_article_sents[0], ex_index)
    else:
        importances = None

    if FLAGS.pg_mmr:
        similarity = np.zeros([FLAGS.beam_size, len(enc_sentences)], dtype=float)
        mmr_init = importances
    else:
        mmr_init = importances


    # Initialize beam_size-many hyptheses
    hyps = [Hypothesis(tokens=[vocab.word2id(data.START_DECODING)],
                       log_probs=[0.0],
                       state=dec_in_state,
                       attn_dists=[],
                       p_gens=[],
                       coverage=np.zeros([batch.enc_batch.shape[1]]),  # zero vector of length attention_length
                       mmr=mmr_init
                       ) for hyp_idx in xrange(FLAGS.beam_size)]
    results = []  # this will contain finished hypotheses (those that have emitted the [STOP] token)


    steps = 0
    while steps < max_dec_steps and len(results) < FLAGS.beam_size:

        latest_tokens = [h.latest_token for h in hyps]  # latest token produced by each hypothesis
        latest_tokens = [t if t in xrange(vocab.size()) else vocab.word2id(data.UNKNOWN_TOKEN) for t in
                         latest_tokens]  # change any in-article temporary OOV ids to [UNK] id, so that we can lookup word embeddings







        states = [h.state for h in hyps]  # list of current decoder states of the hypotheses
        prev_coverage = [h.coverage for h in hyps]  # list of coverage vectors (or None)
        prev_mmr = [h.mmr for h in hyps]
        if FLAGS.pg_mmr:
            if FLAGS.mute_k != -1:
                prev_mmr = [mute_all_except_top_k(mmr, FLAGS.mute_k) for mmr in prev_mmr]
            prev_mmr_for_words = [convert_to_word_level(mmr, batch, enc_tokens) for mmr in prev_mmr]
        else:
            prev_mmr_for_words = [None for _ in prev_mmr]


        # Run one step of the decoder to get the new info
        (topk_ids, topk_log_probs, new_states, attn_dists, p_gens, new_coverage, pre_attn_dists) = model.decode_onestep(sess=sess,
                                                                                                        batch=batch,
                                                                                                        latest_tokens=latest_tokens,
                                                                                                        enc_states=enc_states,
                                                                                                        dec_init_states=states,
                                                                                                        prev_coverage=prev_coverage,
                                                                                                        mmr_score=prev_mmr_for_words)

        # Extend each hypothesis and collect them all in all_hyps
        all_hyps = []
        num_orig_hyps = 1 if steps == 0 else len(
            hyps)  # On the first step, we only had one original hypothesis (the initial hypothesis). On subsequent steps, all original hypotheses are distinct.
        for i in xrange(num_orig_hyps):
            h, new_state, attn_dist, p_gen, new_coverage_i = hyps[i], new_states[i], attn_dists[i], p_gens[i], \
                                                             new_coverage[
                                                                 i]  # take the ith hypothesis and new decoder state info
            for j in xrange(FLAGS.beam_size * 2):  # for each of the top 2*beam_size hyps:
                # Extend the ith hypothesis with the jth option
                new_hyp = h.extend(token=topk_ids[i, j],
                                   log_prob=topk_log_probs[i, j],
                                   state=new_state,
                                   attn_dist=attn_dist,
                                   p_gen=p_gen,
                                   coverage=new_coverage_i,
                                   mmr=h.mmr)
                all_hyps.append(new_hyp)

        # Filter and collect any hypotheses that have produced the end token.
        hyps = []  # will contain hypotheses for the next step
        for h in sort_hyps(all_hyps):  # in order of most likely h
            if h.latest_token == vocab.word2id(data.STOP_DECODING):  # if stop token is reached...
                # If this hypothesis is sufficiently long, put in results. Otherwise discard.
                if steps >= FLAGS.min_dec_steps:
                    results.append(h)
            else:  # hasn't reached stop token, so continue to extend this hypothesis
                hyps.append(h)
            if len(hyps) == FLAGS.beam_size or len(results) == FLAGS.beam_size:
                # Once we've collected beam_size-many hypotheses for the next step, or beam_size-many complete hypotheses, stop.
                break

        if FLAGS.pg_mmr:
            for hyp_idx, hyp in enumerate(hyps):
                if (hyp.latest_token == vocab.word2id(data.PERIOD) and FLAGS.chunk_size == -1) or (    # if in regular mode, and the hyp ends in a period
                    FLAGS.chunk_size > -1 and len(hyp.tokens) % FLAGS.chunk_size == 0   # if in chunk mode, and hyp just finished a chunk
                ) or (not FLAGS.coverage_optimization):
                    summ_sents, summ_tokens = get_summ_sents_and_tokens(hyp.tokens, tokenizer, batch, vocab, FLAGS.chunk_size)
                    similarity_amount = Similarity_Functions.get_similarity(enc_tokens, summ_tokens, FLAGS.similarity_fn, vocab)
                    similarity[hyp_idx] = similarity_amount
                    a = 0
                    hyp.mmr = calc_mmr_from_sim_and_imp(similarity[hyp_idx], importances, hyp.mmr, batch, enc_tokens)
        steps += 1

    # At this point, either we've got beam_size results, or we've reached maximum decoder steps

    if len(
            results) == 0:  # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
        results = hyps

    # Sort hypotheses by average log probability
    hyps_sorted = sort_hyps(results)
    best_hyp = hyps_sorted[0]

    if FLAGS.save_distributions and FLAGS.pg_mmr:
        save_importances_and_coverages(importances, importances_hat, enc_sentences,
                                   enc_tokens, best_hyp, sess, batch, vocab, tokenizer, ex_index)


    # Return the hypothesis with highest average log prob
    return best_hyp


def sort_hyps(hyps):
    """Return a list of Hypothesis objects, sorted by descending average log probability"""
    return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)


