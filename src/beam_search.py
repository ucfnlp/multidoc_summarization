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

import tensorflow as tf
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
from nltk.corpus import stopwords
import string
# import cv2
import PIL
import itertools
from sumy.nlp.tokenizers import Tokenizer
from lex_rank_importance import LexRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
import util
from util import Similarity_Functions

FLAGS = tf.app.flags.FLAGS


class Hypothesis(object):
    """Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis."""

    def __init__(self, tokens, log_probs, state, attn_dists, p_gens, coverage, beta):
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
        self.beta = beta

    def extend(self, token, log_prob, state, attn_dist, p_gen, coverage, beta):
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
                          beta=beta)

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

def chunks(chunkable, n):
    """ Yield successive n-sized chunks from l.
    """
    chunk_list = []
    for i in xrange(0, len(chunkable), n):
        chunk_list.append( chunkable[i:i+n])
    return chunk_list

def plot_importances(article_sents, importances, abstracts_text, save_location=None, save_name=None):
    if save_location is not None:
        plt.ioff()

    sents_per_figure = 40
    num_sents = len(importances)
    max_importance = np.max(importances)
    chunked_sents = chunks(article_sents, sents_per_figure)
    chunked_importances = chunks(importances, sents_per_figure)

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

def chunk_tokens(tokens, chunk_size):
    chunk_size = max(1, chunk_size)
    return (tokens[i:i+chunk_size] for i in xrange(0, len(tokens), chunk_size))

def flatten_list_of_lists(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))

def tokens_to_continuous_text(tokens, vocab, art_oovs):
    words = data.outputids2words(tokens, vocab, art_oovs)
    text = ' '.join(words)
    text = text.decode('utf8')
    return text

def get_enc_sents_and_tokens_with_cutoff_length(original_article, enc_batch_extend_vocab, cutoff_len, tokenizer,
                                                art_oovs, vocab, chunk_size=-1):
    enc_text = tokens_to_continuous_text(enc_batch_extend_vocab, vocab, art_oovs)
    if chunk_size == -1:
        all_enc_sentences = tokenizer.to_sentences(enc_text)
        all_tokens = flatten_list_of_lists([sent.split() for sent in all_enc_sentences])
    else:
        all_tokens = enc_text.split()
        chunked_tokens = chunk_tokens(all_tokens, chunk_size)
        all_enc_sentences = [' '.join(chunk) for chunk in chunked_tokens]
    if len(all_tokens) != len(enc_batch_extend_vocab):
        raise Exception('All_tokens ('+str(len(all_tokens))+
                        ') does not have the same number of tokens as enc_batch ('+str(len(enc_batch_extend_vocab))+')')
    select_enc_sentences = []
    select_enc_tokens = []
    count = 0
    for sent_idx, sent in enumerate(all_enc_sentences):
        sent_to_add = []
        tokens_to_add = []
        should_break = False
        tokens = sent.split()
        for word_idx, word in enumerate(tokens):
            sent_to_add.append(word)
            tokens_to_add.append(enc_batch_extend_vocab[count])
            count += 1
            if count == cutoff_len:
                should_break = True
                break
        select_enc_sentences.append(sent_to_add)
        select_enc_tokens.append(tokens_to_add)
        if should_break:
            break
    return select_enc_sentences, select_enc_tokens


def get_sentences_embeddings(enc_sentences, enc_tokens, sess, batch, vocab):
    embedding_matrix = [v for v in tf.global_variables() if v.name == "seq2seq/embedding/embedding:0"][0]
    embedding_matrix = embedding_matrix.eval(sess)

    stop_words = set(stopwords.words('english'))
    stop_word_tokens = [vocab.word2id(word) for word in stop_words if word in vocab._word_to_id]

    oov_embs = {}
    for oov_token in batch.art_oovs[0]:
        oov_embs[oov_token] = 0.1 * np.random.randn(embedding_matrix.shape[1]).astype(float)

    sent_embs = []
    word_embs_list = []
    words_list = []
    for sent_idx, sent in enumerate(enc_sentences):
        embs = []
        words = []
        for word_idx, word in enumerate(sent):
            token = enc_tokens[sent_idx][word_idx]
            if word in stop_words or word in ('<s>', '</s>'):
                continue
            is_punctuation = [ch in string.punctuation for ch in word]
            if all(is_punctuation):
                continue
            if word in ('-lrb-', '-rrb-', '-lsb-', '-rsb-'):
                continue
            if word in oov_embs:
                emb = oov_embs[word]
            else:
                if token < len(embedding_matrix):
                    emb = embedding_matrix[token]
                else:
                    print('Didnt find word: ' + word)
                    continue
            # emb = emb / np.linalg.norm(emb)
            embs.append(emb)
            words.append(word)
        if len(embs) == 0:
            print (sent)
            embs.append(embedding_matrix[vocab.word2id(data.PAD_TOKEN)])
        embs = np.stack(embs)
        mean_emb = np.mean(embs, 0)
        sent_embs.append(mean_emb)
        word_embs_list.append(embs)
        words_list.append(words)
    return sent_embs, word_embs_list, words_list
    # return np.stack(sent_embs)


def get_summ_sents_and_tokens(summ_tokens, tokenizer, batch, vocab, chunk_size=-1):
    summ_str = tokens_to_continuous_text(summ_tokens, vocab, batch.art_oovs[0])
    if chunk_size == -1:
        sentences = tokenizer.to_sentences(summ_str)
        if data.PERIOD not in sentences[-1]:
            sentences = sentences[:len(sentences) - 1]  # Doesn't include the last sentence if incomplete (no period)
    else:
        all_tokens = summ_str.strip().split(' ')
        chunked_tokens = chunk_tokens(all_tokens, chunk_size)
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

def get_similarity_for_one_summ_sent(enc_sentences, enc_sent_embs, enc_words_embs_list, enc_tokens, summ_embeddings,
                                    summ_words_embs_list, summ_tokens):
    if len(summ_embeddings) == 0:
        return np.zeros([len(enc_sentences)], dtype=float) / len(enc_sentences)
    else:
        normalization_fn = l1_normalize
        importances_hat = Similarity_Functions.get_similarity(enc_sent_embs, enc_words_embs_list, enc_tokens, summ_embeddings[-1:],
                                                    summ_words_embs_list[-1:], summ_tokens[-1:], FLAGS.similarity_fn)
        importances = normalization_fn(importances_hat)
        return importances

def get_coverage(enc_sentences, enc_sent_embs, enc_words_embs_list, enc_tokens, summ_embeddings, summ_words_embs_list, summ_tokens):
    if len(summ_embeddings) == 0:
        return np.ones([len(enc_sentences)], dtype=float) / len(enc_sentences), np.zeros([len(enc_sentences)], dtype=float) / len(enc_sentences)
    else:
        # Calculate similarity matrix [num_sentences_encoder, num_sentences_summary]
        normalization_fn = l1_normalize
        importances_hat = Similarity_Functions.get_similarity(enc_sent_embs, enc_words_embs_list, enc_tokens, summ_embeddings,
                                                    summ_words_embs_list, summ_tokens, FLAGS.similarity_fn)
        importances = normalization_fn(importances_hat)
        uncovered_amount_hat = max(importances_hat) - importances_hat
        uncovered_amount = normalization_fn(uncovered_amount_hat)
        return uncovered_amount, importances


def combine_coverage_and_importance(logan_coverage, logan_importances):
    # if logan_coverage.ndim > 1:
    #     repeated_logan_importances = np.tile(logan_importances, (len(logan_coverage), 1))  # repeat over hypotheses
    #     beta_for_sentences = logan_coverage + logan_importances
    # else:
    beta_for_sentences = logan_coverage + logan_importances
    return beta_for_sentences


def convert_to_word_level(beta_for_sentences, batch, enc_tokens):
    beta = np.ones([len(batch.enc_batch[0])], dtype=float) / len(batch.enc_batch[0])
    # Calculate how much for each word in source
    word_idx = 0
    for sent_idx in range(len(enc_tokens)):
        beta_for_words = np.full([len(enc_tokens[sent_idx])], beta_for_sentences[sent_idx])
        beta[word_idx:word_idx + len(beta_for_words)] = beta_for_words
        word_idx += len(beta_for_words)
    return beta


def l1_normalize(importances):
    return importances / np.sum(importances)

def softmax_trick(distribution, tau):
    return softmax(distribution / tau)


def save_importances_and_coverages(logan_importances, enc_sentences, enc_sent_embs, enc_words_embs_list,
                                   enc_tokens, hyp, sess, batch, vocab, tokenizer, ex_index):
    enc_sentences_str = [' '.join(sent) for sent in enc_sentences]
    summ_sents, summ_tokens = get_summ_sents_and_tokens(hyp.tokens, tokenizer, batch, vocab, FLAGS.logan_chunk_size)
    summ_embeddings, summ_words_embs_list, summ_words_list = get_sentences_embeddings(summ_sents, summ_tokens,
                                                                                      sess, batch, vocab)
    prev_beta = logan_importances

    for sent_idx in range(0, len(summ_sents)):
        cur_summ_sents = summ_sents[:sent_idx]
        cur_summ_embeddings = summ_embeddings[:sent_idx]
        cur_summ_words_embs_list = summ_words_embs_list[:sent_idx]
        cur_summ_tokens = summ_tokens[:sent_idx]
        summ_str = ' '.join([' '.join(sent) for sent in cur_summ_sents])
        uncovered_amount, _ = get_coverage(enc_sentences, enc_sent_embs, enc_words_embs_list, enc_tokens,
                                        cur_summ_embeddings, cur_summ_words_embs_list, cur_summ_tokens)
        similarity_amount = get_similarity_for_one_summ_sent(enc_sentences, enc_sent_embs, enc_words_embs_list, enc_tokens,
                                        cur_summ_embeddings, cur_summ_words_embs_list, cur_summ_tokens)
        if FLAGS.logan_coverage:
            if FLAGS.logan_importance:                  # if both sentence-level options are on
                beta_for_sentences = combine_coverage_and_importance(uncovered_amount, logan_importances)
            else:
                beta_for_sentences = uncovered_amount  # if only sentence-level coverage is on
        elif FLAGS.logan_reservoir:
            beta_for_sentences = calc_beta_from_sim_and_imp(similarity_amount, logan_importances, prev_beta, batch, enc_tokens)
        elif FLAGS.logan_importance:
            beta_for_sentences = logan_importances  # if only sentence-level importance is on
        else:
            beta_for_sentences = None  # Don't use beta if no sentence-level option is used

        distr_dir = os.path.join(FLAGS.log_root, 'beta_distributions')
        if not os.path.exists(distr_dir):
            os.makedirs(distr_dir)
        save_name = os.path.join("%06d_decoded_%s_%d_sent" % (ex_index, '', sent_idx))
        file_path = os.path.join(distr_dir, save_name)
        np.savez(file_path, beta=beta_for_sentences, importances=logan_importances,
                 coverages=uncovered_amount, enc_sentences=enc_sentences, summ_str=summ_str)
        distributions = [('coverage', uncovered_amount),
                         ('similarity', similarity_amount),
                         ('importance', logan_importances),
                         ('beta', beta_for_sentences)]
        for distr_str, distribution in distributions:
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
        prev_beta = beta_for_sentences

def calc_beta_from_cov_and_imp(logan_coverage, logan_importances, batch, enc_tokens):
    if FLAGS.logan_coverage and FLAGS.logan_importance:     # if both sentence-level options are on
        beta_for_sentences = combine_coverage_and_importance(logan_coverage, logan_importances)
    elif FLAGS.logan_coverage:
        beta_for_sentences = logan_coverage     # if only sentence-level coverage is on
    elif FLAGS.logan_importance:
        beta_for_sentences = logan_importances    # if only sentence-level importance is on
    else:
        beta_for_sentences = None     # Don't use beta if it's neither sentence-level option is used

    if beta_for_sentences is not None:
        # beta = convert_to_word_level(beta_for_sentences, batch, enc_tokens)
        if FLAGS.logan_beta_tau != 1.0:
            beta_for_sentences = softmax_trick(beta_for_sentences, FLAGS.logan_beta_tau)
    return beta_for_sentences

def special_squash(distribution):
    res = distribution - np.min(distribution)
    if np.max(res) == 0:
        print('All elements in distribution are 0, so setting all to 0')
        res.fill(0)
    else:
        res = res / np.max(res)
    return res

def calc_beta_from_sim_and_imp(logan_similarity, logan_importances, prev_beta, batch, enc_tokens):
    # to_subtract_for_sentences = logan_importances * logan_similarity
    # to_subtract = convert_to_word_level(to_subtract_for_sentences, batch, enc_tokens)
    # new_beta = prev_beta - to_subtract
    # new_beta = np.maximum([0], new_beta)
    # return new_beta

    new_beta = special_squash(prev_beta) - special_squash(logan_similarity)
    # new_beta = prev_beta - logan_similarity
    new_beta = np.maximum(new_beta, 0)
    return new_beta

def mute_all_except_top_k(array, k):
    num_reservoirs_still_full = np.sum(array > 0)
    if num_reservoirs_still_full < k:
        selected_indices = np.nonzero(array)
    else:
        selected_indices = array.argsort()[::-1][:k]
    array = np.zeros_like(array, dtype=float)
    array[selected_indices] = 1.
    return array




def run_beam_search(sess, model, vocab, batch, ex_index, specific_max_dec_steps=None):
    """Performs beam search decoding on the given example.

    Args:
        sess: a tf.Session
        model: a seq2seq model
        vocab: Vocabulary object
        batch: Batch object that is the same example repeated across the batch

    Returns:
        best_hyp: Hypothesis object; the best hypothesis found by beam search.
    """

    def ids_to_words(ids):
        if len(ids) > 0 and type(ids[0]) == list:
            return [data.outputids2words(sent_ids, vocab, batch.art_oovs[0]) for sent_ids in ids]
        else:
            return data.outputids2words(ids, vocab, batch.art_oovs[0])

    # Use UPitt's max_dec_steps if it is specified, otherwise, use default max
    max_dec_steps = specific_max_dec_steps if specific_max_dec_steps is not None else FLAGS.max_dec_steps

    # Run the encoder to get the encoder hidden states and decoder initial state
    enc_states, dec_in_state = model.run_encoder(sess, batch)
    # dec_in_state is a LSTMStateTuple
    # enc_states has shape [batch_size, <=max_enc_steps, 2*hidden_dim].




    tokenizer = Tokenizer('english')

    enc_sentences, enc_tokens = get_enc_sents_and_tokens_with_cutoff_length(
                    batch.original_articles[0], batch.enc_batch_extend_vocab[0],
                    len(batch.enc_batch_extend_vocab[0]), tokenizer, batch.art_oovs[0], vocab, FLAGS.logan_chunk_size)
    enc_sent_embs, enc_words_embs_list, enc_words_list = get_sentences_embeddings(enc_sentences, enc_tokens,
                                                                                            sess, batch, vocab)
    enc_sentences_str = [' '.join(sent) for sent in enc_sentences]


    if FLAGS.logan_importance:
        # parser = PlaintextParser.from_string(batch.original_articles[0], tokenizer)
        summarizer = LexRankSummarizer()
        logan_importances = summarizer.get_importances(enc_sentences, tokenizer)
        # plot_importances(enc_sentences_str, logan_importances, 'n/a')
        if FLAGS.logan_importance_tau != 1.0:
            logan_importances = softmax_trick(logan_importances, FLAGS.logan_importance_tau)
    else:
        logan_importances = None

    if FLAGS.logan_reservoir:
        logan_similarity = np.zeros([FLAGS.beam_size, len(enc_sentences)], dtype=float)
        beta_init = logan_importances
    elif FLAGS.logan_coverage:
        # Initial coverage (evenly distributed among all sentences)
        logan_coverage = np.ones([FLAGS.beam_size, len(enc_sentences)], dtype=float) / len(enc_sentences)
        beta_init = (0 if logan_importances is None else logan_importances) + logan_coverage[0]
    else:
        logan_coverage = None
        beta_init = logan_importances



    # Initialize beam_size-many hyptheses
    hyps = [Hypothesis(tokens=[vocab.word2id(data.START_DECODING)],
                       log_probs=[0.0],
                       state=dec_in_state,
                       attn_dists=[],
                       p_gens=[],
                       coverage=np.zeros([batch.enc_batch.shape[1]]),  # zero vector of length attention_length
                       beta=beta_init
                       ) for hyp_idx in xrange(FLAGS.beam_size)]
    results = []  # this will contain finished hypotheses (those that have emitted the [STOP] token)


    steps = 0
    while steps < max_dec_steps and len(results) < FLAGS.beam_size:

        latest_tokens = [h.latest_token for h in hyps]  # latest token produced by each hypothesis
        latest_tokens = [t if t in xrange(vocab.size()) else vocab.word2id(data.UNKNOWN_TOKEN) for t in
                         latest_tokens]  # change any in-article temporary OOV ids to [UNK] id, so that we can lookup word embeddings







        states = [h.state for h in hyps]  # list of current decoder states of the hypotheses
        prev_coverage = [h.coverage for h in hyps]  # list of coverage vectors (or None)
        prev_beta = [h.beta for h in hyps]
        if FLAGS.logan_beta:
            if FLAGS.logan_mute:
                prev_beta = [mute_all_except_top_k(beta, FLAGS.logan_mute_k) for beta in prev_beta]
            prev_beta_for_words = [convert_to_word_level(beta, batch, enc_tokens) for beta in prev_beta]
        else:
            prev_beta_for_words = [None for _ in prev_beta]


        # Run one step of the decoder to get the new info
        (topk_ids, topk_log_probs, new_states, attn_dists, p_gens, new_coverage, pre_attn_dists) = model.decode_onestep(sess=sess,
                                                                                                        batch=batch,
                                                                                                        latest_tokens=latest_tokens,
                                                                                                        enc_states=enc_states,
                                                                                                        dec_init_states=states,
                                                                                                        prev_coverage=prev_coverage,
                                                                                                        beta=prev_beta_for_words)

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
                                   beta=h.beta)
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

        if FLAGS.logan_reservoir:
            for hyp_idx, hyp in enumerate(hyps):
                if (hyp.latest_token == vocab.word2id(data.PERIOD) and FLAGS.logan_chunk_size == -1) or (    # if in regular mode, and the hyp ends in a period
                    FLAGS.logan_chunk_size > -1 and len(hyp.tokens) % FLAGS.logan_chunk_size == 0   # if in chunk mode, and hyp just finished a chunk
                ) or (not FLAGS.coverage_optimization):
                    summ_sents, summ_tokens = get_summ_sents_and_tokens(hyp.tokens, tokenizer, batch, vocab, FLAGS.logan_chunk_size)
                    summ_embeddings, summ_words_embs_list, summ_words_list = get_sentences_embeddings(summ_sents, summ_tokens,
                                                                                                      sess, batch, vocab)
                    summ_str = ' '.join([' '.join(sent) for sent in summ_sents])
                    similarity_amount = get_similarity_for_one_summ_sent(enc_sentences, enc_sent_embs, enc_words_embs_list, enc_tokens,
                                                    summ_embeddings, summ_words_embs_list, summ_tokens)
                    # plot_importances(enc_sentences_str, importances, summ_str)
                    logan_similarity[hyp_idx] = similarity_amount
                    a = 0
                    # if FLAGS.logan_coverage_tau != 1.0:
                    #     logan_coverage[hyp_idx] = softmax_trick(logan_coverage[hyp_idx], FLAGS.logan_coverage_tau)
                    hyp.beta = calc_beta_from_sim_and_imp(logan_similarity[hyp_idx], logan_importances, hyp.beta, batch, enc_tokens)
                    a=0
                # with open('log_optimization' + str(FLAGS.coverage_optimization), 'a') as f:
                #     np.savetxt(f, hyp.beta)
        if FLAGS.logan_coverage:
            for hyp_idx, hyp in enumerate(hyps):
                if (hyp.latest_token == vocab.word2id(data.PERIOD) and FLAGS.logan_chunk_size == -1) or (    # if in regular mode, and the hyp ends in a period
                    FLAGS.logan_chunk_size > -1 and len(hyp.tokens) % FLAGS.logan_chunk_size == 0   # if in chunk mode, and hyp just finished a chunk
                ) or (not FLAGS.coverage_optimization):
                    summ_sents, summ_tokens = get_summ_sents_and_tokens(hyp.tokens, tokenizer, batch, vocab, FLAGS.logan_chunk_size)
                    summ_embeddings, summ_words_embs_list, summ_words_list = get_sentences_embeddings(summ_sents, summ_tokens,
                                                                                                      sess, batch, vocab)
                    summ_str = ' '.join([' '.join(sent) for sent in summ_sents])
                    uncovered_amount, similarity_amount = get_coverage(enc_sentences, enc_sent_embs, enc_words_embs_list, enc_tokens,
                                                    summ_embeddings, summ_words_embs_list, summ_tokens)
                    # plot_importances(enc_sentences_str, importances, summ_str)
                    logan_coverage[hyp_idx] = uncovered_amount
                    a = 0
                    if FLAGS.logan_coverage_tau != 1.0:
                        logan_coverage[hyp_idx] = softmax_trick(logan_coverage[hyp_idx], FLAGS.logan_coverage_tau)

                    hyp.beta = calc_beta_from_cov_and_imp(logan_coverage[hyp_idx], logan_importances, batch, enc_tokens)
                # with open('log_optimization' + str(FLAGS.coverage_optimization), 'a') as f:
                #     np.savetxt(f, hyp.beta)
        if steps==13:
            a=0
        steps += 1

    # At this point, either we've got beam_size results, or we've reached maximum decoder steps

    if len(
            results) == 0:  # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
        results = hyps

    # Sort hypotheses by average log probability
    hyps_sorted = sort_hyps(results)
    best_hyp = hyps_sorted[0]

    if FLAGS.logan_save_distributions and ((FLAGS.logan_importance and FLAGS.logan_coverage) or FLAGS.logan_reservoir):
        save_importances_and_coverages(logan_importances, enc_sentences, enc_sent_embs, enc_words_embs_list,
                                   enc_tokens, best_hyp, sess, batch, vocab, tokenizer, ex_index)
    # if FLAGS.logan_importance and FLAGS.logan_reservoir:
    #     save_importances_and_similarities(logan_importances, enc_sentences, enc_sent_embs, enc_words_embs_list,
    #                                enc_tokens, best_hyp, sess, batch, vocab, tokenizer, ex_index)


    # Return the hypothesis with highest average log prob
    return best_hyp


def sort_hyps(hyps):
    """Return a list of Hypothesis objects, sorted by descending average log probability"""
    return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)
