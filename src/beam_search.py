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

import data
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
import textwrap as tw
from nltk.corpus import stopwords
import string
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from lex_rank_importance import LexRankSummarizer

FLAGS = tf.app.flags.FLAGS


class Hypothesis(object):
    """Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis."""

    def __init__(self, tokens, log_probs, state, attn_dists, p_gens, coverage):
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

    def extend(self, token, log_prob, state, attn_dist, p_gen, coverage):
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
                          coverage=coverage)

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

def plot_importances(article_sents, importances, abstracts_text):

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
    plt.show()

def get_enc_sents_and_tokens_with_cutoff_length(original_article, enc_batch, cutoff_len, tokenizer):
    all_enc_sentences = tokenizer.to_sentences(original_article)
    select_enc_sentences = []
    select_enc_tokens = []
    count = 0
    for sent_idx, sent in enumerate(all_enc_sentences):
        sent_to_add = []
        tokens_to_add = []
        should_break = False
        tokens = tokenizer._get_word_tokenizer('english').tokenize(sent)
        for word_idx, word in enumerate(tokens):
            sent_to_add.append(word)
            tokens_to_add.append(enc_batch[count])
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


def get_summ_sents_and_tokens(summ_tokens, tokenizer, batch, vocab):
    summ_str = ''
    for token in summ_tokens:
        if token >= vocab.size():
            try:
                word = batch.art_oovs[0][token - vocab.size()]
            except:
                print('Problem with token ', token, 'and oov\'s', batch.art_oovs[0])
                raise Exception('')

        else:
            word = vocab.id2word(token)
        summ_str += word + ' '
    sentences = tokenizer.to_sentences(summ_str)
    if data.PERIOD not in sentences[-1]:
        sentences = sentences[:len(sentences) - 1]  # Doesn't include the last sentence if incomplete (no period)
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


def get_nGram(l, n=2):
    l = list(l)
    return list(set(zip(*[l[i:] for i in range(n)])))


def ngram_similarity(article_sents, abstract_sents):
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


def tokenwise_sentence_similarity(enc_words_embs_list, summ_words_embs_list):
    sentence_similarity_matrix = np.zeros([len(enc_words_embs_list), len(summ_words_embs_list)], dtype=float)
    for enc_sent_idx, enc_sent in enumerate(enc_words_embs_list):
        for summ_sent_idx, summ_sent in enumerate(summ_words_embs_list):
            similarity_matrix = cosine_similarity(enc_sent, summ_sent)
            # if enc_sent_idx == 1 and summ_sent_idx == 1:
            #     print similarity_matrix
            pair_similarity = np.sum(similarity_matrix) / np.size(similarity_matrix)
            pair_similarity = max(0, pair_similarity)  # don't allow negative values
            sentence_similarity_matrix[enc_sent_idx, summ_sent_idx] = pair_similarity
    return sentence_similarity_matrix

def get_coverage(enc_sentences, enc_sent_embs, enc_words_embs_list, enc_tokens, summ_embeddings, summ_words_embs_list, summ_tokens):
        if len(summ_embeddings) == 0:
            return np.ones([len(enc_sentences)], dtype=float) / len(enc_sentences)
        else:
            # Calculate similarity matrix [num_sentences_encoder, num_sentences_summary]
            similarity_fn = tokenwise_sentence_similarity
            normalization_fn = l1_normalize
            if similarity_fn == cosine_similarity:
                similarity_matrix = similarity_fn(enc_sent_embs, summ_embeddings)
                # Calculate amount of uncovered information for each sentence in the source
                importances_hat = np.sum(similarity_matrix, 1)
            elif similarity_fn == tokenwise_sentence_similarity:
                similarity_matrix = tokenwise_sentence_similarity(enc_words_embs_list, summ_words_embs_list)
                # Calculate amount of uncovered information for each sentence in the source
                importances_hat = np.sum(similarity_matrix, 1)
            elif similarity_fn == ngram_similarity:
                importances_hat = ngram_similarity(enc_tokens, summ_tokens)
            importances = normalization_fn(importances_hat)
            uncovered_amount_hat = max(importances_hat) - importances_hat
            uncovered_amount = normalization_fn(uncovered_amount_hat)
            return uncovered_amount


def combine_coverage_and_importance(logan_coverage, logan_importances):
    repeated_logan_importances = np.tile(logan_importances, (len(logan_coverage), 1))  # repeat over hypotheses
    beta_for_sentences = logan_coverage + repeated_logan_importances
    return beta_for_sentences


def convert_to_word_level(beta_for_sentences, hyps, batch, enc_tokens):
    beta = np.ones([len(hyps), len(batch.enc_batch[0])], dtype=float) / len(batch.enc_batch[0])
    # Calculate how much for each word in source
    for hyp_idx in range(len(hyps)):
        word_idx = 0
        for sent_idx in range(len(enc_tokens)):
            beta_for_words = np.full([len(enc_tokens[sent_idx])], beta_for_sentences[hyp_idx, sent_idx])
            beta[hyp_idx, word_idx:word_idx + len(beta_for_words)] = beta_for_words
            word_idx += len(beta_for_words)
    return beta


def l1_normalize(importances):
    return importances / np.sum(importances)

def softmax_trick(distribution, tau):
    return softmax(distribution / tau)


def run_beam_search(sess, model, vocab, batch, specific_max_dec_steps=None):
    """Performs beam search decoding on the given example.

    Args:
        sess: a tf.Session
        model: a seq2seq model
        vocab: Vocabulary object
        batch: Batch object that is the same example repeated across the batch

    Returns:
        best_hyp: Hypothesis object; the best hypothesis found by beam search.
    """

    # Use UPitt's max_dec_steps if it is specified, otherwise, use default max
    max_dec_steps = specific_max_dec_steps if specific_max_dec_steps is not None else FLAGS.max_dec_steps

    # Run the encoder to get the encoder hidden states and decoder initial state
    enc_states, dec_in_state = model.run_encoder(sess, batch)
    # dec_in_state is a LSTMStateTuple
    # enc_states has shape [batch_size, <=max_enc_steps, 2*hidden_dim].

    # Initialize beam_size-many hyptheses
    hyps = [Hypothesis(tokens=[vocab.word2id(data.START_DECODING)],
                       log_probs=[0.0],
                       state=dec_in_state,
                       attn_dists=[],
                       p_gens=[],
                       coverage=np.zeros([batch.enc_batch.shape[1]])  # zero vector of length attention_length
                       ) for _ in xrange(FLAGS.beam_size)]
    results = []  # this will contain finished hypotheses (those that have emitted the [STOP] token)




    if FLAGS.logan_beta:
        tokenizer = Tokenizer('english')

        enc_sentences, enc_tokens = get_enc_sents_and_tokens_with_cutoff_length(
                        batch.original_articles[0], batch.enc_batch[0], len(batch.enc_batch[0]), tokenizer)
        enc_sent_embs, enc_words_embs_list, enc_words_list = get_sentences_embeddings(enc_sentences, enc_tokens,
                                                                                                sess, batch, vocab)
        enc_sentences_str = [' '.join(sent) for sent in enc_sentences]



    if FLAGS.logan_coverage:
        # Initial coverage (evenly distributed among all sentences)
        logan_coverage = np.ones([len(hyps), len(enc_sentences)], dtype=float) / len(enc_sentences)
    else:
        logan_coverage = None

    if FLAGS.logan_importance:
        # parser = PlaintextParser.from_string(batch.original_articles[0], tokenizer)
        summarizer = LexRankSummarizer()
        logan_importances = summarizer.get_importances(enc_sentences, tokenizer)
        # plot_importances(enc_sentences_str, logan_importances, 'n/a')
        if FLAGS.logan_importance_tau != 1.0:
            logan_importances = softmax_trick(logan_importances, FLAGS.logan_importance_tau)


    steps = 0
    while steps < max_dec_steps and len(results) < FLAGS.beam_size:
        def ids_to_words(ids):
            if len(ids) > 0 and type(ids[0]) == list:
                return [data.outputids2words(sent_ids, vocab, batch.art_oovs[0]) for sent_ids in ids]
            else:
                return data.outputids2words(ids, vocab, batch.art_oovs[0])

        latest_tokens = [h.latest_token for h in hyps]  # latest token produced by each hypothesis
        latest_tokens = [t if t in xrange(vocab.size()) else vocab.word2id(data.UNKNOWN_TOKEN) for t in
                         latest_tokens]  # change any in-article temporary OOV ids to [UNK] id, so that we can lookup word embeddings

        if FLAGS.logan_coverage:
            for hyp_idx, hyp in enumerate(hyps):
                summ_sents, summ_tokens = get_summ_sents_and_tokens(hyp.tokens, tokenizer, batch, vocab)
                summ_embeddings, summ_words_embs_list, summ_words_list = get_sentences_embeddings(summ_sents, summ_tokens,
                                                                                                  sess, batch, vocab)
                summ_str = ' '.join([' '.join(sent) for sent in summ_sents])
                uncovered_amount = get_coverage(enc_sentences, enc_sent_embs, enc_words_embs_list, enc_tokens,
                                                summ_embeddings, summ_words_embs_list, summ_tokens)
                # plot_importances(enc_sentences_str, importances, summ_str)
                logan_coverage[hyp_idx] = uncovered_amount
                a = 0
            if FLAGS.logan_coverage_tau != 1.0:
                logan_coverage = softmax_trick(logan_coverage, FLAGS.logan_coverage_tau)


        if FLAGS.logan_coverage and FLAGS.logan_importance:     # if both sentence-level options are on
            beta_for_sentences = combine_coverage_and_importance(logan_coverage, logan_importances)
        elif FLAGS.logan_coverage:
            beta_for_sentences = logan_coverage     # if only sentence-level coverage is on
        elif FLAGS.logan_importance:
            beta_for_sentences = np.tile(logan_importances, (len(hyps), 1))     # if only sentence-level importance is on
        else:
            beta_for_sentences = None     # Don't use beta if it's neither sentence-level option is used

        if beta_for_sentences is not None:
            beta = convert_to_word_level(beta_for_sentences, hyps, batch, enc_tokens)
            if FLAGS.logan_beta_tau != 1.0:
                beta = softmax_trick(beta, FLAGS.logan_beta_tau)
        else:
            beta = None






        states = [h.state for h in hyps]  # list of current decoder states of the hypotheses
        prev_coverage = [h.coverage for h in hyps]  # list of coverage vectors (or None)

        # Run one step of the decoder to get the new info
        (topk_ids, topk_log_probs, new_states, attn_dists, p_gens, new_coverage, pre_attn_dists) = model.decode_onestep(sess=sess,
                                                                                                        batch=batch,
                                                                                                        latest_tokens=latest_tokens,
                                                                                                        enc_states=enc_states,
                                                                                                        dec_init_states=states,
                                                                                                        prev_coverage=prev_coverage,
                                                                                                        beta=beta)

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
                                   coverage=new_coverage_i)
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

        steps += 1

    # At this point, either we've got beam_size results, or we've reached maximum decoder steps

    if len(
            results) == 0:  # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
        results = hyps

    # Sort hypotheses by average log probability
    hyps_sorted = sort_hyps(results)
    best_hyp = hyps_sorted[0]

    best_hyp_words = ids_to_words(best_hyp.tokens)

    # Return the hypothesis with highest average log prob
    return best_hyp


def sort_hyps(hyps):
    """Return a list of Hypothesis objects, sorted by descending average log probability"""
    return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)
