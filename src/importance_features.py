import warnings

import numpy as np
import itertools
import util, data
from sklearn.metrics.pairwise import cosine_similarity
from lex_rank_importance import LexRankSummarizer
import tensorflow as tf
import batcher

FLAGS = tf.app.flags.FLAGS

class SentRep:
    def __init__(self, abs_sent_indices, rel_sent_indices_normalized, rel_sent_indices_0_to_10, sent_lens, sent_lens_normalized,
        sent_representations_average, sent_representations_fw_bw, sent_representations_separate, cluster_rep_sent_average, 
        cluster_rep_sent_fw_bw, cluster_rep_sent_separate, lexrank_score):
        self.abs_sent_indices = abs_sent_indices
        self.rel_sent_indices_normalized = rel_sent_indices_normalized
        self.rel_sent_indices_0_to_10 = rel_sent_indices_0_to_10
        self.sent_lens = sent_lens
        self.sent_lens_normalized = sent_lens_normalized
        self.sent_representations_average = sent_representations_average
        self.sent_representations_fw_bw = sent_representations_fw_bw
        self.sent_representations_separate = sent_representations_separate
        self.cluster_rep_sent_average = cluster_rep_sent_average
        self.cluster_rep_sent_fw_bw = cluster_rep_sent_fw_bw
        self.cluster_rep_sent_separate = cluster_rep_sent_separate
        self.lexrank_score = lexrank_score
        self.y = None

def get_features_list(include_y):
    features = []
    if FLAGS.normalize_features:
        features.append('rel_sent_indices_normalized')
        features.append('sent_lens_normalized')
    else:
        features.append('abs_sent_indices')
        features.append('rel_sent_indices_0_to_10')
        features.append('sent_lens')
    if FLAGS.sent_vec_feature_method == 'average':
        features.append('sent_representations_average')
        features.append('cluster_rep_sent_average')
    elif FLAGS.sent_vec_feature_method == 'fw_bw':
        features.append('sent_representations_fw_bw')
        features.append('cluster_rep_sent_fw_bw')
    elif FLAGS.sent_vec_feature_method == 'separate':
        features.append('sent_representations_separate')
        features.append('cluster_rep_sent_separate')
    else:
        raise Exception('Flag sent_vec_feature_method (%s) must be one of {average, fw_bw, separate}' % FLAGS.sent_vec_feature_method)
    if FLAGS.lexrank_as_feature:
        features.append('lexrank_score')
    if include_y:
        features.append('y')

    return features

def get_importance_features_for_article(enc_states, enc_sentences, sent_indices, tokenizer, sent_representations_separate, use_cluster_dist=False):
    abs_sent_indices = sent_indices
    rel_sent_indices_normalized, rel_sent_indices_0_to_10 = get_relative_sent_indices(sent_indices)
    # sent_indices = list(range(len(enc_sentences)))
    sent_lens, sent_lens_normalized = get_sent_lens(enc_sentences)
    sent_representations_average, sent_representations_fw_bw = get_sent_representations(enc_states, enc_sentences)
    cluster_rep_sent_average, cluster_rep_sent_fw_bw, cluster_rep_sent_separate = get_cluster_representations(
        sent_representations_average, sent_representations_fw_bw, sent_representations_separate)
    summarizer = LexRankSummarizer()
    lexrank_score = summarizer.get_importances(enc_sentences, tokenizer)
    assert len(sent_lens) == len(sent_representations_average)
    # assert len(sent_indices) == len(sent_reps)

    sent_reps = []
    for i in range(len(abs_sent_indices)):
        sent_reps.append(SentRep(abs_sent_indices[i], rel_sent_indices_normalized[i], rel_sent_indices_0_to_10[i],
            sent_lens[i], sent_lens_normalized[i],
            sent_representations_average[i], sent_representations_fw_bw[i], sent_representations_separate[i],
            cluster_rep_sent_average, cluster_rep_sent_fw_bw, cluster_rep_sent_separate, lexrank_score[i]))
    return sent_reps

# def features_to_array(sent_indices, abs_sent_indices, rel_sent_indices, sent_lens, lexrank_score, sent_reps, cluster_rep):
#     if FLAGS.use_cluster_dist:
#         cluster_representations = np.expand_dims(cluster_rep, 1)
#     else:
#         cluster_representations = np.tile(cluster_rep, [len(sent_indices), 1])
#     feature_list = []
#     if not FLAGS.normalize_features:
#         feature_list.append(np.expand_dims(abs_sent_indices, 1))
#     feature_list.append(np.expand_dims(rel_sent_indices, 1))
#     feature_list.append(np.expand_dims(sent_lens, 1))
#     feature_list.append(sent_reps)
#     feature_list.append(cluster_representations)
#     if FLAGS.lexrank_as_feature:
#         feature_list.append(np.expand_dims(lexrank_score, 1))
#     x = np.concatenate(feature_list, 1)
#     return x
def features_to_array(sent_reps, features_list):
    x = []
    for rep in sent_reps:
        x_i = []
        for feature in features_list:
            val = getattr(rep, feature)
            if util.is_list_type(val):
                x_i.extend(val)
            else:
                x_i.append(val)
        x.append(x_i)
    return np.array(x)

def get_relative_sent_indices(sent_indices):
    relative_sent_indices = []
    prev_idx = -1
    cur_sent_indices = []
    for idx in sent_indices:
        if idx <= prev_idx:
            relative = [float(i)/len(cur_sent_indices) for i in cur_sent_indices]
            relative_sent_indices += relative
            prev_idx = -1
            cur_sent_indices = []
        cur_sent_indices.append(idx)
        prev_idx = idx
    if len(cur_sent_indices) > 0:
        relative = [float(i)/len(cur_sent_indices) for i in cur_sent_indices]
        relative_sent_indices += relative
    # if not FLAGS.normalize_features:
    relative_sent_indices_0_to_10 = [int(idx * 10) for idx in relative_sent_indices]
    return relative_sent_indices, relative_sent_indices_0_to_10


def get_sent_indices(enc_sentences, doc_indices):
    cur_doc_idx = 0
    cur_sent_idx = 1
    count = 0
    sent_indices = []
    for sent_idx, sent in enumerate(enc_sentences):
        if cur_doc_idx != doc_indices[count]:
            cur_doc_idx = doc_indices[count]
            cur_sent_idx = 1
        sent_indices.append(cur_sent_idx)
        for word_idx, word in enumerate(sent):
            count += 1
        cur_sent_idx += 1
    return sent_indices

def get_sent_lens(enc_sentences):
    sent_lens = [len(sent) for sent in enc_sentences]
    # if FLAGS.normalize_features:
    max_len = 50
    sent_lens_normalized = [min(1., float(length)/max_len) for length in sent_lens]
    return sent_lens, sent_lens_normalized


def get_ROUGE_Ls(art_oovs, all_original_abstracts_sents, vocab, enc_tokens):
    human_tokens = get_tokens_for_human_summaries(art_oovs, all_original_abstracts_sents, vocab)  # list (of 4 human summaries) of list of token ids
    # human_sents, human_tokens = get_summ_sents_and_tokens(human_tokens, tokenizer, batch, vocab, FLAGS.chunk_size)
    metric = 'recall' if FLAGS.rouge_l_prec_rec else 'f1'
    # similarity_matrix = util.Similarity_Functions.rouge_l_similarity(enc_tokens, human_tokens, metric=metric)
    # importances_hat = np.sum(similarity_matrix, 1)
    importances_hat = util.Similarity_Functions.rouge_l_similarity(enc_tokens, human_tokens, vocab, metric=metric)
    logan_importances = special_squash(importances_hat)
    # logan_importances = importances_hat
    return logan_importances

def get_enc_sents_and_tokens_with_cutoff_length(enc_batch_extend_vocab, tokenizer,
                                                art_oovs, vocab, doc_indices, skip_failures, chunk_size=-1, cutoff_len=1000000):
    art_oovs = [s.replace(' ', '_') for s in art_oovs]
    enc_text = tokens_to_continuous_text(enc_batch_extend_vocab, vocab, art_oovs)
    if chunk_size == -1:
        all_enc_sentences = tokenizer.to_sentences(enc_text)
        all_tokens = flatten_list_of_lists([sent.split(' ') for sent in all_enc_sentences])
    else:
        all_tokens = enc_text.split(' ')
        chunked_tokens = chunk_tokens(all_tokens, chunk_size)
        all_enc_sentences = [' '.join(chunk) for chunk in chunked_tokens]
    if len(all_tokens) != len(enc_batch_extend_vocab):
        if skip_failures:
            return None, None, None
        else:
            warnings.warn('All_tokens ('+str(len(all_tokens))+
                        ') does not have the same number of tokens as enc_batch ('+str(len(enc_batch_extend_vocab))+')')
    select_enc_sentences = []
    select_enc_tokens = []
    select_sent_indices = []
    count = 0
    cur_doc_idx = 0
    cur_sent_idx = 1
    for sent_idx, sent in enumerate(all_enc_sentences):
        sent_to_add = []
        tokens_to_add = []
        should_break = False
        tokens = sent.split(' ')
        if cur_doc_idx != doc_indices[count]:
            cur_doc_idx = doc_indices[count]
            cur_sent_idx = 1
        select_sent_indices.append(cur_sent_idx)
        for word_idx, word in enumerate(tokens):
            sent_to_add.append(word)
            tokens_to_add.append(enc_batch_extend_vocab[count])
            count += 1
            if count == cutoff_len:
                should_break = True
                break
        select_enc_sentences.append(sent_to_add)
        select_enc_tokens.append(tokens_to_add)
        cur_sent_idx += 1
        if should_break:
            break
    return select_enc_sentences, select_enc_tokens, select_sent_indices


def get_tokens_for_human_summaries(art_oovs, all_original_abstracts_sents, vocab):
    def get_all_summ_tokens(all_summs):
        return [get_summ_tokens(summ) for summ in all_summs]
    def get_summ_tokens(summ):
        summ_tokens = [get_sent_tokens(sent) for sent in summ]
        return list(itertools.chain.from_iterable(summ_tokens))     # combines all sentences into one list of tokens for summary
    def get_sent_tokens(sent):
        words = sent.split()
        return data.abstract2ids(words, vocab, art_oovs)
    human_summaries = all_original_abstracts_sents
    all_summ_tokens = get_all_summ_tokens(human_summaries)
    return all_summ_tokens

def special_squash(distribution):
    res = distribution - np.min(distribution)
    if np.max(res) == 0:
        print('All elements in distribution are 0, so setting all to 0')
        res.fill(0)
    else:
        res = res / np.max(res)
    return res

def get_sent_representations(enc_states, enc_sentences):
    sent_positions = get_sentence_splits(enc_sentences)
    # if FLAGS.average_over_word_states:
    sent_representations_average = get_average_word_enc_state(enc_states, sent_positions)
    # else:
    sent_representations_fw_bw = get_fw_bw_sent_reps(enc_states, sent_positions)

    return sent_representations_average, sent_representations_fw_bw

def get_cluster_representations(sent_representations_average, sent_representations_fw_bw, sent_representations_separate):
    # # cluster_representation = get_fw_bw_rep(enc_states, 0, len(enc_states)-1)
    # # if use_cluster_dist:
    # cluster_mean = np.mean(sent_reps, axis=0)
    # cluster_rep = cosine_similarity(sent_reps, cluster_mean.reshape(1,-1))
    # cluster_rep = np.squeeze(cluster_rep)
    # # else:
    cluster_rep_sent_average = np.mean(sent_representations_average, axis=0)
    cluster_rep_sent_fw_bw = np.mean(sent_representations_fw_bw, axis=0)
    cluster_rep_sent_separate = np.mean(sent_representations_separate, axis=0)
    return cluster_rep_sent_average, cluster_rep_sent_fw_bw, cluster_rep_sent_separate


def tokens_to_continuous_text(tokens, vocab, art_oovs):
    words = data.outputids2words(tokens, vocab, art_oovs)
    text = ' '.join(words)
    # text = text.decode('utf8')
    split_text = text.split(' ')
    if len(split_text) != len(words):
        for i in range(min(len(words), len(split_text))):
            try:
                print '%s\t%s'%(words[i], split_text[i])
            except:
                print 'FAIL\tFAIL'
        raise Exception('text ('+str(len(text.split()))+
                        ') does not have the same number of tokens as words ('+str(len(words))+')')

    return text

def chunk_tokens(tokens, chunk_size):
    chunk_size = max(1, chunk_size)
    return (tokens[i:i+chunk_size] for i in xrange(0, len(tokens), chunk_size))

def flatten_list_of_lists(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))

# def get_enc_sents(enc_batch_extend_vocab, tokenizer,
#                                                 art_oovs, vocab, chunk_size=-1):
#     art_oovs = [s.replace(' ', '_') for s in art_oovs]
#     enc_text = tokens_to_continuous_text(enc_batch_extend_vocab, vocab, art_oovs)
#     if chunk_size == -1:
#         all_enc_sentences = tokenizer.to_sentences(enc_text)
#         all_tokens = flatten_list_of_lists([sent.split(' ') for sent in all_enc_sentences])
#     else:
#         all_tokens = enc_text.split()
#         chunked_tokens = chunk_tokens(all_tokens, chunk_size)
#         all_enc_sentences = [' '.join(chunk) for chunk in chunked_tokens]
#     if len(all_tokens) != len(enc_batch_extend_vocab):
#         # print 'Art_oovs', art_oovs
#         warnings.warn('All_tokens ('+str(len(all_tokens))+
#                         ') does not have the same number of tokens as enc_batch ('+str(len(enc_batch_extend_vocab))+').' +
#                       'Skipping')
#         return None, None
#     tokenized_sentences = [sent.split(' ') for sent in all_enc_sentences]
#
#     enc_tokens = []
#     count = 0
#     for sent_idx, sent in enumerate(tokenized_sentences):
#         sent_to_add = []
#         tokens_to_add = []
#         for word_idx, word in enumerate(sent):
#             tokens_to_add.append(enc_batch_extend_vocab[count])
#             count += 1
#         enc_tokens.append(tokens_to_add)
#     return tokenized_sentences, enc_tokens

def get_sentence_splits(enc_sentences):
    '''Returns a list of indices, representing the word index for the first word of each sentence'''
    cur_idx = 0
    indices = []
    for sent in enc_sentences:
        indices.append(cur_idx)
        cur_idx += len(sent)
    return indices

def get_fw_bw_rep(enc_states, start_idx, end_idx):
    fw_state_size = enc_states.shape[1] / 2
    assert fw_state_size * 2 == enc_states.shape[1]
    fw_sent_rep = enc_states[end_idx, :fw_state_size]
    bw_sent_rep = enc_states[start_idx, fw_state_size:]
    rep = np.concatenate([fw_sent_rep, bw_sent_rep])
    return rep


def get_fw_bw_sent_reps(enc_states, sent_positions):
    reps = []
    for idx in range(len(sent_positions)):
        start_idx = sent_positions[idx]
        if idx+1 < len(sent_positions):
            end_idx = sent_positions[idx+1]-1
        else:
            end_idx = enc_states.shape[0]-1
        rep = get_fw_bw_rep(enc_states, start_idx, end_idx)
        reps.append(rep)
    reps = np.stack(reps)
    return reps

def get_average_word_enc_state(enc_states, sent_positions):
    reps = []
    for idx in range(len(sent_positions)):
        start_idx = sent_positions[idx]
        if idx+1 < len(sent_positions):
            end_idx = sent_positions[idx+1]
        else:
            end_idx = enc_states.shape[0]
        word_states = enc_states[start_idx:end_idx]
        rep = np.mean(word_states, axis=0)
        reps.append(rep)
    reps = np.stack(reps)
    return reps

def get_separate_enc_states(model, sess, enc_sentences, vocab, hps):
    reps = []
    examples = []
    for enc_sent in enc_sentences:
        sent_str = ' '.join(enc_sent)
        doc_indices = [0] * len(enc_sent)                   # just filler, shouldn't do anything
        ex = batcher.Example(sent_str, [], [[]], doc_indices, None, vocab, hps)
        examples.append(ex)
    chunks = util.chunks(examples, hps.batch_size)
    if len(chunks[-1]) != hps.batch_size:                   # If last chunk is not filled, then just artificially fill it
        for i in range(hps.batch_size - len(chunks[-1])):
            chunks[-1].append(examples[-1])
    for chunk in chunks:
        batch = batcher.Batch(chunk, hps, vocab)
        batch_enc_states, _ = model.run_encoder(sess, batch)
        for batch_idx, enc_states in enumerate(batch_enc_states):
            start_idx = 0
            end_idx = batch.enc_lens[batch_idx] - 1
            rep = get_fw_bw_rep(enc_states, start_idx, end_idx)
            reps.append(rep)
    reps = reps[:len(enc_sentences)]                        # Removes the filler examples
    return reps
