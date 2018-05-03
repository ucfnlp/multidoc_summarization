import warnings

import numpy as np
import itertools
import util, data
from sklearn.metrics.pairwise import cosine_similarity


def get_importance_features_for_article(enc_states, enc_sentences, use_cluster_dist=False):

    # sent_indices = list(range(len(enc_sentences)))
    sent_lens = [len(sent) for sent in enc_sentences]
    sent_reps = get_sent_representations(enc_states, enc_sentences)
    # cluster_representation = get_fw_bw_rep(enc_states, 0, len(enc_states)-1)
    if use_cluster_dist:
        cluster_mean = np.mean(sent_reps, axis=0)
        cluster_representation = cosine_similarity(sent_reps, cluster_mean.reshape(1,-1))
        cluster_representation = np.squeeze(cluster_representation)
    else:
        cluster_representation = np.mean(sent_reps, axis=0)
    assert len(sent_lens) == len(sent_reps)
    # assert len(sent_indices) == len(sent_reps)
    return sent_lens, sent_reps, cluster_representation

def get_ROUGE_Ls(art_oovs, all_original_abstracts_sents, vocab, enc_tokens):
    human_tokens = get_tokens_for_human_summaries(art_oovs, all_original_abstracts_sents, vocab)  # list (of 4 human summaries) of list of token ids
    # human_sents, human_tokens = get_summ_sents_and_tokens(human_tokens, tokenizer, batch, vocab, FLAGS.chunk_size)
    similarity_matrix = util.Similarity_Functions.rouge_l_similarity(enc_tokens, human_tokens)
    importances_hat = np.sum(similarity_matrix, 1)
    logan_importances = special_squash(importances_hat)
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
    cur_sent_idx = 0
    for sent_idx, sent in enumerate(all_enc_sentences):
        sent_to_add = []
        tokens_to_add = []
        should_break = False
        tokens = sent.split(' ')
        if cur_doc_idx != doc_indices[count]:
            cur_doc_idx = doc_indices[count]
            cur_sent_idx = 0
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
    sent_representations = get_fw_bw_sent_reps(enc_states, sent_positions)

    return sent_representations

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