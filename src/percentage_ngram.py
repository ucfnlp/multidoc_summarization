
import numpy as np
import struct
from tensorflow.core.example import example_pb2
import os
import glob

tac_source_dir = '/home/logan/data/multidoc_summarization/TAC_Data/full_article_tf_examples/test'
duc_source_dir = '/home/logan/data/multidoc_summarization/DUC/full_article_tf_examples/test'

tac_all_summary_dir = '/home/logan/data/multidoc_summarization/logs/tac_2011/decode_test_10000maxenc_4beam_35mindec_100maxdec_ckpt-238410/decoded'
tac_clustering_summary_dir = '/home/logan/data/multidoc_summarization/logs/tac_2011_clustering/decode_test_10000maxenc_4beam_35mindec_100maxdec_ckpt-238410/decoded'
duc_all_summary_dir = '/home/logan/data/multidoc_summarization/logs/duc_2004/decode_test_10000maxenc_4beam_35mindec_100maxdec_ckpt-238410/decoded'
duc_clustering_summary_dir = '/home/logan/data/multidoc_summarization/logs/duc_2004_clustering/decode_test_10000maxenc_4beam_35mindec_100maxdec_ckpt-238410/decoded'


summary_dir = tac_all_summary_dir

def get_nGram(l, n = 2):
    l = list(l)
    return set(zip(*[l[i:] for i in range(n)]))

def get_article_text(source_file):
    reader = open(source_file, 'rb')
    len_bytes = reader.read(8)
    if not len_bytes: return  # finished reading this file
    str_len = struct.unpack('q', len_bytes)[0]
    example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
    e = example_pb2.Example.FromString(example_str)
    article_text = e.features.feature['article'].bytes_list.value[
        0].lower()  # the article text was saved under the key 'article' in the data files
    return article_text

def get_summary_text(summary_file):
    with open(summary_file) as f:
        summary_text = f.read()
    return summary_text

def split_into_tokens(text):
    tokens = text.split()
    return tokens

ngram_orders = [1,2,3]

print('Ngrams ', ngram_orders, ' appearing in source documents')

for summ_method in ['tac_all', 'tac_clustering', 'duc_all', 'duc_clustering']:
    if 'tac' in summ_method:
        source_dir = tac_source_dir
        if 'clustering' in summ_method:
            summary_dir = tac_clustering_summary_dir
        else:
            summary_dir = tac_all_summary_dir

    else:
        source_dir = duc_source_dir
        if 'clustering' in summ_method:
            summary_dir = duc_clustering_summary_dir
        else:
            summary_dir = duc_all_summary_dir


    ngram_percentages = []

    summary_files = sorted(glob.glob(summary_dir + '/*'))
    source_files = sorted(glob.glob(source_dir + '/*'))
    for file_idx in range(len(summary_files)):
        article_text = get_article_text(source_files[file_idx])
        article_tokens = split_into_tokens(article_text)

        summary_text = get_summary_text(summary_files[file_idx])
        summary_tokens = split_into_tokens(summary_text)

        ngram_results = []
        for ngram_order in ngram_orders:
            article_ngrams = get_nGram(article_tokens, n=ngram_order)
            summary_ngrams = get_nGram(summary_tokens, n=ngram_order)
            num_appear_in_source = sum([1 for ngram in list(summary_ngrams) if ngram in article_ngrams])
            percent_appear_in_source = num_appear_in_source * 1. / len(list(summary_ngrams))
            ngram_results.append(percent_appear_in_source)
        ngram_percentages.append(ngram_results)
        a=0

    print summ_method
    for ngram_idx in range(len(ngram_orders)):
        average_percent = sum([tup[ngram_idx] for tup in ngram_percentages]) / len(ngram_percentages)
        print '%.4f\t' % average_percent,
    print '\n'
a=0





























