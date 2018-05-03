
import numpy as np
import struct
from tensorflow.core.example import example_pb2
import os
import glob
import tensorflow as tf


tac_source_dir = '/home/logan/data/multidoc_summarization/TAC_Data/full_article_tf_examples/test'
duc_source_dir = '/home/logan/data/multidoc_summarization/DUC/full_article_tf_examples/test'

tac_all_summary_dir = '/home/logan/data/multidoc_summarization/logs/tac_2011/decode_test_10000maxenc_4beam_35mindec_100maxdec_ckpt-238410/decoded'
tac_clustering_summary_dir = '/home/logan/data/multidoc_summarization/logs/tac_2011_clustering/decode_test_10000maxenc_4beam_35mindec_100maxdec_ckpt-238410/decoded'
duc_all_summary_dir = '/home/logan/data/multidoc_summarization/logs/duc_2004/decode_test_10000maxenc_4beam_35mindec_100maxdec_ckpt-238410/decoded'
duc_clustering_summary_dir = '/home/logan/data/multidoc_summarization/logs/duc_2004_clustering/decode_test_10000maxenc_4beam_35mindec_100maxdec_ckpt-238410/decoded'


FLAGS = tf.app.flags.FLAGS

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

data_dir = '/home/logan/data/multidoc_summarization/'
source_files_dir = '/full_article_tf_examples/test'
log_dir = '/home/logan/data/multidoc_summarization/logs/'
max_enc_steps = 100000
min_dec_steps = 100
max_dec_steps = 120
tac_folder = 'TAC_Data'
duc_folder = 'DUC'


rouge_logan_dec_dir = '/home/logan/data/multidoc_summarization/logs/scratch/decode_test_100000maxenc_4beam_70mindec_100maxdec_ckpt-238410/decoded'
rouge_orig_dec_dir = '/home/logan/data/multidoc_summarization/logs/tac_2011/decode_test_10000maxenc_4beam_35mindec_100maxdec_ckpt-238410/decoded'

tf.app.flags.DEFINE_string('exp_name', 'reference', 'Path to system-generated summaries that we want to evaluate.' +
                           ' If you want to run on human summaries, then enter "reference".')
tf.app.flags.DEFINE_string('dataset', 'tac', 'Which dataset to use. Can be {tac,duc}')

summary_dir = tac_all_summary_dir

def get_nGram(l, n = 2):
    l = list(l)
    return set(zip(*[l[i:] for i in range(n)]))

def get_tf_example(source_file):
    reader = open(source_file, 'rb')
    len_bytes = reader.read(8)
    if not len_bytes: return  # finished reading this file
    str_len = struct.unpack('q', len_bytes)[0]
    example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
    e = example_pb2.Example.FromString(example_str)
    return e

def get_article_text(source_file):
    e = get_tf_example(source_file)
    article_text = e.features.feature['article'].bytes_list.value[
        0].lower()  # the article text was saved under the key 'article' in the data files
    return article_text

def get_summary_text(summary_file, is_reference):
    with open(summary_file) as f:
        summary_text = f.read()
    return summary_text

def get_human_summary_texts(summary_file):
    summary_texts = []
    e = get_tf_example(summary_file)
    for abstract in e.features.feature['abstract'].bytes_list.value:
        summary_texts.append(abstract)  # the abstracts texts was saved under the key 'abstract' in the data files
    return summary_texts

def split_into_tokens(text):
    tokens = text.split()
    return tokens

def get_ngram_results(article_tokens, summary_tokens):
    ngram_results = []
    for ngram_order in ngram_orders:
        article_ngrams = get_nGram(article_tokens, n=ngram_order)
        summary_ngrams = get_nGram(summary_tokens, n=ngram_order)
        num_appear_in_source = sum([1 for ngram in list(summary_ngrams) if ngram in article_ngrams])
        percent_appear_in_source = num_appear_in_source * 1. / len(list(summary_ngrams))
        ngram_results.append(percent_appear_in_source)
    return ngram_results

ngram_orders = [1,2,3,4]

def main(unused_argv):

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)
    # original_dir = dec_dir.split('/')
    # original_dir[-1] = 'reference'
    # rouge_ref_dir = '/'.join(original_dir)

    if FLAGS.dataset == 'tac':
        source_dir = data_dir + tac_folder + source_files_dir
    elif FLAGS.dataset == 'duc':
        source_dir = data_dir + duc_folder + source_files_dir
    else:
        raise Exception('Bad value for FLAG: dataset. Must be one of {tac,duc}')
    if FLAGS.exp_name == 'reference':
        summary_dir = source_dir
        is_reference = True
    else:
        summary_dir = log_dir + FLAGS.exp_name + '/decode_test_' + str(max_enc_steps) + \
                        'maxenc_4beam_' + str(min_dec_steps) + 'mindec_' + str(max_dec_steps) + 'maxdec_ckpt-238410/decoded'
        is_reference = False
    print('Ngrams ', ngram_orders, ' appearing in source documents')


    ngram_percentages = []

    summary_files = sorted(glob.glob(summary_dir + '/*'))
    source_files = sorted(glob.glob(source_dir + '/*'))
    for file_idx in range(len(summary_files)):
        article_text = get_article_text(source_files[file_idx])
        article_tokens = split_into_tokens(article_text)

        if is_reference:
            human_summary_texts = get_human_summary_texts(summary_files[file_idx])

        else:
            summary_text = get_summary_text(summary_files[file_idx], is_reference)
            summary_tokens = split_into_tokens(summary_text)

            ngram_results = get_ngram_results(article_tokens, summary_tokens)
            ngram_percentages.append(ngram_results)
        a=0

    print 'Experiment name: ', FLAGS.exp_name
    for ngram_idx in range(len(ngram_orders)):
        average_percent = sum([tup[ngram_idx] for tup in ngram_percentages]) / len(ngram_percentages)
        print '%.4f\t' % average_percent,
    print '\n'
    a=0



if __name__ == '__main__':
    tf.app.run()






























