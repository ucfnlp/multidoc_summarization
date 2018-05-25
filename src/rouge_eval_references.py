import os

import struct

from tensorflow.core.example import example_pb2
from tqdm import tqdm
import pyrouge
from absl import logging
import nltk
import glob
import logging as log
from absl import flags
from absl import app
import shutil
import util
import data

data_dir = '/home/logan/data/multidoc_summarization/tf_examples'
log_dir = '/home/logan/data/multidoc_summarization/logs/'
max_enc_steps = 100000
min_dec_steps = 100
max_dec_steps = 120


FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'tac_2011', 'Which dataset to use. Can be {duc_2004, tac_2011, etc}')


def make_html_safe(s):
    """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s

def write_for_rouge(all_reference_sents, decoded_sents, ex_index, ref_dir, dec_dir):
    """Write output to file in correct format for eval with pyrouge. This is called in single_pass mode.

    Args:
        all_reference_sents: list of list of strings
        decoded_sents: list of strings
        ex_index: int, the index with which to label the files
    """

    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    decoded_sents = [make_html_safe(w) for w in decoded_sents]
    all_reference_sents = [[make_html_safe(w) for w in abstract] for abstract in all_reference_sents]

    # Write to file
    decoded_file = os.path.join(dec_dir, "%06d_decoded.txt" % ex_index)

    for abs_idx, abs in enumerate(all_reference_sents):
        ref_file = os.path.join(ref_dir, "%06d_reference.%s.txt" % (
            ex_index, chr(ord('A') + abs_idx)))
        with open(ref_file, "w") as f:
            for idx, sent in enumerate(abs):
                f.write(sent + "\n")
    with open(decoded_file, "w") as f:
        for idx, sent in enumerate(decoded_sents):
            f.write(sent + "\n")

def rouge_eval(ref_dir, dec_dir):
    """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
    r = pyrouge.Rouge155()
#   r.model_filename_pattern = '#ID#_reference.txt'
    r.model_filename_pattern = '#ID#_reference.[A-Z].txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir
    log.getLogger('global').setLevel(log.WARNING) # silence pyrouge logging
    rouge_args = ['-e', '/home/logan/ROUGE/RELEASE-1.5.5/data',
         '-c',
         '95',
         '-2', '4',        # This is the only one we changed (changed the max skip from -1 to 4)
         '-U',
         '-r', '1000',
         '-n', '4',
         '-w', '1.2',
         '-a',
         '-l', '100']
    rouge_args = ' '.join(rouge_args)
    rouge_results = r.convert_and_evaluate(rouge_args=rouge_args)
    return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write):
    """Log ROUGE results to screen and write to file.

    Args:
        results_dict: the dictionary returned by pyrouge
        dir_to_write: the directory where we will write the results to"""
    log_str = ""
    for x in ["1","2","l","s4","su4"]:
        log_str += "\nROUGE-%s:\n" % x
        for y in ["f_score", "recall", "precision"]:
            key = "rouge_%s_%s" % (x,y)
            key_cb = key + "_cb"
            key_ce = key + "_ce"
            val = results_dict[key]
            val_cb = results_dict[key_cb]
            val_ce = results_dict[key_ce]
            log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
    print(log_str) # log to screen
    results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
    print("Writing final ROUGE results to %s...", results_file)
    with open(results_file, "w") as f:
        f.write(log_str)

    print "\nROUGE-1, ROUGE-2, ROUGE-SU4 (PRF):\n"
    sheets_str = ""
    for x in ["1", "2", "su4"]:
        for y in ["precision", "recall", "f_score"]:
            key = "rouge_%s_%s" % (x, y)
            val = results_dict[key]
            sheets_str += "%.4f\t" % (val)
    sheets_str += "\n"
    print sheets_str
    sheets_results_file = os.path.join(dir_to_write, "sheets_results.txt")
    with open(sheets_results_file, "w") as f:
        f.write(sheets_str)

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
    tokens = [t for t in tokens if t != '<s>' and t != '</s>']
    return tokens

def split_into_sent_tokens(text):
    sent_tokens = [[t for t in tokens.strip().split() if t != '<s>' and t != '</s>'] for tokens in text.strip().split('\n')]
    return sent_tokens




def main(unused_argv):

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    source_dir = os.path.join(data_dir, FLAGS.dataset)
    source_files = sorted(glob.glob(source_dir + '/*'))

    for i in range(4):
        ref_dir = os.path.join(log_dir, 'reference_' + str(i), 'reference')
        dec_dir = os.path.join(log_dir, 'reference_' + str(i), 'decoded')
        util.create_dirs(ref_dir)
        util.create_dirs(dec_dir)
        for source_idx, source_file in enumerate(source_files):
            human_summary_texts = get_human_summary_texts(source_file)
            summaries = []
            for summary_text in human_summary_texts:
                summary = data.abstract2sents(summary_text)
                summaries.append(summary)
            candidate = summaries[i]
            references = [summaries[idx] for idx in range(len(summaries)) if idx != i]
            write_for_rouge(references, candidate, source_idx, ref_dir, dec_dir)

        results_dict = rouge_eval(ref_dir, dec_dir)
        # print("Results_dict: ", results_dict)
        rouge_log(results_dict, os.path.join(log_dir, 'reference_' + str(i)))




if __name__ == '__main__':
    app.run(main)



























