
import numpy as np
import struct
from tensorflow.core.example import example_pb2
import os
import glob
import tensorflow as tf
import write_data
from absl import flags
from absl import app
import cPickle
from util import create_dirs
import shutil



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

def save_to_file(text, file_name):
    with open(file_name, 'wb') as f:
        f.write(text)


unprocessed_path = '/home/logan/data/multidoc_summarization/results_unprocessed'
out_path = '/home/logan/data/multidoc_summarization/FINAL_RESULTS'
data_dir = '/home/logan/data/multidoc_summarization/tf_examples'

def main(unused_argv):

    datasets = ['duc_2004', 'tac_2011']

    for dataset in datasets:
        create_dirs(os.path.join(out_path, dataset))
        source_dir = os.path.join(data_dir, dataset)
        source_files = sorted(glob.glob(source_dir + '/*'))
        for file_idx in range(len(source_files)):
            example = get_tf_example(source_files[file_idx])
            raw_article_sents = example.features.feature['raw_article_sents'].bytes_list.value
            out_text = '\n'.join(raw_article_sents)
            example_path = os.path.join(out_path, dataset, 'example_%03d'%file_idx)
            create_dirs(example_path)
            save_to_file(out_text, os.path.join(example_path, 'source.txt'))

        exp_paths = glob.glob(os.path.join(unprocessed_path, dataset, '*'))
        for exp_path in exp_paths:
            exp_name = os.path.basename(exp_path)
            print 'Copying files from %s' % exp_name
            summary_files = sorted(glob.glob(exp_path + '/*'))
            for file_idx in range(len(summary_files)):
                example_path = os.path.join(out_path, dataset, 'example_%03d'%file_idx)
                shutil.copyfile(summary_files[file_idx], os.path.join(example_path, exp_name + '.txt'))





if __name__ == '__main__':
    app.run(main)





















