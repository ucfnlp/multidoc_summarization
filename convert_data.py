# -*- coding: utf8 -*-

"""
Script to convert multi-document inputs to TensorFlow examples which can be sent to the PG-MMR model.
"""

import glob
import struct
import shutil
from tensorflow.core.example import example_pb2
import nltk
import os
from bs4 import BeautifulSoup
import io
from absl import flags
from absl import app
import sys
reload(sys)
sys.setdefaultencoding('utf8')

FLAGS = flags.FLAGS

p_start_tag = '<P>'
p_end_tag = '</P>'

def fix_bracket_token(token):
    if token == '(':
        return '-lrb-'
    elif token == ')':
        return '-rrb-'
    elif token == '[':
        return '-lsb-'
    elif token == ']':
        return '-rsb-'
    else:
        return token

def is_quote(tokens):
    contains_quotation_marks = "''" in tokens and len(tokens) > 0 and tokens[0] == "``"
    doesnt_end_with_period = len(tokens) > 0 and tokens[-1] != "."
    # contains_says = "says" in tokens or "said" in tokens
    decision = contains_quotation_marks or doesnt_end_with_period
    if decision:
        print "Skipping quote: ", ' '.join(tokens)
    return decision

def process_sent(sent):
    line = sent.lower()
    tokenized_sent = nltk.word_tokenize(line)
    tokenized_sent = [fix_bracket_token(token) for token in tokenized_sent]
    return tokenized_sent

def process_dataset(dataset_name, out_data_path, TAC_path='', DUC_path='', custom_dataset_path=''):
    data_dirs = {
        'tac_2011': {
            'article_dir': os.path.join(TAC_path, 'summary_data/s11/test_doc_files'),
            'abstract_dir': os.path.join(TAC_path, 'summary_data/s11/models')
        },
        'tac_2010': {
            'article_dir': os.path.join(TAC_path, 'summary_data/s10/test_doc_files'),
            'abstract_dir': os.path.join(TAC_path, 'summary_data/s10/models')
        },
        'tac_2008': {
            'article_dir': os.path.join(TAC_path, 'summary_data/s08/test_doc_files'),
            'abstract_dir': os.path.join(TAC_path, 'summary_data/s08/models')
        },
        'duc_2004': {
            'article_dir': os.path.join(DUC_path, 'Original/DUC2004_Summarization_Documents/duc2004_testdata/tasks1and2/duc2004_tasks1and2_docs/docs'),
            'abstract_dir': os.path.join(DUC_path, 'past_duc/duc2004/duc2004_results/ROUGE/eval/models/2')
        },
        'duc_2003': {
            'article_dir': os.path.join(DUC_path, 'Original/DUC2003_Summarization_Documents/duc2003_testdata/task2/docs'),
            'abstract_dir': os.path.join(DUC_path, 'past_duc/duc2003/results/detagged.duc2003.abstracts/models')
        }
    }
    if dataset_name == 'duc_tac':
        combine_duc_2003_tac_2008_tac_2010(out_data_path)
        return
    if dataset_name in data_dirs:
        article_dir = data_dirs[dataset_name]['article_dir']
        abstract_dir = data_dirs[dataset_name]['abstract_dir']
        is_tac = 'tac' in dataset_name
        is_custom_dataset = False
    else:
        article_dir = custom_dataset_path
        abstract_dir = custom_dataset_path
        is_tac = False
        is_custom_dataset = True
    out_dir = os.path.join(out_data_path, dataset_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    multidoc_dirnames = sorted(os.listdir(article_dir))
    out_idx = 1
    for multidoc_dirname in multidoc_dirnames:
        article, abstracts, doc_indices, raw_article_sents = get_article_abstract(multidoc_dirname, article_dir, abstract_dir, is_tac, is_custom_dataset)
        with open(os.path.join(out_dir, 'test_{:03d}.bin'.format(out_idx)), 'wb') as writer:
            write_example(article, abstracts, doc_indices, raw_article_sents, writer)
        out_idx += 1

def combine_duc_2003_tac_2008_tac_2010(out_data_path):
    out_dir = os.path.join(out_data_path, 'duc_tac')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_idx = 1
    for dataset_name in ['duc_2003', 'tac_2008', 'tac_2010']:
        example_dir = os.path.join(out_data_path, dataset_name)
        if not os.path.exists(example_dir) or len(os.listdir(example_dir)) == 0:
            process_dataset(dataset_name)
        example_files = glob.glob(os.path.join(example_dir,'*'))
        for old_file in example_files:
            new_file = os.path.join(out_dir, 'test_{:03d}.bin'.format(out_idx))
            shutil.copyfile(old_file, new_file)
            out_idx += 1

def concatenate_p_tags(soup):
    lines = []
    for tag in soup.findAll('p'):
        lines.append(tag.renderContents().replace('\n', ' ').strip())
    contents = ' '.join(lines)
    return contents

def has_p_tags(soup):
    return soup.find('p')

def fix_exceptions(sentences):
    new_sentences = []
    for sent in sentences:
        split_sents = sent.split('Hun Sen. ')
        if len(split_sents) == 2:
            sent1 = split_sents[0] + 'Hun Sen.'
            sent2 = split_sents[1].strip()
            new_sentences.append(sent1)
            if sent2 != '':
                new_sentences.append(sent2)
        else:
            new_sentences.append(sent)
    return new_sentences

def add_sents_to_article(sentences, article, raw_article_sents, doc_indices, doc_idx):
    for orig_sent in sentences:
        tokenized_sent = process_sent(orig_sent)
        if is_quote(tokenized_sent):
            continue
        sent = ' '.join(tokenized_sent)
        article += sent + ' '

        doc_indices_for_tokens = [doc_idx] * len(tokenized_sent)
        doc_indices_str = ' '.join(str(x) for x in doc_indices_for_tokens)
        doc_indices += doc_indices_str + ' '
        raw_article_sents.append(orig_sent)
    return article, raw_article_sents, doc_indices

def get_article(article_dir, multidoc_dirname, is_tac):
    if is_tac:
        multidoc_dir = os.path.join(article_dir, multidoc_dirname, multidoc_dirname + '-A')
    else:
        multidoc_dir = os.path.join(article_dir, multidoc_dirname)

    doc_names = os.listdir(multidoc_dir)
    doc_names = sorted([f for f in doc_names if os.path.isfile(os.path.join(multidoc_dir, f)) and '.py' not in f])
    article = ''
    doc_indices = ''
    raw_article_sents = []
    for doc_idx, doc_name in enumerate(doc_names):
        doc_path = os.path.join(multidoc_dir, doc_name)
        with open(doc_path) as f:
            article_text = f.read()
        soup = BeautifulSoup(article_text, 'html.parser')
        if is_tac:
            contents = concatenate_p_tags(soup)
            sentences = nltk.tokenize.sent_tokenize(contents)
            article, raw_article_sents, doc_indices = add_sents_to_article(sentences, article, raw_article_sents, doc_indices, doc_idx)
        else:
            if has_p_tags(soup):
                contents = concatenate_p_tags(soup)
            else:
                contents = soup.findAll('text')[0].renderContents().replace('\n', ' ').strip()
                contents = ' '.join(contents.split())
            sentences = nltk.tokenize.sent_tokenize(contents)
            fixed_sentences = fix_exceptions(sentences)
            article, raw_article_sents, doc_indices = add_sents_to_article(sentences, article, raw_article_sents, doc_indices, doc_idx)
    article = article.encode('utf-8').strip()
    return article, doc_indices, raw_article_sents

def process_abstract(abstract_lines):
    abstract = ''
    for line in abstract_lines:
        line = line.lower()
        line = line.replace(u'\x92', "'")
        tokenized_sent = nltk.word_tokenize(line)
        tokenized_sent = [fix_bracket_token(token) for token in tokenized_sent]
        sent = ' '.join(tokenized_sent)
        abstract += '<s> ' + sent + ' </s> '
    abstract = abstract.encode('utf-8').strip()
    abstract = abstract.strip()
    return abstract

def get_abstract(multidoc_dirname, abstract_dir, is_tac):
    abstracts = []
    doc_num = ''.join([s for s in multidoc_dirname if s.isdigit()])
    all_doc_names = os.listdir(abstract_dir)
    if is_tac:
        abstract_doc_name = 'D' + doc_num + '-A'
    else:
        abstract_doc_name = 'D' + doc_num + '.M'
    selected_doc_names = [doc_name for doc_name in all_doc_names if abstract_doc_name in doc_name]
    if len(selected_doc_names) == 0:
        raise Exception('no docs found for doc ' + doc_num)
    for selected_doc_name in selected_doc_names:
        with io.open(os.path.join(abstract_dir, selected_doc_name), encoding='utf-8', errors='ignore') as f:
            abstract_lines = f.readlines()
        abstract = process_abstract(abstract_lines)
        abstracts.append(abstract)
    return abstracts

def get_article_abstract(multidoc_dirname, article_dir, abstract_dir, is_tac, is_custom_dataset):
    if is_custom_dataset:
        article, abstracts, doc_indices, raw_article_sents = get_custom_article_abstract(multidoc_dirname, article_dir)
    else:
        article, doc_indices, raw_article_sents = get_article(article_dir, multidoc_dirname, is_tac, is_single_doc=False)
        abstracts = get_abstract(multidoc_dirname, abstract_dir, is_tac)
    return article, abstracts, doc_indices, raw_article_sents


def write_example(article, abstracts, doc_indices, raw_article_sents, writer):
    tf_example = example_pb2.Example()
    tf_example.features.feature['article'].bytes_list.value.extend([article])
    for abstract in abstracts:
        tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
    if doc_indices is not None:
        tf_example.features.feature['doc_indices'].bytes_list.value.extend([doc_indices])
    for sent in raw_article_sents:
        tf_example.features.feature['raw_article_sents'].bytes_list.value.extend([sent])
    tf_example_str = tf_example.SerializeToString()
    str_len = len(tf_example_str)
    writer.write(struct.pack('q', str_len))
    writer.write(struct.pack('%ds' % str_len, tf_example_str))

def get_custom_article_abstract(multidoc_dirname, article_dir):
    with open(os.path.join(article_dir, multidoc_dirname)) as f:
        text = f.read()
    docs = [[sent.strip() for sent in doc.strip().split('\n')] for doc in text.split('<SUMMARIES>')[0].strip().split('\n\n')]
    article = ''
    doc_indices = ''
    raw_article_sents = []
    for doc_idx, sentences in enumerate(docs):
        article, raw_article_sents, doc_indices = add_sents_to_article(sentences, article, raw_article_sents, doc_indices, doc_idx)
    article = article.encode('utf-8').strip()
    if '<SUMMARIES>' in text:
        abstracts_unprocessed = [[sent.strip() for sent in abs.strip().split('\n')] for abs in text.split('<SUMMARIES>')[1].strip().split('\n\n')]
        abstracts = []
        for abstract_lines in abstracts_unprocessed:
            abstract = process_abstract(abstract_lines)
            abstracts.append(abstract)
    else:
        abstracts = []
    return article, abstracts, doc_indices, raw_article_sents

def main(unused_argv):
    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)
    if FLAGS.dataset_name == '':
        raise Exception('Must specify which dataset to convert.')
    process_dataset(FLAGS.dataset_name, FLAGS.out_data_path, FLAGS.TAC_path, FLAGS.DUC_path, FLAGS.custom_dataset_path)
    
if __name__ == '__main__':
    flags.DEFINE_string('dataset_name', 'example_custom_dataset', 'Which dataset to convert from raw data to tf examples')
    flags.DEFINE_string('out_data_path', 'tf_data', 'Where to put output tf examples')
    flags.DEFINE_string('TAC_path', '', 'Path to raw TAC data.')
    flags.DEFINE_string('DUC_path', '', 'Path to raw DUC data.')
    flags.DEFINE_string('custom_dataset_path', 'example_custom_dataset/', 'Path to custom dataset. Format of custom dataset must be:\n'
                        + 'One file for each topic...\n'
                        + 'Distinct articles will be separated by one blank line (two carriage returns \\n)...\n'
                        + 'Each sentence of the article will be on its own line\n'
                        + 'After all articles, there will be one blank line, followed by \'<SUMMARIES>\' without the quotes...\n'
                        + 'Distinct summaries will be separated by one blank line...'
                        + 'Each sentence of the summary will be on its own line'
                        + 'See the directory example_custom_dataset for an example')
    app.run(main)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    