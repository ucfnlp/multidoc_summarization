import glob
import struct
from tensorflow.core.example import example_pb2
import nltk
import os
from bs4 import BeautifulSoup
import re
import subprocess
import io
import tensorflow as tf
import sys
reload(sys)
sys.setdefaultencoding('utf8')
# article_file = '/home/logan/data/multidoc_summarization/TAC_Data/logans_test/article'
# abstract_file = '/home/logan/data/multidoc_summarization/TAC_Data/logans_test/abstract'
# out_file = '/home/logan/data/multidoc_summarization/TAC_Data/logans_test/test/tf_example'
# 
# def get_article_abstract():
#     with open(article_file) as f:
#         article = f.read()
#     with open(abstract_file) as f:
#         abstract_lines = f.readlines()
#         abstract = ''
#         for line in abstract_lines:
#             tokenized_sent = nltk.word_tokenize(line)
#             sent = ' '.join(tokenized_sent)
#             abstract += '<s> ' + sent + ' </s> '
#     return article, abstract
# 
# def write_example(article, abstract, writer):
#     tf_example = example_pb2.Example()
#     tf_example.features.feature['article'].bytes_list.value.extend([article])
#     tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
#     tf_example_str = tf_example.SerializeToString()
#     str_len = len(tf_example_str)
#     writer.write(struct.pack('q', str_len))
#     writer.write(struct.pack('%ds' % str_len, tf_example_str))
# 
# article, abstract = get_article_abstract()
# with open(out_file, 'wb') as writer:
#     write_example(article, abstract, writer)


FLAGS = tf.app.flags.FLAGS


data_dirs = {
    'tac_2011': {
        'article_dir': '/home/logan/data/multidoc_summarization/TAC_Data/summary_data/s11/test_doc_files',
        'abstract_dir': '/home/logan/data/multidoc_summarization/TAC_Data/summary_data/s11/models'
    },
    'tac_2010': {
        'article_dir': '/home/logan/data/multidoc_summarization/TAC_Data/summary_data/s10/test_doc_files',
        'abstract_dir': '/home/logan/data/multidoc_summarization/TAC_Data/summary_data/s10/models'
    },
    'tac_2008': {
        'article_dir': '/home/logan/data/multidoc_summarization/TAC_Data/summary_data/s08/test_doc_files',
        'abstract_dir': '/home/logan/data/multidoc_summarization/TAC_Data/summary_data/s08/models'
    },
    'duc_2004': {
        'article_dir': '/home/logan/data/multidoc_summarization/DUC/Original/DUC2004_Summarization_Documents/duc2004_testdata/tasks1and2/duc2004_tasks1and2_docs/docs',
        'abstract_dir': '/home/logan/data/multidoc_summarization/DUC/past_duc/duc2004/duc2004_results/ROUGE/eval/models/2'
    },
    'duc_2003': {
        'article_dir': '/home/logan/data/multidoc_summarization/DUC/Original/DUC2003_Summarization_Documents/duc2004_testdata/tasks1and2/duc2004_tasks1and2_docs/docs',
        'abstract_dir': '/home/logan/data/multidoc_summarization/DUC/past_duc/duc2003/duc2004_results/ROUGE/eval/models/2'
    }
}


use_stanford_tokenize = False

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

def stanford_corenlp_tokenize(text):
    input_file = 'stanford-core-nlp-input.txt'
    output_file = 'stanford-core-nlp-input.txt.conll'
    with open(input_file, 'wb') as f:
        f.write(text)
    subprocess.check_output(['java', '-cp', '"/home/logan/stanford-corenlp-full-2018-02-27/*"',
                                   '-Xmx2g', 'edu.stanford.nlp.pipeline.StanfordCoreNLP',
                                   '-annotators', 'tokenize,ssplit', '-file', input_file,
                                   '-outputFormat', 'conll'])
    with open(output_file) as f:
        res = f.read()
    sentences_conll = res.split('\n\n')
    sents = []
    for sent_conll in sentences_conll:
        if sent_conll == '':
            continue
        lines = sent_conll.split('\n')
        try:
            words = [fix_bracket_token(line.split('\t')[1]).lower() for line in lines]
        except:
            a = 0
        sents.append(words)
    return sents

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

def process_dataset(dataset_name):
    is_tac = 'tac' in dataset_name
    article_dir = data_dirs[dataset_name]['article_dir']
    abstract_dir = data_dirs[dataset_name]['abstract_dir']
    out_dir = os.path.join('/home/logan/data/multidoc_summarization/tf_examples', dataset_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    multidoc_dirnames = sorted(os.listdir(article_dir))
    out_idx = 1
    for multidoc_dirname in multidoc_dirnames:
        article, abstracts, doc_indices, raw_article_sents = get_article_abstract(multidoc_dirname, article_dir, abstract_dir, is_tac, is_single_doc=False)
        with open(os.path.join(out_dir, 'test_{:03d}.bin'.format(out_idx)), 'wb') as writer:
            write_example(article, abstracts, doc_indices, raw_article_sents, writer)
        out_idx += 1


def get_article_abstract_sumy(multidoc_dirname, article_dir, abstract_dir):
    if FLAGS.use_TAC:
        multidoc_dir = os.path.join(article_dir, multidoc_dirname, multidoc_dirname + '-A')
    else:
        multidoc_dir = os.path.join(article_dir, multidoc_dirname)

    doc_names = sorted(os.listdir(multidoc_dir))
    article = ''
    for doc_name in doc_names:
        if 'ENG' in doc_name or not FLAGS.use_TAC:

            doc_path = os.path.join(multidoc_dir, doc_name)
            with io.open(doc_path, encoding = "ISO-8859-1") as f:
                article_text = f.read()
            soup = BeautifulSoup(article_text, 'html.parser')
            if FLAGS.use_TAC:
                lines = []
                for tag in soup.findAll('p'):
                    contents = tag.renderContents().replace('\n', ' ').strip()
                    contents = ' '.join(contents.split())
                    lines.append(contents)
                article += ' '.join(lines) + ' '
            else:
                contents = soup.findAll('text')[0].renderContents().replace('\n', ' ').strip()
                contents = ' '.join(contents.split())
                article += contents + ' '
    abstracts = []
    doc_num = ''.join([s for s in multidoc_dirname if s.isdigit()])
    all_doc_names = os.listdir(abstract_dir)
    if FLAGS.use_TAC:
        abstract_doc_name = 'D' + doc_num + '-A'
    else:
        abstract_doc_name = 'D' + doc_num
    selected_doc_names = [doc_name for doc_name in all_doc_names if abstract_doc_name in doc_name]
    if len(selected_doc_names) == 0:
        raise Exception('no docs found for doc ' + doc_num)
    for selected_doc_name in selected_doc_names:
        with open(os.path.join(abstract_dir, selected_doc_name)) as f:
            abstract = f.read()
        abstract = abstract.replace('\x92', "'")
        abstract = abstract.encode('utf-8').strip()
        abstracts.append(abstract)
    # article = article.replace('\x92', "'")
    article = article.encode('utf-8').strip()
    return article, abstracts


def get_article(article_dir, multidoc_dirname, is_tac, full_article=True, remove_quotes=True, is_single_doc=False):
    if is_single_doc:
        doc_names = sorted(os.listdir(os.path.join(clustering_article_dir, multidoc_dirname)), key=int)
        article = ''
        for doc_name in doc_names:
            doc_path = os.path.join(clustering_article_dir, multidoc_dirname, doc_name)
            with open(doc_path) as f:
                lines = f.readlines()
            for line in lines:
                line = line.lower()
                tokenized_sent = nltk.word_tokenize(line)
                tokenized_sent = [fix_bracket_token(token) for token in tokenized_sent]
                sent = ' '.join(tokenized_sent)
                article += sent + ' '
        return article, None
    else:
        if is_tac:
            multidoc_dir = os.path.join(article_dir, multidoc_dirname, multidoc_dirname + '-A')
        else:
            multidoc_dir = os.path.join(article_dir, multidoc_dirname)

        doc_names = sorted(os.listdir(multidoc_dir))
        article = ''
        doc_indices = ''
        raw_article_sents = []
        for doc_idx, doc_name in enumerate(doc_names):
            if 'ENG' in doc_name or not is_tac:
                doc_path = os.path.join(multidoc_dir, doc_name)
                with open(doc_path) as f:
                    article_text = f.read()
                soup = BeautifulSoup(article_text, 'html.parser')
                if is_tac:
                    lines = []
                    for tag in soup.findAll('p'):
                        lines.append(tag.renderContents().replace('\n', ' ').strip())
                    contents = ' '.join(lines)
                    if use_stanford_tokenize:
                        sentences = stanford_corenlp_tokenize(contents)
                        if not full_article:
                            sentences = sentences[:5]
                        for sent in sentences:
                            sentence_text = ' '.join(sent)
                            article += sentence_text + ' '

                            doc_indices_for_tokens = [doc_idx] * len(sent)
                            doc_indices_str = ' '.join(str(x) for x in doc_indices_for_tokens)
                            doc_indices += doc_indices_str + ' '
                    else:
                        sentences = nltk.tokenize.sent_tokenize(contents)
                        if not full_article:
                            sentences = sentences[:5]
                        for orig_sent in sentences:
                            tokenized_sent = process_sent(orig_sent)
                            if remove_quotes and is_quote(tokenized_sent):
                                continue
                            sent = ' '.join(tokenized_sent)
                            article += sent + ' '

                            doc_indices_for_tokens = [doc_idx] * len(tokenized_sent)
                            doc_indices_str = ' '.join(str(x) for x in doc_indices_for_tokens)
                            doc_indices += doc_indices_str + ' '
                            raw_article_sents.append(orig_sent)
                else:
                    contents = soup.findAll('text')[0].renderContents().replace('\n', ' ').strip()
                    contents = ' '.join(contents.split())
                    if use_stanford_tokenize:
                        sentences = stanford_corenlp_tokenize(contents)
                        if not full_article:
                            sentences = sentences[:5]
                        for sent in sentences:
                            sentence_text = ' '.join(sent)
                            article += sentence_text + ' '

                            doc_indices_for_tokens = [doc_idx] * len(sent)
                            doc_indices_str = ' '.join(str(x) for x in doc_indices_for_tokens)
                            doc_indices += doc_indices_str + ' '
                    else:
                        sentences = nltk.tokenize.sent_tokenize(contents)
                        if not full_article:
                            sentences = sentences[:5]
                        for orig_sent in sentences:
                            tokenized_sent = process_sent(orig_sent)
                            if remove_quotes and is_quote(tokenized_sent):
                                continue
                            sent = ' '.join(tokenized_sent)
                            article += sent + ' '

                            doc_indices_for_tokens = [doc_idx] * len(tokenized_sent)
                            doc_indices_str = ' '.join(str(x) for x in doc_indices_for_tokens)
                            doc_indices += doc_indices_str + ' '
                            raw_article_sents.append(orig_sent)
        return article, doc_indices, raw_article_sents


def get_article_abstract(multidoc_dirname, article_dir, abstract_dir, is_tac, full_article=True, remove_quotes=True, is_single_doc=False):
    article, doc_indices, raw_article_sents = get_article(article_dir, multidoc_dirname, is_tac,
                                                          is_single_doc=False)
    #                 article += '<s> ' + sent + ' </s> '
    abstracts = []
    doc_num = ''.join([s for s in multidoc_dirname if s.isdigit()])
    all_doc_names = os.listdir(abstract_dir)
    if is_tac:
        abstract_doc_name = 'D' + doc_num + '-A'
    else:
        abstract_doc_name = 'D' + doc_num
    selected_doc_names = [doc_name for doc_name in all_doc_names if abstract_doc_name in doc_name]
    if len(selected_doc_names) == 0:
        raise Exception('no docs found for doc ' + doc_num)
    for selected_doc_name in selected_doc_names:
        with open(os.path.join(abstract_dir, selected_doc_name)) as f:
            abstract_lines = f.readlines()
        abstract = ''
        for line in abstract_lines:
            line = line.lower()
            line = line.replace('\x92', "'")
            tokenized_sent = nltk.word_tokenize(line)
            tokenized_sent = [fix_bracket_token(token) for token in tokenized_sent]
            sent = ' '.join(tokenized_sent)
            abstract += '<s> ' + sent + ' </s> '
        abstract = abstract.encode('utf-8').strip()
        abstracts.append(abstract)
    article = article.encode('utf-8').strip()
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


def main():
    if FLAGS.use_TAC:
        original_article_dir = '/home/logan/data/multidoc_summarization/TAC_Data/summary_data/s11/test_doc_files'
        original_abstract_dir = '/home/logan/data/multidoc_summarization/TAC_Data/summary_data/s11/models'
        sumy_article_out_dir = '/home/logan/data/multidoc_summarization/TAC_Data/logans_test/for_sumy/articles'
        sumy_abstract_out_dir = '/home/logan/data/multidoc_summarization/TAC_Data/logans_test/for_sumy/abstracts'
        stanford_out_dir = '/home/logan/data/multidoc_summarization/TAC_Data/logans_test/test'
        stanford_full_article_out_dir = '/home/logan/data/multidoc_summarization/TAC_Data/full_article_tf_examples/test'

        stanford_clustering_out_dir = '/home/logan/data/multidoc_summarization/20180314_TAC_2011_clustering/tf_examples/test'
        clustering_article_dir = '/home/logan/data/multidoc_summarization/20180314_TAC_2011_clustering/raw'
    else:
        original_article_dir = '/home/logan/data/multidoc_summarization/DUC/Original/DUC2004_Summarization_Documents/duc2004_testdata/tasks1and2/duc2004_tasks1and2_docs/docs'
        original_abstract_dir = '/home/logan/data/multidoc_summarization/DUC/past_duc/duc2004/duc2004_results/ROUGE/eval/models/2'
        sumy_article_out_dir = '/home/logan/data/multidoc_summarization/DUC/logans_test/for_sumy/articles'
        sumy_abstract_out_dir = '/home/logan/data/multidoc_summarization/DUC/logans_test/for_sumy/abstracts'
        stanford_out_dir = '/home/logan/data/multidoc_summarization/DUC/logans_test/test'
        stanford_full_article_out_dir = '/home/logan/data/multidoc_summarization/DUC/full_article_tf_examples/test'

        stanford_clustering_out_dir = '/home/logan/data/multidoc_summarization/20180314_DUC_2004_clustering/tf_examples/test'
        clustering_article_dir = '/home/logan/data/multidoc_summarization/20180314_DUC_2004_clustering/raw'

    if FLAGS.for_clustering_single_doc:
        article_dir = clustering_article_dir
        abstract_dir = original_abstract_dir
        out_dir = stanford_clustering_out_dir
    else:
        article_dir = original_article_dir
        abstract_dir = original_abstract_dir
        if FLAGS.full_article:
            out_dir = stanford_full_article_out_dir
        else:
            out_dir = stanford_out_dir

    if not os.path.exists(article_dir):
        os.makedirs(article_dir)
    if not os.path.exists(abstract_dir):
        os.makedirs(abstract_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(sumy_article_out_dir):
        os.makedirs(sumy_article_out_dir)
    if not os.path.exists(sumy_abstract_out_dir):
        os.makedirs(sumy_abstract_out_dir)

    if FLAGS.for_sumy:
        multidoc_dirnames = sorted(os.listdir(article_dir))
        out_idx = 1
        for multidoc_dirname in multidoc_dirnames:
            article, abstracts = get_article_abstract_sumy(multidoc_dirname, article_dir, abstract_dir)
            with open(os.path.join(sumy_article_out_dir, 'test_{:03d}.bin'.format(out_idx)), 'wb') as writer:
                writer.write(article)
            with open(os.path.join(sumy_abstract_out_dir, 'test_{:03d}.bin'.format(out_idx)), 'wb') as writer:
                writer.write('\n\n'.join(abstracts))
            out_idx += 1
    else:

        multidoc_dirnames = sorted(os.listdir(article_dir))
        out_idx = 1
        for multidoc_dirname in multidoc_dirnames:
            article, abstracts, doc_indices, raw_article_sents = get_article_abstract(multidoc_dirname, article_dir,
                              abstract_dir, FLAGS.use_TAC, full_article=FLAGS.full_article,
                              remove_quotes=FLAGS.remove_quotes, is_single_doc=FLAGS.for_clustering_single_doc)
            with open(os.path.join(out_dir, 'test_{:03d}.bin'.format(out_idx)), 'wb') as writer:
                write_example(article, abstracts, doc_indices, raw_article_sents, writer)
            out_idx += 1
    
if __name__ == '__main__':

    tf.app.flags.DEFINE_boolean('for_clustering_single_doc', False, 'Whether to use clustering format')
    tf.app.flags.DEFINE_boolean('for_sumy', False, 'Whether to use clustering format')
    tf.app.flags.DEFINE_boolean('use_TAC', True, 'Whether to use TAC or DUC')
    tf.app.flags.DEFINE_boolean('full_article', True, '')
    tf.app.flags.DEFINE_boolean('remove_quotes', True, 'Whether to use clustering format')
    tf.app.run()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    