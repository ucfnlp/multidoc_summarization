import glob
import struct
from tensorflow.core.example import example_pb2
import nltk
import os
from bs4 import BeautifulSoup
import re
import subprocess
import io
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
    
for_clustering_single_doc = False
for_sumy = False
use_TAC = True     # use TAC or DUC
full_article = True

use_stanford_tokenize = False

if use_TAC:
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

p_start_tag = '<P>'
p_end_tag = '</P>'

if for_clustering_single_doc:
    article_dir = clustering_article_dir
    abstract_dir = original_abstract_dir
    out_dir = stanford_clustering_out_dir
else:
    article_dir = original_article_dir
    abstract_dir = original_abstract_dir
    if full_article:
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


if for_sumy:
    def get_article_abstract(multidoc_dirname, article_dir, abstract_dir):
        if use_TAC:
            multidoc_dir = os.path.join(article_dir, multidoc_dirname, multidoc_dirname + '-A')
        else:
            multidoc_dir = os.path.join(article_dir, multidoc_dirname)

        doc_names = sorted(os.listdir(multidoc_dir))
        article = ''
        for doc_name in doc_names:
            if 'ENG' in doc_name or not use_TAC:

                doc_path = os.path.join(multidoc_dir, doc_name)
                with io.open(doc_path, encoding = "ISO-8859-1") as f:
                    article_text = f.read()
                soup = BeautifulSoup(article_text, 'html.parser')
                if use_TAC:
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
        if use_TAC:
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

    multidoc_dirnames = sorted(os.listdir(article_dir))
    out_idx = 1
    for multidoc_dirname in multidoc_dirnames:
        article, abstracts = get_article_abstract(multidoc_dirname, article_dir, abstract_dir)
        with open(os.path.join(sumy_article_out_dir, 'test_{:03d}.bin'.format(out_idx)), 'wb') as writer:
            writer.write(article)
        with open(os.path.join(sumy_abstract_out_dir, 'test_{:03d}.bin'.format(out_idx)), 'wb') as writer:
            writer.write('\n\n'.join(abstracts))
        out_idx += 1
else:
    def get_article(multidoc_dirname, is_single_doc=False):
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
            return article
        else:
            if use_TAC:
                multidoc_dir = os.path.join(article_dir, multidoc_dirname, multidoc_dirname + '-A')
            else:
                multidoc_dir = os.path.join(article_dir, multidoc_dirname)

            doc_names = sorted(os.listdir(multidoc_dir))
            article = ''
            for doc_name in doc_names:
                if 'ENG' in doc_name or not use_TAC:
                    doc_path = os.path.join(multidoc_dir, doc_name)
                    with open(doc_path) as f:
                        article_text = f.read()
                    soup = BeautifulSoup(article_text, 'html.parser')
                    if use_TAC:
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
                        else:
                            sentences = nltk.tokenize.sent_tokenize(contents)
                            if not full_article:
                                sentences = sentences[:5]
                            for line in sentences:
                                line = line.lower()
                                tokenized_sent = nltk.word_tokenize(line)
                                tokenized_sent = [fix_bracket_token(token) for token in tokenized_sent]
                                sent = ' '.join(tokenized_sent)
                                article += sent + ' '
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
                        else:
                            sentences = nltk.tokenize.sent_tokenize(contents)
                            if not full_article:
                                sentences = sentences[:5]
                            for line in sentences:
                                line = line.lower()
                                tokenized_sent = nltk.word_tokenize(line)
                                tokenized_sent = [fix_bracket_token(token) for token in tokenized_sent]
                                sent = ' '.join(tokenized_sent)
                                article += sent + ' '
            return article

    def get_article_abstract(multidoc_dirname, article_dir, abstract_dir):
        article = get_article(multidoc_dirname, is_single_doc=for_clustering_single_doc)
        #                 article += '<s> ' + sent + ' </s> '
        abstracts = []
        doc_num = ''.join([s for s in multidoc_dirname if s.isdigit()])
        all_doc_names = os.listdir(abstract_dir)
        if use_TAC:
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
        return article, abstracts

    def write_example(article, abstracts, writer):
        tf_example = example_pb2.Example()
        tf_example.features.feature['article'].bytes_list.value.extend([article])
        for abstract in abstracts:
            tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
        tf_example_str = tf_example.SerializeToString()
        str_len = len(tf_example_str)
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, tf_example_str))

    multidoc_dirnames = sorted(os.listdir(article_dir))
    out_idx = 1
    for multidoc_dirname in multidoc_dirnames:
        article, abstracts = get_article_abstract(multidoc_dirname, article_dir, abstract_dir)
        with open(os.path.join(out_dir, 'test_{:03d}.bin'.format(out_idx)), 'wb') as writer:
            write_example(article, abstracts, writer)
        out_idx += 1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    