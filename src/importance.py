import string
# import time
# import os
# import tensorflow as tf
import numpy as np
import struct
from tensorflow.core.example import example_pb2
from gensim.models import KeyedVectors
import nltk
from nltk.corpus import stopwords
import os
stop_words = set(stopwords.words('english'))
from matplotlib import pyplot as plt
import textwrap as tw
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from ngram import NGram
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
font = {'family' : 'normal',
        'size'   : 5}

embedding_file = '/home/logan/data/multidoc_summarization/GoogleNews-vectors-negative300.bin'
input_dir = '/home/logan/data/multidoc_summarization/TAC_Data/logans_test/test'
file_name = 'test_001.bin'

def tokenize_and_embed(text, model, oov_embs=None):
    if oov_embs is None:
        oov_embs = {}
    sent_embs = []
    sents = text.split(' . ')
    for sent in sents:
        embs = []
        tokens = sent.split(' ')
        for token in tokens:
            if token in stop_words or token in ('<s>', '</s>'):
                continue
            is_punctuation = [ch in string.punctuation for ch in token]
            if all(is_punctuation):
                continue
            if token in ('-lrb-', '-rrb-', '-lsb-', '-rsb-'):
                continue
            if token in oov_embs:
                embs.append(oov_embs[token])
            elif not token in model:
                oov_embs[token] = 0.1 * np.random.randn(300).astype(float)
                embs.append(oov_embs[token])
            else:
                embs.append(model[token])
        if len(embs) == 0:
            continue
        embs = np.stack(embs)
        mean_emb = np.mean(embs, 0)
        sent_embs.append(mean_emb)
    return np.stack(sent_embs), oov_embs


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.

    Rows are scores for each class.
    Columns are predictions (samples).
    """
    scoreMatExp = np.exp(np.asarray(x))
    return scoreMatExp / scoreMatExp.sum(0)

# Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
model = KeyedVectors.load_word2vec_format(embedding_file, binary=True)

dog = model['dog']
print(dog.shape)
print(dog[:10])

def dot_product_similarity(article_embs, abstract_embs):
    return np.matmul(article_embs, np.transpose(abstract_embs))

def ngram_similarity(article_sents, abstract_sents):
    res = np.zeros([len(article_sents), len(abstract_sents)], dtype=float)
    for art_idx, art_sent in enumerate(article_sents):
        for abs_idx, abs_sent in enumerate(abstract_sents):
            res[art_idx, abs_idx] = NGram.compare(art_sent, abs_sent)
    return res

def calc_importances(file_name, model, similarity_fn=cosine_similarity, max_articles=5, use_tf_idf=False, use_softmax=True):
    reader = open(os.path.join(input_dir, file_name), 'rb')
    article_idx = 0
    if max_articles is None:
        max_articles = np.inf
    while article_idx < max_articles:
        article_idx += 1
        len_bytes = reader.read(8)
        if not len_bytes: break  # finished reading this file
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        e = example_pb2.Example.FromString(example_str)
        article_text = e.features.feature['article'].bytes_list.value[
            0].lower()  # the article text was saved under the key 'article' in the data files
        article_sents_for_graph = article_text.split(' . ')
        for i in range(len(article_sents_for_graph)):
            article_sents_for_graph[i] = str(i) + '. ' + article_sents_for_graph[i]
        abstract_texts = []
        for abstract in e.features.feature['abstract'].bytes_list.value:
            abstract_texts.append(abstract.lower())
        article_sents = article_text.split(' . ')
        abstract_sents = abstract_texts[0].replace('<s>', '').replace('</s>', '').split(' . ')
        if similarity_fn == ngram_similarity:
            article_embs = article_sents
            abstract_embs = abstract_sents
        elif use_tf_idf:
            documents = article_sents + abstract_sents
            # documents = []
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
            article_embs = tfidf_matrix[:len(article_sents)]
            abstract_embs = tfidf_matrix[len(article_sents):]
        else:
            article_embs, oov_embs = tokenize_and_embed(article_text, model)
            abstract_embs, _ = tokenize_and_embed(abstract_texts[0], model, oov_embs)
            for key in oov_embs.iterkeys():
                print key
            print '-----------'
        similarity_matrix = similarity_fn(article_embs, abstract_embs)
        importances_hat = np.sum(similarity_matrix, 1)
        print('Importances_hat:', importances_hat)
        if use_softmax:
            importances = softmax(importances_hat)
        else:
            importances = importances_hat / np.sum(importances_hat)

        fig, ax1 = plt.subplots()
        fig.subplots_adjust(left=0.9, top=1.0, bottom=0.03, right=1.0)

        # Example data
        people = (article_sents_for_graph)
        y_pos = np.arange(len(people))
        performance = importances

        ax1.barh(y_pos, performance, align='center',
                color='green', ecolor='black')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(people)
        ax1.invert_yaxis()  # labels read top-to-bottom
        ax1.set_xlabel('Performance')
        ax1.set_title('How fast do you want to go today?')

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
        fig_txt = tw.fill(tw.dedent(abstract_texts[0]), width=80)
        plt.figtext(0.5, 0.5, fig_txt, horizontalalignment='center',
                    fontsize=9, multialignment='left',
                    bbox=dict(boxstyle="round", facecolor='#D8D8D8',
                              ec="0.5", pad=0.5, alpha=1), fontweight='bold')
        # plt.tight_layout()
        plt.show()
        a=0

calc_importances(file_name, model)
a=0

from sumy.parsers.html import HtmlParser
import sumy.evaluation.rouge as rouge
