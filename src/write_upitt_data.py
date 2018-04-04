import glob
import struct
from tensorflow.core.example import example_pb2
import nltk
import os
from bs4 import BeautifulSoup
import re
import subprocess
import io
import string
import sys
reload(sys)
sys.setdefaultencoding('utf8')

article_dir = '/home/logan/data/multidoc_summarization/NN/raw'
out_dir = '/home/logan/data/multidoc_summarization/NN/tf_examples/test'

if not os.path.exists(article_dir):
    os.makedirs(article_dir)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

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

def get_article_abstract(doc_name, article_dir):
    article = ''
    with open(os.path.join(article_dir, doc_name)) as f:
        lines = f.readlines()
    for line in lines:
        line = line.lower()
        # line = fix_bracket_token(line)
        words = line.split()
        if words[-1] not in string.punctuation:
            words += ['.']
        article += ' '.join(words) + ' '
    max_dec_steps = doc_name.split('_')[0]
    abstracts = ['<s>' + max_dec_steps + '</s>']
    abstracts += ['<s>' + doc_name + '</s>']
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

doc_names = sorted(os.listdir(article_dir))
out_idx = 1
for doc_name in doc_names:
    article, abstracts = get_article_abstract(doc_name, article_dir)
    # with open(os.path.join(out_dir, ('test_{:03d}.bin' + doc_name).format(out_idx)), 'wb') as writer:
    with open(os.path.join(out_dir, doc_name), 'wb') as writer:
        write_example(article, abstracts, writer)
    out_idx += 1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    