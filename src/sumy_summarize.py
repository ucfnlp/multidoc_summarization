#Import library essentials
from sumy.parsers.html import HtmlParser #We're choosing a plaintext parser here, other parsers available for HTML etc.
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer #We're choosing Lexrank, other algorithms are also built in
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
import os
from tqdm import tqdm
import pyrouge
import logging as log
import nltk
import glob
from absl import logging

dataset = 'TAC'

if dataset == 'TAC':
    articles_dir = '/home/logan/data/multidoc_summarization/TAC_Data/logans_test/for_sumy/articles'
    abstract_dir = '/home/logan/data/multidoc_summarization/TAC_Data/logans_test/for_sumy/abstracts'
    out_dir = '/home/logan/data/multidoc_summarization/TAC_Data/logans_test/for_sumy'
elif 'DUC':
    articles_dir = '/home/logan/data/multidoc_summarization/DUC/logans_test/for_sumy/articles'
    abstract_dir = '/home/logan/data/multidoc_summarization/DUC/logans_test/for_sumy/abstracts'
    out_dir = '/home/logan/data/multidoc_summarization/DUC/logans_test/for_sumy'

reference_folder = 'reference'
decoded_folder = 'decoded'

summary_methods = ['lexrank', 'kl', 'sumbasic']

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
    log_str = "\nROUGE-1, ROUGE-2, ROUGE-SU4 (PRF):\n"
    for x in ["1","2","su4"]:
        for y in ["precision","recall","f_score"]:
            key = "rouge_%s_%s" % (x,y)
            val = results_dict[key]
            log_str += "%.4f\t" % (val)
    log_str += "\n"
    print(log_str)
    results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
    print("Writing final ROUGE results to %s...", results_file)
    with open(results_file, "w") as f:
        f.write(log_str)

for summary_method in summary_methods:
    print('Summarizing using the method: ' + summary_method)
    if summary_method == 'lexrank':
        summary_fn = LexRankSummarizer
    elif summary_method == 'kl':
        summary_fn = KLSummarizer
    elif summary_method == 'sumbasic':
        summary_fn = SumBasicSummarizer
    else:
        raise Exception('Could not find summary method ' + summary_method)

    if not os.path.exists(os.path.join(out_dir, summary_method, reference_folder)):
        os.makedirs(os.path.join(out_dir, summary_method, reference_folder))
    if not os.path.exists(os.path.join(out_dir, summary_method, decoded_folder)):
        os.makedirs(os.path.join(out_dir, summary_method, decoded_folder))
    print (os.path.join(out_dir, summary_method))
    article_names = sorted(os.listdir(articles_dir))
    for art_idx, article_name in enumerate(tqdm(article_names)):
        file = os.path.join(articles_dir, article_name)
        parser = HtmlParser.from_file(file, "", Tokenizer("english"))
        summarizer = summary_fn()

        summary = summarizer(parser.document, 5) #Summarize the document with 5 sentences
        summary = [str(sentence) for sentence in summary]
        with open(os.path.join(out_dir, summary_method, decoded_folder, article_name), 'wb') as f:
            f.write('\n'.join(summary))

        summary_tokenized = []
        for sent in summary:
            summary_tokenized.append(' '.join(nltk.tokenize.word_tokenize(sent.lower())))
        with open(os.path.join(abstract_dir, article_name)) as f:
            abstracts_text = f.read()
        abstracts = abstracts_text.split('\n\n')
        abstracts_sentences = []
        for abs_idx, abstract in enumerate(abstracts):
            abstract_sents = abstract.split('\n')
            tokenized_abstract_sents = [' '.join(nltk.tokenize.word_tokenize(sent)) for sent in abstract_sents]
            # tokenized_abstract_sents = '\n'.join(tokenized_abstract_sents)
            abstracts_sentences.append(tokenized_abstract_sents)

        write_for_rouge(abstracts_sentences, summary_tokenized, art_idx, os.path.join(out_dir, summary_method, reference_folder), os.path.join(out_dir, summary_method, decoded_folder))

    results_dict = rouge_eval(os.path.join(out_dir, summary_method, reference_folder), os.path.join(out_dir, summary_method, decoded_folder))
    print("Results_dict: ", results_dict)
    rouge_log(results_dict, os.path.join(out_dir, summary_method))

