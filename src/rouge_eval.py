#Import library essentials
from sumy.parsers.html import HtmlParser #We're choosing a plaintext parser here, other parsers available for HTML etc.
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer #We're choosing Lexrank, other algorithms are also built in
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
import os
from tqdm import tqdm
import pyrouge
from absl import logging
import nltk
import glob
import logging as log
from absl import flags
from absl import app
import shutil


summaries_dir = '/home/logan/data/multidoc_summarization/sumrepo_duc2004'
ref_dir = '/home/logan/data/multidoc_summarization/sumrepo_duc2004/rouge/reference'
out_dir = '/home/logan/data/multidoc_summarization/sumrepo_duc2004/rouge'


summary_methods = ['Centroid', 'ICSISumm', 'DPP', 'Submodular']

data_dir = '/home/logan/data/multidoc_summarization/tf_examples'
log_dir = '/home/logan/data/multidoc_summarization/logs/'
max_enc_steps = 100000
min_dec_steps = 100
max_dec_steps = 120
test_folder = 'decode_test_' + str(max_enc_steps) + \
                        'maxenc_4beam_' + str(min_dec_steps) + 'mindec_' + str(max_dec_steps) + 'maxdec_ckpt-238410'

flags.DEFINE_string('exp_name', 'reference', 'Path to system-generated summaries that we want to evaluate.' +
                           ' If you want to run on human summaries, then enter "reference".')
flags.DEFINE_string('dataset', 'tac_2011', 'Which dataset to use. Can be {duc_2004, tac_2011, etc}')

FLAGS = flags.FLAGS

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

def extract_digits(text):
    digits = int(''.join([s for s in text if s.isdigit()]))
    return digits

def main(unused_argv):

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    if FLAGS.exp_name == 'extractive':
        for summary_method in summary_methods:
            if not os.path.exists(os.path.join(out_dir, summary_method, 'decoded')):
                os.makedirs(os.path.join(out_dir, summary_method, 'decoded'))
            if not os.path.exists(os.path.join(out_dir, summary_method, 'reference')):
                os.makedirs(os.path.join(out_dir, summary_method, 'reference'))
            print (os.path.join(out_dir, summary_method))
            method_dir = os.path.join(summaries_dir, summary_method)
            file_names = sorted([name for name in os.listdir(method_dir) if name[0] == 'd'])
            for art_idx, article_name in enumerate(tqdm(file_names)):
                file = os.path.join(method_dir, article_name)
                with open(file, 'rb') as f:
                    lines = f.readlines()
                tokenized_sents = [[token.lower() for token in nltk.tokenize.word_tokenize(line)] for line in lines]
                sentences = [' '.join(sent) for sent in tokenized_sents]
                processed_summary = '\n'.join(sentences)
                out_name = '%06d_decoded.txt' % art_idx
                with open(os.path.join(out_dir, summary_method, 'decoded', out_name), 'wb') as f:
                    f.write(processed_summary)

                reference_files = glob.glob(os.path.join(ref_dir, '%06d'%art_idx + '*'))
                abstract_sentences = []
                for ref_file in reference_files:
                    with open(ref_file) as f:
                        lines = f.readlines()
                    abstract_sentences.append(lines)
                write_for_rouge(abstract_sentences, sentences, art_idx, os.path.join(out_dir, summary_method, 'reference'), os.path.join(out_dir, summary_method, 'decoded'))

            results_dict = rouge_eval(ref_dir, os.path.join(out_dir, summary_method, 'decoded'))
            # print("Results_dict: ", results_dict)
            rouge_log(results_dict, os.path.join(out_dir, summary_method))

        for summary_method in summary_methods:
            print summary_method
        all_results = ''
        for summary_method in summary_methods:
            sheet_results_file = os.path.join(out_dir, summary_method, "sheets_results.txt")
            with open(sheet_results_file) as f:
                results = f.read()
            all_results += results
        print all_results
        a=0

    else:
        # source_dir = os.path.join(data_dir, FLAGS.dataset)
        summary_dir = os.path.join(log_dir,FLAGS.exp_name,test_folder)
        ref_dir = os.path.join(summary_dir, 'reference')
        dec_dir = os.path.join(summary_dir, 'decoded')
        summary_files = glob.glob(os.path.join(log_dir + FLAGS.exp_name, 'test_*.txt.result.summary'))
        if len(summary_files) > 0 and not os.path.exists(dec_dir):      # reformat files from extract + rewrite
            os.makedirs(ref_dir)
            os.makedirs(dec_dir)
            for summary_file in summary_files:
                ex_index = extract_digits(os.path.basename(summary_file))
                new_file = os.path.join(dec_dir, "%06d_decoded.txt" % ex_index)
                shutil.copyfile(summary_file, new_file)

            ref_files_to_copy = glob.glob(os.path.join(log_dir, FLAGS.dataset, test_folder, 'reference', '*'))
            for file in ref_files_to_copy:
                basename = os.path.basename(file)
                shutil.copyfile(file, os.path.join(ref_dir, basename))
        
        results_dict = rouge_eval(ref_dir, dec_dir)
        rouge_log(results_dict, summary_dir)


if __name__ == '__main__':
    app.run(main)
