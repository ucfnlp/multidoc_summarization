import os
import numpy as np
import pyrouge
import tensorflow as tf
from absl import flags
from absl import flags
from absl import app
from absl import logging
import logging as log

FLAGS = flags.FLAGS

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

log_dir = '/home/logan/data/multidoc_summarization/logs/'
max_enc_steps = 100000
min_dec_steps = 100
max_dec_steps = 120

rouge_logan_dec_dir = 'duc_2004_reservoir_lambda_0.6_mute_7_tfidf'
rouge_orig_dec_dir = 'duc_2004_reservoir_lambda_0.6_mute_2_oracle'

flags.DEFINE_string('candidate_exp_name', rouge_logan_dec_dir, 'Path to system-generated summaries that we want to evaluate.')
flags.DEFINE_string('original_exp_name', rouge_orig_dec_dir, 'Path to system-generated summaries by a system we are comparing to.')



def rouge_eval_individual(ref_dir, dec_dir, id):
    """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
    r = pyrouge.Rouge155()
    id_str = '%05d' % id
    # id_str = ''
#   r.model_filename_pattern = '#ID#_reference.txt'
    r.model_filename_pattern = '#ID#' + id_str + '_reference.[A-Z].txt'
    r.system_filename_pattern = '(\d+)' + id_str + '_decoded.txt'
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

def rouge_all_docs(rouge_ref_dir, rouge_dec_dir):
    docs = os.listdir(rouge_dec_dir)
    all_results_dict = {}
    for doc_idx in range(len(docs)):
        ind_results_dict = rouge_eval_individual(rouge_ref_dir, rouge_dec_dir, doc_idx)
        all_results_dict[doc_idx] = ind_results_dict
    return all_results_dict

def print_colored_scores(logan_results_dict, orig_results_dict, candidate_dec_dir):
    docs = os.listdir(candidate_dec_dir)
    for doc_idx in range(len(docs)):
        logan_dict = logan_results_dict[doc_idx]
        orig_dict = orig_results_dict[doc_idx]
        out_str = "\nDoc #%d\nROUGE-1, ROUGE-2, ROUGE-SU4 (PRF):\n" % doc_idx
        # if (logan_dict['rouge_su4_f_score'] < orig_dict['rouge_su4_f_score']) or (
        #         logan_dict['rouge_1_f_score'] < orig_dict['rouge_1_f_score']) or (
        #         logan_dict['rouge_2_f_score'] < orig_dict['rouge_2_f_score']
        # ):
        for results_dict in [logan_dict, orig_dict]:
            for x in ["1", "2", "su4"]:
                for y in ["precision", "recall", "f_score"]:
                    key = "rouge_%s_%s" % (x, y)
                    val = results_dict[key]
                    logan_val = logan_dict[key]
                    orig_val = orig_dict[key]
                    if results_dict == logan_dict and logan_val < orig_val:
                        out_str += bcolors.WARNING
                    out_str += "%.4f\t" % (val)
                    if results_dict == logan_dict and logan_val < orig_val:
                        out_str += bcolors.ENDC
            out_str += "\n"
        print out_str

def main(unused_argv):
    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)
    candidate_dec_dir = log_dir + FLAGS.candidate_exp_name + '/decode_test_' + str(max_enc_steps) + \
                        'maxenc_4beam_' + str(min_dec_steps) + 'mindec_' + str(max_dec_steps) + 'maxdec_ckpt-238410/decoded'
    original_dec_dir = log_dir + FLAGS.original_exp_name + '/decode_test_' + str(max_enc_steps) + \
                        'maxenc_4beam_' + str(min_dec_steps) + 'mindec_' + str(max_dec_steps) + 'maxdec_ckpt-238410/decoded'
    original_dir = original_dec_dir.split('/')
    original_dir[-1] = 'reference'
    rouge_ref_dir = '/'.join(original_dir)
    logan_results_dict = rouge_all_docs(rouge_ref_dir, candidate_dec_dir)
    orig_results_dict = rouge_all_docs(rouge_ref_dir, original_dec_dir)
    print_colored_scores(logan_results_dict, orig_results_dict, candidate_dec_dir)





if __name__ == '__main__':
    app.run(main)






























