import os
import numpy as np
import pyrouge
import logging

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

rouge_logan_dec_dir = '/home/logan/data/multidoc_summarization/logs/tac_2011_logan/decode_test_10000maxenc_4beam_35mindec_100maxdec_ckpt-238410/decoded'
rouge_orig_dec_dir = '/home/logan/data/multidoc_summarization/logs/tac_2011/decode_test_10000maxenc_4beam_35mindec_100maxdec_ckpt-238410/decoded'
rouge_ref_dir = '/home/logan/data/multidoc_summarization/logs/tac_2011_logan/decode_test_10000maxenc_4beam_35mindec_100maxdec_ckpt-238410/reference'

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
    logging.getLogger('global').setLevel(logging.WARNING) # silence pyrouge logging
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

def print_colored_scores(logan_results_dict, orig_results_dict):
    docs = os.listdir(rouge_logan_dec_dir)
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


logan_results_dict = rouge_all_docs(rouge_ref_dir, rouge_logan_dec_dir)
orig_results_dict = rouge_all_docs(rouge_ref_dir, rouge_orig_dec_dir)
print_colored_scores(logan_results_dict, orig_results_dict)



































