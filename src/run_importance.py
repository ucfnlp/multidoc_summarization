# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to process data into batches"""

import os
import matplotlib
from matplotlib import pyplot as plt
if not "DISPLAY" in os.environ:
    matplotlib.use("Agg")
import Queue
import glob
from random import shuffle, random
from sklearn import svm
from threading import Thread
import time
import numpy as np
import tensorflow as tf
import data
import sys
import time
import gpu_util
# best_gpu = str(gpu_util.pick_gpu_lowest_memory())
# if best_gpu != 'None':
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_util.pick_gpu_lowest_memory())
import tensorflow as tf
import numpy as np
from collections import namedtuple
from data import Vocab
from batcher import Batcher
from model import SummarizationModel
from decode import BeamSearchDecoder
import util
import cPickle
from tensorflow.python import debug as tf_debug
from tqdm import tqdm
from tqdm import trange
from absl import flags
from absl import app
from absl import logging

FLAGS = flags.FLAGS



def run_training(x, y):
    logging.info("starting run_training")
    if FLAGS.importance_fn == 'svr':
        clf = svm.SVR()
    elif FLAGS.importance_fn == 'svm':
        clf = svm.SVC()

    clf.fit(x, y)
    return clf

def load_data(data_path, num_instances, half_cnn_half_dm=False):
    print 'Loading data'
    filelist = glob.glob(data_path) # get the list of datafiles
    assert filelist, ('Error: Empty filelist at %s' % data_path) # check filelist isn't empty
    filelist = sorted(filelist)
    instances = []
    for file_name in tqdm(filelist):
        with open(file_name) as f:
            examples = cPickle.load(f)
        if num_instances == -1:
            num_instances = np.inf
        remaining_number = num_instances - sum([len(b) for b in instances])
        if len(examples) < remaining_number:
            instances.extend(examples)
        else:
            instances.extend(examples[:remaining_number])
            break
    print 'Finished loading data. Number of instances=%d' % len(instances)
    return instances

def predict_rouge_l(clf, x):
    logging.info("starting prediction")
    pred_y = clf.predict(x)
    return pred_y

def run_eval(pred_y, test_y):
    fig, ax = plt.subplots()
    rects1 = ax.bar(xrange(100), pred_y[:100], 0.5, color='r')
    rects2 = ax.bar(np.arange(100) + 0.5, test_y[:100], 0.5, color='g')
    plt.show()


def main(unused_argv):
    start_time = time.time()
    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    tf.set_random_seed(111) # a seed value for randomness
    logging.set_verbosity(logging.INFO) # choose what level of logging you want
    logging.info('Starting importance in %s mode...', (FLAGS.mode))

    if not os.path.exists(FLAGS.model_path):
        if FLAGS.mode=="train":
            os.makedirs(FLAGS.model_path)
        else:
            raise Exception("Model dir %s doesn't exist. Run in train mode to create it." % (FLAGS.model_path))
    if not os.path.exists(FLAGS.save_path):
        if FLAGS.mode=="decode":
            os.makedirs(FLAGS.save_path)



    # # If single_pass=True, check we're in decode mode
    # if FLAGS.single_pass and (FLAGS.mode!='decode'):
    #     raise Exception("The single_pass flag should only be True in decode mode")

    # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
    hparam_list = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm', 'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps', 'coverage', 'cov_loss_wt', 'pointer_gen']
    hps_dict = {}
    for key,val in FLAGS.__flags.iteritems(): # for each flag
        if key in hparam_list: # if it's in the list
            hps_dict[key] = val # add it to the dict
    hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)

    # # Create a batcher object that will create minibatches of data
    # batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=FLAGS.single_pass)


    if hps.mode == 'train':
        print "creating model..."
        x, y = load_data(FLAGS.data_path, FLAGS.training_instances)
        clf = run_training(x, y)
        with open(os.path.join(FLAGS.model_path, 'model.pickle'), 'wb') as f:
            cPickle.dump(clf, f)
        # test_x, test_y = load_data(FLAGS.data_path, 200000)
        # test_x = test_x[10000:20000]
        # test_y = test_y[10000:20000]
        a=0
    elif hps.mode == 'eval':
        _, test_y = load_data(FLAGS.data_path, FLAGS.training_instances)  # load test data
        pred_y = np.load(os.path.join(FLAGS.save_path, 'rouge_l.npz'))['arr_0']
        run_eval(pred_y, test_y)
    elif hps.mode == 'decode':
        with open(os.path.join(FLAGS.model_path, 'importance_svr_regular.pickle'), 'rb') as f:
            clf = cPickle.load(f)                                               # load model
        test_x, test_y = load_data(FLAGS.data_path, FLAGS.training_instances)   # load test data
        test_x = test_x[:100]
        test_y = test_y[:100]
        pred_y = predict_rouge_l(clf, test_x)                                   # predict rouge l scores
        print 'Saving ROUGE-L scores at %s' % os.path.join(FLAGS.save_path, 'rouge_l')
        np.savez_compressed(os.path.join(FLAGS.save_path, 'rouge_l'), pred_y)   # save rouge l scores
        run_eval(pred_y, test_y)
    else:
        raise ValueError("The 'mode' flag must be one of train/eval/decode")

    localtime = time.asctime( time.localtime(time.time()) )
    print ("Finished at: ", localtime)
    time_taken = time.time() - start_time
    if time_taken < 60:
        print('Execution time: ', time_taken, ' sec')
    elif time_taken < 3600:
        print('Execution time: ', time_taken/60., ' min')
    else:
        print('Execution time: ', time_taken/3600., ' hr')



if __name__ == '__main__':

    # Where to find data
    flags.DEFINE_string('data_path', '',
                               'Path expression to numpy datafiles. Can include wildcards to access multiple datafiles.')

    # Important settings
    flags.DEFINE_string('mode', '', 'must be one of train/eval/decode')

    # Where to save output
    flags.DEFINE_string('model_path', '/home/logan/data/multidoc_summarization/logs',
                               'Path expression to save model.')
    flags.DEFINE_string('save_path', '/home/logan/data/multidoc_summarization/cnn-dailymail/rouge-l-predictions',
                               'Path expression to save ROUGE-L scores.')
    flags.DEFINE_integer('training_instances', -1,
                                'Number of instances to load for training. Set to -1 to train on all.')
    flags.DEFINE_integer('feat_dim', 1026, 'Number of features in instances.')

    try:
        app.run(main)
    except util.InfinityValueError as e:
        sys.exit(100)
    except:
        raise
