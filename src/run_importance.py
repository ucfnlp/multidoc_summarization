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
import os
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
from matplotlib import pyplot as plt

FLAGS = tf.app.flags.FLAGS

# Where to find data
tf.app.flags.DEFINE_string('data_path', '', 'Path expression to numpy datafiles. Can include wildcards to access multiple datafiles.')

# Important settings
tf.app.flags.DEFINE_string('mode', '', 'must be one of train/eval/decode')

# Where to save output
tf.app.flags.DEFINE_string('model_path', '/home/logan/data/multidoc_summarization/logs/importance_svr', 'Path expression to save model.')
tf.app.flags.DEFINE_string('save_path', '/home/logan/data/multidoc_summarization/cnn-dailymail/rouge-l-predictions', 'Path expression to save ROUGE-L scores.')
tf.app.flags.DEFINE_integer('training_instances', -1, 'Number of instances to load for training. Set to -1 to train on all.')
tf.app.flags.DEFINE_integer('feat_dim', 1026, 'Number of features in instances.')



# class Example(object):
#     """Class representing a train/val/test example for text summarization."""
#
#     def __init__(self, x, y, hps):
#         """Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.
#
#         Args:
#             article: source text; a string. each token is separated by a single space.
#             abstract_sentences: list of strings, one per abstract sentence. In each sentence, each token is separated by a single space.
#             vocab: Vocabulary object
#             hps: hyperparameters
#         """
#         self.hps = hps
#         self.x = x
#         self.y = y
#
#
# class Batch(object):
#     """Class representing a minibatch of train/val/test examples for text summarization."""
#
#     def __init__(self, example_list, hps, vocab):
#         """Turns the example_list into a Batch object.
#
#         Args:
#              example_list: List of Example objects
#              hps: hyperparameters
#              vocab: Vocabulary object
#         """
#         self.init_encoder_seq(example_list, hps) # initialize the input to the encoder
#         self.init_decoder_seq(example_list, hps) # initialize the input and targets for the decoder
#
#     def init_encoder_seq(self, example_list, hps):
#         """Initializes the following:
#                 self.enc_batch:
#                     numpy array of shape (batch_size, <=max_enc_steps) containing integer ids (all OOVs represented by UNK id), padded to length of longest sequence in the batch
#                 self.enc_lens:
#                     numpy array of shape (batch_size) containing integers. The (truncated) length of each encoder input sequence (pre-padding).
#                 self.enc_padding_mask:
#                     numpy array of shape (batch_size, <=max_enc_steps), containing 1s and 0s. 1s correspond to real tokens in enc_batch and target_batch; 0s correspond to padding.
#
#             If hps.pointer_gen, additionally initializes the following:
#                 self.max_art_oovs:
#                     maximum number of in-article OOVs in the batch
#                 self.art_oovs:
#                     list of list of in-article OOVs (strings), for each example in the batch
#                 self.enc_batch_extend_vocab:
#                     Same as self.enc_batch, but in-article OOVs are represented by their temporary article OOV number.
#         """
#
#         # self.input = np.zeros((hps.batch_size, example_list[0].shape[0]), dtype=np.float32)
#         self.input = np.stack([ex.x for ex in example_list])
#
#     def init_decoder_seq(self, example_list, hps):
#         """Initializes the following:
#                 self.dec_batch:
#                     numpy array of shape (batch_size, max_dec_steps), containing integer ids as input for the decoder, padded to max_dec_steps length.
#                 self.target_batch:
#                     numpy array of shape (batch_size, max_dec_steps), containing integer ids for the target sequence, padded to max_dec_steps length.
#                 self.dec_padding_mask:
#                     numpy array of shape (batch_size, max_dec_steps), containing 1s and 0s. 1s correspond to real tokens in dec_batch and target_batch; 0s correspond to padding.
#                 """
#         self.target = np.stack([ex.y for ex in example_list])
#
#
# class Batcher(object):
#     """A class to generate minibatches of data. Buckets examples together based on length of the encoder sequence."""
#
#     BATCH_QUEUE_MAX = 100 # max number of batches the batch_queue can hold
#
#     def __init__(self, data_path, hps, single_pass):
#         """Initialize the batcher. Start threads that process the data into batches.
#
#         Args:
#             data_path: tf.Example filepattern.
#             vocab: Vocabulary object
#             hps: hyperparameters
#             single_pass: If True, run through the dataset exactly once (useful for when you want to run evaluation on the dev or test set). Otherwise generate random batches indefinitely (useful for training).
#         """
#         self._data_path = data_path
#         self._hps = hps
#         self._single_pass = single_pass
#
#         # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
#         self._batch_queue = Queue.Queue(self.BATCH_QUEUE_MAX)
#         self._example_queue = Queue.Queue(self.BATCH_QUEUE_MAX * self._hps.batch_size)
#
#         # Different settings depending on whether we're in single_pass mode or not
#         if single_pass:
#             self._num_example_q_threads = 1 # just one thread, so we read through the dataset just once
#             self._num_batch_q_threads = 1	# just one thread to batch examples
#             self._bucketing_cache_size = 1 # only load one batch's worth of examples before bucketing; this essentially means no bucketing
#             self._finished_reading = False # this will tell us when we're finished reading the dataset
#         else:
#             self._num_example_q_threads = 16 # num threads to fill example queue
#             self._num_batch_q_threads = 4	# num threads to fill batch queue
#             self._bucketing_cache_size = 100 # how many batches-worth of examples to load into cache before bucketing
#
#         # Start the threads that load the queues
#         self._example_q_threads = []
#         for _ in xrange(self._num_example_q_threads):
#             self._example_q_threads.append(Thread(target=self.fill_example_queue))
#             self._example_q_threads[-1].daemon = True
#             self._example_q_threads[-1].start()
#         self._batch_q_threads = []
#         for _ in xrange(self._num_batch_q_threads):
#             self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
#             self._batch_q_threads[-1].daemon = True
#             self._batch_q_threads[-1].start()
#
#         # Start a thread that watches the other threads and restarts them if they're dead
#         if not single_pass: # We don't want a watcher in single_pass mode because the threads shouldn't run forever
#             self._watch_thread = Thread(target=self.watch_threads)
#             self._watch_thread.daemon = True
#             self._watch_thread.start()
#
#
#     def next_batch(self):
#         """Return a Batch from the batch queue.
#
#         If mode='decode' or 'calc_features' then each batch contains a single example repeated beam_size-many times; this is necessary for beam search.
#
#         Returns:
#             batch: a Batch object, or None if we're in single_pass mode and we've exhausted the dataset.
#         """
#         # If the batch queue is empty, print a warning
#         if self._batch_queue.qsize() == 0:
#             tf.logging.warning('Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i', self._batch_queue.qsize(), self._example_queue.qsize())
#             if self._single_pass and self._finished_reading:
#                 tf.logging.info("Finished reading dataset in single_pass mode.")
#                 return None
#
#         batch = self._batch_queue.get() # get the next Batch
#         return batch
#
#     def fill_example_queue(self):
#         """Reads data from file and processes into Examples which are then placed into the example queue."""
#
#         input_gen = example_generator(self._data_path, self._single_pass)
#         # counter = 0
#         while True:
#             try:
#                 (x, y) = input_gen.next()  # read the next example from file. article and abstract are both strings.
#             except StopIteration:  # if there are no more examples:
#                 tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
#                 if self._single_pass:
#                     tf.logging.info(
#                         "single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
#                     self._finished_reading = True
#                     break
#                 else:
#                     raise Exception("single_pass mode is off but the example generator is out of data; error.")
#
#             example = Example(x, y, self._hps)  # Process into an Example.
#             self._example_queue.put(example)  # place the Example in the example queue.
#             # print "example num", counter
#             # counter += 1
#
#     def fill_batch_queue(self):
#         """Takes Examples out of example queue, sorts them by encoder sequence length, processes into Batches and places them in the batch queue.
#
#         In decode mode, makes batches that each contain a single example repeated.
#         """
#         while True:
#
#             # print 'hi'
#             if self._hps.mode != 'decode' and self._hps.mode != 'calc_features':
#                 # Get bucketing_cache_size-many batches of Examples into a list, then sort
#                 inputs = []
#                 for _ in xrange(self._hps.batch_size * self._bucketing_cache_size):
#                     inputs.append(self._example_queue.get())
#
#                 # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
#                 batches = []
#                 for i in xrange(0, len(inputs), self._hps.batch_size):
#                     batches.append(inputs[i:i + self._hps.batch_size])
#                 # if not self._single_pass:
#                 #     shuffle(batches)
#                 for b in batches:	# each b is a list of Example objects
#                     self._batch_queue.put(Batch(b, self._hps, self._vocab))
#
#             elif self._hps.mode == 'decode': # beam search decode mode
#                 ex = self._example_queue.get()
#                 b = [ex for _ in xrange(self._hps.batch_size)]
#                 self._batch_queue.put(Batch(b, self._hps, self._vocab))
#
#
#
#     def watch_threads(self):
#         """Watch example queue and batch queue threads and restart if dead."""
#         while True:
#             time.sleep(60)
#             for idx,t in enumerate(self._example_q_threads):
#                 if not t.is_alive(): # if the thread is dead
#                     tf.logging.error('Found example queue thread dead. Restarting.')
#                     new_t = Thread(target=self.fill_example_queue)
#                     self._example_q_threads[idx] = new_t
#                     new_t.daemon = True
#                     new_t.start()
#             for idx,t in enumerate(self._batch_q_threads):
#                 if not t.is_alive(): # if the thread is dead
#                     tf.logging.error('Found batch queue thread dead. Restarting.')
#                     new_t = Thread(target=self.fill_batch_queue)
#                     self._batch_q_threads[idx] = new_t
#                     new_t.daemon = True
#                     new_t.start()
#
#
#
# def example_generator(data_path, single_pass):
#     """Generates tf.Examples from data files.
#
#         Binary data format: <length><blob>. <length> represents the byte size
#         of <blob>. <blob> is serialized tf.Example proto. The tf.Example contains
#         the tokenized article text and summary.
#
#     Args:
#         data_path:
#             Path to tf.Example data files. Can include wildcards, e.g. if you have several training data chunk files train_001.bin, train_002.bin, etc, then pass data_path=train_* to access them all.
#         single_pass:
#             Boolean. If True, go through the dataset exactly once, generating examples in the order they appear, then return. Otherwise, generate random examples indefinitely.
#
#     Yields:
#         Deserialized tf.Example.
#     """
#     while True:
#         filelist = glob.glob(data_path) # get the list of datafiles
#         assert filelist, ('Error: Empty filelist at %s' % data_path) # check filelist isn't empty
#         if single_pass:
#             filelist = sorted(filelist)
#         else:
#             random.shuffle(filelist)
#         for f in filelist:
#             examples = np.load(f)
#             reader = open(f, 'rb')
#             for example in examples:
#                 yield example[:-1], example[-1]
#         if single_pass:
#             print "example_generator completed reading all datafiles. No more data."
#             break
#


def run_training(x, y):
    tf.logging.info("starting run_training")
    clf = svm.SVR()
    clf.fit(x, y)
    return clf

def load_data(data_path, num_instances):
    print 'Loading data'
    filelist = glob.glob(data_path) # get the list of datafiles
    assert filelist, ('Error: Empty filelist at %s' % data_path) # check filelist isn't empty
    filelist = sorted(filelist)
    count = 0
    x = []
    y = []
    for f in tqdm(filelist):
        examples = np.load(f)['arr_0']
        my_x = examples[:,:-1]
        my_y = examples[:,-1]
        if num_instances == -1:
            num_instances = np.inf
        remaining_number = num_instances - sum([len(b) for b in x])
        if len(examples) < remaining_number:
            x.append(my_x)
            y.append(my_y)
        else:
            x.append(my_x[:remaining_number])
            y.append(my_y[:remaining_number])
            break
    x = np.concatenate(x)
    y = np.concatenate(y)
    print 'Finished loading data. Number of instances=%d' % len(x)
    return x, y

def predict_rouge_l(clf, x):
    tf.logging.info("starting prediction")
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
    tf.logging.set_verbosity(tf.logging.INFO) # choose what level of logging you want
    tf.logging.info('Starting importance in %s mode...', (FLAGS.mode))

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
        with open(os.path.join(FLAGS.model_path, 'model.pickle'), 'rb') as f:
            clf = cPickle.load(f)                                               # load model
        test_x, test_y = load_data(FLAGS.data_path, FLAGS.training_instances)   # load test data
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
    try:
        tf.app.run()
    except util.InfinityValueError as e:
        sys.exit(100)
    except:
        raise
