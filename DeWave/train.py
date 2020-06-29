'''
Script to train the model
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import re
import argparse
import numpy as np
import tensorflow as tf
from datagenerator import DataGenerator
from model import Model

from constant import *


parser = argparse.ArgumentParser("The function is to pack the audio files")
parser.add_argument("-m", "--model_dir", type=str, help="root directory which \
                    contains the fold of audio files from each speaker")
parser.add_argument("-s", "--summary_dir", type=str, help="output file name")
parser.add_argument("-t", "--train_pkl", type=str ,help="train_pkl localation")
parser.add_argument("-v","--val_pkl",type=str,help='val_pkl location')
args = parser.parse_args()


## model_dir is the directory in which stores the trained model
## sum_dir is the directory in which stores the summary of training process
## train_pkl are a list of training datasets in pkl format 
## val_pkl are a list of validation datasets in pkl format
def train(model_dir, sum_dir, train_pkl, val_pkl):
    print('model',model_dir)
    print('sun',sum_dir)
    print('train_pkl',train_pkl)
    print('val_pkl',val_pkl)
    lr = LEARNING_RATE
    n_hidden = N_HIDDEN
    max_steps = MAX_STEP
    batch_size = TRAIN_BATCH_SIZE

    train_loss_file = os.path.join(sum_dir, "train_loss")
    val_loss_file = os.path.join(sum_dir, "val_loss")

    with tf.Graph().as_default():
        # dropout keep probability
        p_keep_ff = tf.placeholder(tf.float32, shape=None)
        p_keep_rc = tf.placeholder(tf.float32, shape=None)
        # generator for training set and validation set
        data_generator = DataGenerator(train_pkl, batch_size)
        val_generator = DataGenerator(val_pkl, batch_size)
        # placeholder for input log spectrum, VAD info.,
        # and speaker indicator function
        in_data = tf.placeholder(
            tf.float32, shape=[batch_size, FRAMES_PER_SAMPLE, NEFF])
        VAD_data = tf.placeholder(
            tf.float32, shape=[batch_size, FRAMES_PER_SAMPLE, NEFF])
        Y_data = tf.placeholder(
            tf.float32, shape=[batch_size, FRAMES_PER_SAMPLE, NEFF, 2])
        # init the model
        BiModel = Model(n_hidden, batch_size, p_keep_ff, p_keep_rc)
        # build the net structure
        embedding = BiModel.inference2(in_data)
        print('embedding',embedding)
        Y_data_reshaped = tf.reshape(Y_data, [-1, NEFF, 2])
        VAD_data_reshaped = tf.reshape(VAD_data, [-1, NEFF])
        # compute the loss
        loss = BiModel.loss(embedding, Y_data_reshaped, VAD_data_reshaped)
        # get the train operation
        train_op = BiModel.train(loss, lr)
        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()
        sess = tf.Session()

        # either train from scratch or a trained model
        seeds = [f for f in os.listdir(model_dir) if re.match(r'model\.ckpt.*', f)]
        if len(seeds) > 0:
          saver.restore(sess, os.path.join(model_dir, "model.ckpt"))
        else:
          init = tf.global_variables_initializer()
          sess.run(init)

        init_step = 0
        if os.path.isfile(train_loss_file):
          train_loss = np.load(train_loss_file)
        else:
          train_loss = []
        if os.path.isfile(val_loss_file):
          val_loss = np.load(val_loss_file)
        else:
          val_loss = []

        summary_writer = tf.summary.FileWriter(sum_dir, sess.graph)
        last_epoch = data_generator.epoch

        for step in range(init_step, init_step + max_steps):
            start_time = time.time()
            data_batch = data_generator.gen_batch()
            # concatenate the samples into batch data
            in_data_np = np.concatenate(
                [np.reshape(item['Sample'], [1, FRAMES_PER_SAMPLE, NEFF])
                 for item in data_batch])
            VAD_data_np = np.concatenate(
                [np.reshape(item['VAD'], [1, FRAMES_PER_SAMPLE, NEFF])
                 for item in data_batch])
            VAD_data_np = VAD_data_np.astype('int')
            Y_data_np = np.concatenate(
                [np.reshape(item['Target'], [1, FRAMES_PER_SAMPLE, NEFF, 2])
                 for item in data_batch])
            Y_data_np = Y_data_np.astype('int')
            # train the model
            loss_value, _, summary_str = sess.run(
                [loss, train_op, summary_op],
                feed_dict={in_data: in_data_np,
                           VAD_data: VAD_data_np,
                           Y_data: Y_data_np,
                           p_keep_ff: 1 - P_DROPOUT_FF,
                           p_keep_rc: 1 - P_DROPOUT_RC})
            summary_writer.add_summary(summary_str, step)
            duration = time.time() - start_time
            # if np.isnan(loss_value):
                # import ipdb; ipdb.set_trace()
            assert not np.isnan(loss_value)
            if step % 10 == 0: 
                train_loss.append(loss_value.copy())
                # show training progress every 100 steps
                num_examples_per_step = batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = (
                    '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                    'sec/batch, epoch %d)')
                print (format_str % (datetime.now(), step, loss_value,
                                     examples_per_sec, sec_per_batch,
                                     data_generator.epoch))
            if step % 500 == 0:
                checkpoint_path = os.path.join(model_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path)

            if last_epoch != data_generator.epoch:
                # doing validation every training epoch
                print('Doing validation')
                val_epoch = val_generator.epoch
                count = 0
                loss_sum = 0
                # average the validation loss
                while(val_epoch == val_generator.epoch):
                    count += 1
                    data_batch = val_generator.gen_batch()
                    in_data_np = np.concatenate(
                        [np.reshape(item['Sample'],
                         [1, FRAMES_PER_SAMPLE, NEFF])
                         for item in data_batch])
                    VAD_data_np = np.concatenate(
                        [np.reshape(item['VAD'], [1, FRAMES_PER_SAMPLE, NEFF])
                         for item in data_batch])
                    VAD_data_np = VAD_data_np.astype('int')
                    Y_data_np = np.concatenate(
                        [np.reshape(item['Target'],
                         [1, FRAMES_PER_SAMPLE, NEFF, 2])
                         for item in data_batch])
                    Y_data_np = Y_data_np.astype('int')
                    loss_value, = sess.run(
                        [loss],
                        feed_dict={in_data: in_data_np,
                                   VAD_data: VAD_data_np,
                                   Y_data: Y_data_np,
                                   p_keep_ff: 1,
                                   p_keep_rc: 1})
                    loss_sum += loss_value
                val_loss.append(loss_sum / count)
                print ('validation loss: %.3f' % (loss_sum / count))
                np.save(train_loss_file, train_loss)
                np.save(val_loss_file, val_loss)

            last_epoch = data_generator.epoch

        np.save(train_loss_file, train_loss)
        np.save(val_loss_file, val_loss)
if __name__ == "__main__":
    print('%s start' % datetime.now())
    train(model_dir=args.model_dir,sum_dir=args.summary_dir,train_pkl=args.train_pkl,val_pkl=args.val_pkl)
