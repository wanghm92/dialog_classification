# Last modified by: Hongmin Wang
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trains a Keras model for dialog state prediction"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

# from . import model
# from . import utils

import model
import utils

import tensorflow as tf


def get_args():
    """Argument parser.

    Returns:
    Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='local or GCS location for writing checkpoints and exporting models')
    parser.add_argument(
        '--pkg-dir',
        type=str,
        default='trainer',
	help='')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=2,
        help='number of times to go through the data, default=20')
    parser.add_argument(
        '--min-freq',
        type=int,
        default=2,
        help='number of times a word appears in the dataset to be included in the vocab')
    parser.add_argument(
        '--batch-size',
        default=64,
        type=int,
        help='number of records to read during each training step, default=128')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    args, _ = parser.parse_known_args()
    return args


def train_and_evaluate(args):
    """Trains and evaluates the Keras model.

    Uses the Keras model defined in model.py and trains on data loaded and
    preprocessed in util.py. Saves the trained model in TensorFlow SavedModel
    format to the path defined in part by the --job-dir argument.

    Args:
    args: dictionary of arguments - see get_args() for details
    """
    train_data, val_data, test_data, vocab_size = utils.prepare_data(args)

    # Create the Keras Model
    keras_model = model.create_keras_model(vocab_size)

    # Setup Learning Rate decay.
    lr_decay_cb = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: args.learning_rate + 0.02 * (0.5 ** (1 + epoch)),
        verbose=True)

    # Setup TensorBoard callback.
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        os.path.join(args.job_dir, 'keras_tensorboard'),
        histogram_freq=1)

    # Train model
    keras_model.fit(
        train_data,
        epochs=args.num_epochs,
        validation_data=val_data,
        validation_steps=1
    )

    '''
    eval_loss, eval_acc = model.evaluate(test_data)
    print('\nEval loss: {:.3f}, Eval accuracy: {:.3f}'.format(eval_loss, eval_acc))
    '''

    export_path = os.path.join(args.job_dir, 'keras_export')
    tf.contrib.saved_model.save_keras_model(keras_model, export_path)
    print('Model exported to: {}'.format(export_path))


if __name__ == '__main__':
    args = get_args()
    tf.logging.set_verbosity(args.verbosity)
    train_and_evaluate(args)
