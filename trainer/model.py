"""Defines a Keras model and input function for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def create_keras_model(vocab_size):

	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Embedding(vocab_size, 64))
	model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))

	model.add(tf.keras.layers.Dense(64, activation='relu'))
	model.add(tf.keras.layers.Dense(64, activation='relu'))

	model.add(tf.keras.layers.Dense(3, activation='softmax'))

	model.compile(
	optimizer='adam',
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy']
	)

	return model
