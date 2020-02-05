"""Defines a Keras model and input function for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def create_keras_model(vocab_size):

	model = tf.keras.Sequential()

	# Embedding look up layer
	model.add(tf.keras.layers.Embedding(vocab_size, 64))

	# Bi-LSTM layer
	model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))

	# Two MLP layers
	model.add(tf.keras.layers.Dense(64, activation='relu'))
	model.add(tf.keras.layers.Dense(64, activation='relu'))

	# 3-way classification layer
	model.add(tf.keras.layers.Dense(3, activation='softmax'))

	# Compile the model, training using CE loss and Adam optimizer
	model.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)

	return model
