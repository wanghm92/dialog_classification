from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_datasets as tfds
import os, sys, io
import spacy
#from pprint import pprint
from tqdm import tqdm
from collections import Counter
categories = ['NOT_INTERESTED', 'HUMAN_FALLBACK', 'STILL_INTERESTED']
DIR = 'trainer/'
#DIR = 'clean'
BUFFER_SIZE = 50000

def clean():
    nlp = spacy.load("en_core_web_sm")
    for label in categories:
        with io.open('{}/{}.txt'.format(DIR, label), 'r', encoding='utf-8') as fin, \
        io.open('{}/{}.clean.txt'.format(DIR, label), 'w', encoding='utf-8') as fout:
            samples = fin.read().split(label)
            for s in tqdm(samples[1:]):
                words = s.lower().replace('|', ' . ').split()[:-1]
                words = [t.strip().strip('"') for t in words]
                temp = ' '.join([w for w in words if len(w) > 0])
                tokens = [token.text for token in nlp(temp)]
                if len(tokens) > 0:
                    sent = ' '.join(tokens)
                    fout.write('{}\n'.format(sent))


def load_data(args):
    # only need to run once
    # clean()

    def labeler(example, index):
        return example, tf.cast(index, tf.int64)

    vocab = Counter()
    data_size = 0
    data_sets = []

    for i, label in enumerate(categories):
        with io.open('{}{}.clean.txt'.format(DIR, label), 'r', encoding='utf-8') as fin:
            texts = fin.read().strip()
            data_size += len(texts.split('\n'))
            vocab.update([x.strip() for x in texts.split() if len(x.strip()) > 0])
        lines_dataset = tf.data.TextLineDataset('{}{}.clean.txt'.format(DIR, label))
        labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
        data_sets.append(labeled_dataset)

    return data_sets, vocab, data_size


def prepare_data(args):

    data_sets, vocab, data_size = load_data(args)

    all_data = data_sets[0]
    for x in data_sets[1:]:
        all_data = all_data.concatenate(x)
    all_data = all_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)

    def encode(text_tensor, label):
        encoded_text = encoder.encode(text_tensor.numpy())
        return encoded_text, label

    def encode_map_fn(text, label):
        return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))

    encoder = tfds.features.text.TokenTextEncoder(vocab)
    encoded_data = all_data.map(encode_map_fn)

    train_size = int(0.7 * data_size)
    valid_size = int(0.15 * data_size)
    test_size = int(0.15 * data_size)

    train_data = encoded_data.take(train_size).shuffle(BUFFER_SIZE)
    train_data = train_data.padded_batch(args.batch_size, padded_shapes=([-1],[]))

    remaining = encoded_data.skip(train_size)

    test_data = remaining.take(test_size)
    test_data = test_data.padded_batch(args.batch_size, padded_shapes=([-1],[]))

    val_data = remaining.skip(valid_size)
    val_data = val_data.padded_batch(args.batch_size, padded_shapes=([-1],[]))

    return train_data, val_data, test_data, vocab
