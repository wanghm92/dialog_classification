from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_datasets as tfds
import os, sys, io, json, spacy, random, pdb
from pprint import pprint
from tqdm import tqdm
from collections import Counter
categories = ['NOT_INTERESTED', 'HUMAN_FALLBACK', 'STILL_INTERESTED']
DIR = 'trainer'
BUFFER_SIZE = 50000
random.seed(0)

def preprocess(args):
    nlp = spacy.load("en_core_web_sm")

    alldata = {}

    for i, label in enumerate(categories):
        alldata[i] = []
        with io.open('{}/{}.txt'.format(DIR, label), 'r', encoding='utf-8') as fin:
            samples = fin.read().split(label)
            for s in tqdm(samples[1:]):
                words = s.lower().replace('|', ' . ').split()[:-1]
                words = [t.strip().strip('"') for t in words]
                temp = ' '.join([w for w in words if len(w) > 0])
                tokens = [token.text for token in nlp(temp)]
                if len(tokens) > 0:
                    sent = ' '.join(tokens)
                    alldata[i].append(sent)

    dataset = {
        'train':{
            '0': [],
            '1': [],
            '2': []
        },
        'valid': {
            '0': [],
            '1': [],
            '2': []
        },
        'test': {
            '0': [],
            '1': [],
            '2': []
        }
    }

    for label, samples in alldata.items():
        size = len(samples)
        random.shuffle(samples)
        start = 0
        for key, cutoff in zip(['train', 'valid', 'test'], [0.7, 0.85, 1.0]):
            end = int(cutoff*size)
            print('start = {}'.format(start))
            print('end = {}'.format(end))
            dataset[key][str(label)].extend(samples[start:end])
            start = end
    # pdb.set_trace()
    for key, subset in dataset.items():
        for label, samples in subset.items():
            with io.open('{}.{}.txt'.format(key, label), 'w', encoding='utf-8') as fout:
                for sent in samples:
                    fout.write('{}\n'.format(sent))

    counter = Counter()
    for _, samples in dataset['train'].items():
        for sent in samples:
            counter.update([word.strip() for word in sent.strip().split() if len(word.strip()) > 0])

    vocab = [word for word, count in counter.most_common() if count >= args.min_freq]
    with io.open('vocab.json', 'w', encoding='utf-8') as fout:
        json.dump(vocab, fout)

    encoder = tfds.features.text.TokenTextEncoder(vocab)
    encoder.save_to_file('encoder')


def load_data(args):

    preprocess(args)

    def labeler(example, index):
        return example, tf.cast(index, tf.int64)

    datasets = {}

    for label, key in enumerate(['train', 'valid', 'test']):
        lines_dataset = tf.data.TextLineDataset('{}.{}.txt'.format(key, label))
        datasets[key] = lines_dataset.map(lambda ex: labeler(ex, label))

    encoder = tfds.features.text.TokenTextEncoder.load_from_file('encoder')

    return datasets, encoder


def prepare_data(args):

    datasets, encoder = load_data(args)

    def encode(text_tensor, label):
        encoded_text = encoder.encode(text_tensor.numpy())
        return encoded_text, label

    def encode_map_fn(text, label):
        return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))

    train_data = datasets['train'].map(encode_map_fn).shuffle(BUFFER_SIZE)
    train_data = train_data.padded_batch(args.batch_size, padded_shapes=([-1],[]))

    test_data = datasets['test'].map(encode_map_fn).padded_batch(args.batch_size, padded_shapes=([-1],[]))
    val_data = datasets['valid'].map(encode_map_fn).padded_batch(args.batch_size, padded_shapes=([-1],[]))

    return train_data, val_data, test_data, encoder.vocab_size
