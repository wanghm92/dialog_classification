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
    """preprocess the datasets

    Read text files from 3 categories
    Split into train, valid, test sets with 70: 15: 15 ratios
    Build the vocabulary from the train set and save it as a json file
    Build the tfds.features.text.TokenTextEncoder object and save it for encoding the texts later

    Args:
    args: dictionary of arguments - see get_args() for details
    """
    # for tokenization
    nlp = spacy.load("en_core_web_sm")

    '''
        Read text files, tokenize and conver to lower case
    '''
    alldata = {}
    for i, label in enumerate(categories):
        # assign numerical labels for the 3 categories
        alldata[i] = []
        # files are saved here for easy access for the purpose of a demo
        # TODO: move to a designated directory
        with io.open('{}/{}.txt'.format(DIR, label), 'r', encoding='utf-8') as fin:
            samples = fin.read().split(label)
            for s in tqdm(samples[1:]):
                # replacing the pipe '|' symbols with periods '.' to indicate the end of a sentence
                # TODO: another way is to use each sentence as individual samples
                words = s.lower().replace('|', ' . ').split()[:-1]
                # remove the leading/trailing quotes
                words = [t.strip().strip('"') for t in words]
                # retain non-empty words only
                temp = ' '.join([w for w in words if len(w) > 0])
                tokens = [token.text for token in nlp(temp)]
                # retain non-empty sentences only
                if len(tokens) > 0:
                    sent = ' '.join(tokens)
                    alldata[i].append(sent)

    '''
        Split into train, valid, test sets
    '''

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

    # save the datasets into separate text files for easy process later using the tf.data.TextLineDataset API
    # TODO: only for the purpose of this demo, should save more elegantly into json files
    for key, subset in dataset.items():
        for label, samples in subset.items():
            with io.open('{}.{}.txt'.format(key, label), 'w', encoding='utf-8') as fout:
                for sent in samples:
                    fout.write('{}\n'.format(sent))

    # build the vocab
    counter = Counter()
    for _, samples in dataset['train'].items():
        for sent in samples:
            counter.update([word.strip() for word in sent.strip().split() if len(word.strip()) > 0])

    # only retain words with >= min_freq frequencies
    vocab = [word for word, count in counter.most_common() if count >= args.min_freq]
    with io.open('vocab.json', 'w', encoding='utf-8') as fout:
        json.dump(vocab, fout)

    # save the compound TokenTextEncoder object to be used later
    encoder = tfds.features.text.TokenTextEncoder(vocab)
    encoder.save_to_file('encoder')


def load_data(args):

    # run preprocessing, only needed for the 1st run or text files are updated
    # preprocess(args)

    def labeler(example, index):
        return example, tf.cast(index, tf.int64)

    datasets = {
        'train': [],
        'test': [],
        'valid': []
    }

    # pair the numerical labels 0, 1, 2 with the samples
    for key in ['train', 'valid', 'test']:
        for label in range(3):
            lines_dataset = tf.data.TextLineDataset('{}.{}.txt'.format(key, label))
            datasets[key].append(lines_dataset.map(lambda ex: labeler(ex, label)))

    # Respectively for train, valid, test sets, combine sentences from 3 categories together
    for key, subsets in datasets.items():
        data = subsets[0]
        for x in subsets[1:]:
            data = data.concatenate(x)
        datasets[key] = data

    encoder = tfds.features.text.TokenTextEncoder.load_from_file('encoder')

    return datasets, encoder


def prepare_data(args):

    datasets, encoder = load_data(args)

    def encode(text_tensor, label):
        encoded_text = encoder.encode(text_tensor.numpy())
        return encoded_text, label

    def encode_map_fn(text, label):
        return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))

    # building tensorflow dataset objects, shuffle the train set
    train_data = datasets['train'].map(encode_map_fn).shuffle(BUFFER_SIZE)
    train_data = train_data.padded_batch(args.batch_size, padded_shapes=([-1],[]))

    test_data = datasets['test'].map(encode_map_fn).padded_batch(args.batch_size, padded_shapes=([-1],[]))
    val_data = datasets['valid'].map(encode_map_fn).padded_batch(args.batch_size, padded_shapes=([-1],[]))

    return train_data, val_data, test_data, encoder.vocab_size
