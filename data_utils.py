#-*- coding:utf-8 -*-

import collections
import tensorflow as tf
import numpy as np

def file2words(filename):
    with open(filename, "r") as ifile:
        return ifile.read().decode("utf-8").replace("\n", "<eos>").split()

def build_word_id_map(words):
    counter = collections.Counter(words)
    # Counter is an unordered collection so sort words by their frequency
    # to get the uniquely identified id.
    word_cnt_pairs = sorted(counter.items(), key=lambda x: (x[1], x[0]))
    words, _ = zip(*word_cnt_pairs)
    num_words = len(words)
    word_id_map = dict(zip(words, range(num_words)))
    id_word_map = dict(zip(range(num_words), words))
    return word_id_map, id_word_map

def file2wordids(words, word_id_map):
    return [word_id_map[w] for w in words if w in word_id_map]

def get_raw_words(train_filename, test_filename, valid_filename):
    train_words = file2words(train_filename)
    test_words = file2words(test_filename)
    valid_words = file2words(valid_filename)
    return train_words, test_words, valid_words

def get_raw_data(word_id_map, train_words, test_words, valid_words):
    train_data = file2wordids(train_words, word_id_map)
    test_data = file2wordids(test_words, word_id_map)
    valid_data = file2wordids(valid_words, word_id_map)
    return train_data, test_data, valid_data

def data_iterator(raw_data, batch_size, num_steps):
    # I've read the ptb data processing code here:
    # https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/reader.py
    # However, tf.strided_slice brings a very big puzzle to me, 
    # until I found a very similar code here:
    # https://github.com/claravania/rnn-lm-tensorflow/blob/master/utils.py
    # Using the original loop method to replace tf.strided_slice,
    # and I will comment as detailedly as possible.
    # Given raw_data = [4, 3, 2, 1, 0, 5, 6, 1, 1, 1, 1, 0, 3, 4, 1], here we go!
    batch_len = len(raw_data) / batch_size
    # Split raw_data into batch_size parts, each batch contains batch_len words.
    # Given batch_size = 3, num_steps = 2, raw_data is splited as:
    # data = [[4, 3, 2, 1, 0], [5, 6, 1, 1, 1], [1, 0, 3, 4, 1]]
    data = [raw_data[batch_len * i : batch_len * (i + 1)] for i in range(batch_size)]
    # Each data element can generate epoch_size samples, e.g.
    # data[0] = [4, 3, 2, 1, 0] can generate 2 samples:
    # Xs = [[4, 3], [2, 1]] ys = [[3, 2], [1, 0]]
    epoch_size = (batch_len - 1) / num_steps
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
    for i in range(epoch_size):
        # E.g. i = 0, X_b = [[4, 3], [5, 6], [1, 0]], y_b = [[3, 2], [6, 1], [0, 3]]
        # i = 1, X_b = [[2, 1], [1, 1], [3, 4]], y_b = [[1, 0], [1, 1], [4, 1]]
        X_b = [data[j][i * num_steps : (i + 1) * num_steps] for j in range(batch_size)]
        y_b = [data[j][i * num_steps + 1 : (i + 1) * num_steps + 1] for j in range(batch_size)]
        yield (X_b, y_b)

def build_char_id_map(words):
    chars = [c for w in words for c in w]
    counter = collections.Counter(chars)
    char_cnt_pairs = sorted(counter.items(), key=lambda x: (x[1], x[0]))
    chars, _ = zip(*char_cnt_pairs)
    num_chars = len(chars)
    # 0, 1 and 2 are reserved for padding token, word begining token and
    # word endding token, respectively. So ids of normal characters starts
    # from 3.
    char_id_map = dict(zip(chars, [ i + 3 for i in range(num_chars)]))
    id_char_map = dict(zip([i + 3 for i in range(num_chars)], chars))
    return char_id_map, id_char_map

def word_ids_to_char_ids(word_ids, id_word_map, char_id_map, max_word_len):
    def word_id_to_char_ids(word_id):
        w = id_word_map[word_id]
        assert len(w) <= max_word_len
        char_ids = [0] * (max_word_len + 2)
        for i in range(len(w)):
            char_ids[i + 1] = char_id_map[w[i]]
        # Marking the begining and end of the word.
        char_ids[0] = 1
        char_ids[len(w) + 1] = 2
        return char_ids
    word_ids = np.array(word_ids)
    dest_shape = word_ids.shape + (max_word_len + 2,)
    flat = np.reshape(word_ids, (-1))
    char_ids = [word_id_to_char_ids(wid) for wid in flat]
    char_ids = np.reshape(char_ids, dest_shape)
    return char_ids

def get_max_word_length(words):
    sws = sorted(words, key=lambda x : len(x))
    return len(sws[-1])

