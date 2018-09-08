import numpy as np
import tensorflow as tf

class CharAwareModel(object):
    def __init__(self, seq_len, char_vocab_size, char_embed_size,
                 max_word_len, filter_widths, num_filters,
                 num_highway_layer, num_lstm_layer, lstm_hidden_size,
                 word_vocab_size, keep_prob, clip_norm, is_training):
        self.seq_len = seq_len
        self.char_vocab_size = char_vocab_size
        self.char_embed_size = char_embed_size
        self.max_word_len = max_word_len
        self.filter_widths = filter_widths
        self.num_filters = num_filters
        assert len(self.filter_widths) == len(self.num_filters)
        self.num_highway_layer = num_highway_layer
        self.num_lstm_layer = num_lstm_layer
        self.lstm_hidden_size = lstm_hidden_size
        self.word_vocab_size = word_vocab_size
        self.keep_prob = keep_prob
        self.clip_norm = clip_norm
        self.is_training = is_training
        self.input_X = tf.placeholder(
            tf.int32, shape=(None, seq_len, max_word_len))
        self.input_y = tf.placeholder(tf.int32, shape=(None, seq_len))
        self.learning_rate = tf.placeholder(tf.float32, shape=[])

    def char_embedding(self):
        with tf.device("/cpu:0"):
            embeddings = tf.get_variable(
                "char_embedding",
                shape=[self.char_vocab_size, self.char_embed_size],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
            embed = tf.nn.embedding_lookup(embeddings, self.input_X)
            return embed

    def _conv_layer(self, X):
        all_pools = []
        for i in range(len(self.filter_widths)):
            width = self.filter_widths[i]
            with tf.variable_scope("conv_of_witdh_%d" % width):
                out_channels = self.num_filters[i]
                filters = tf.get_variable(
                    "filter",
                    shape=[width, self.char_embed_size, 1, out_channels],
                    dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable(
                    "bias",
                    initializer=tf.constant(0.1, shape=[out_channels]))
                # conv.shape = [batch_size, out_height, out_width, out_channels], where
                # out_width = max_word_len - width + 1
                conv = tf.nn.conv2d(
                    X,
                    filters,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                feature_map = tf.tanh(tf.nn.bias_add(conv, b), name="tanh")
                # pool.shape=[batch_size, 1, 1, out_channels]
                pool = tf.nn.max_pool(
                    feature_map,
                    ksize=[1, self.max_word_len - width + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool")
                all_pools.append(pool)
        # concat all pools along the dimension of out channels
        joined_pool = tf.concat(all_pools, axis=3)
        # total_num_filter = len(self.filter_widths) * self.num_filters
        total_num_filter = sum(self.num_filters)
        flat_pool = tf.reshape(
            joined_pool, shape=[-1, total_num_filter])
        return flat_pool

    def _highway_layer(self, X, name):
        input_size = X.shape[1].value
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            W_H = tf.get_variable(
                "W_H", shape=[input_size, input_size], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
            b_H = tf.get_variable(
                "b_H", initializer=tf.add(tf.zeros([input_size], tf.float32), -3))
            W_T = tf.get_variable(
                "W_H", shape=[input_size, input_size], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
            b_T = tf.get_variable(
                "b_T", initializer=tf.add(tf.zeros([input_size], tf.float32), -3))
            H_out = tf.nn.relu(tf.nn.xw_plus_b(X, W_H, b_H), name="H_out")
            T_out = tf.nn.relu(tf.nn.xw_plus_b(X, W_T, b_T), name="T_out")
            output = tf.multiply(H_out, T_out) + tf.multiply(X, 1 - T_out)
            return output

    def _lstm_layer(self, X):
        def make_cell():
            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.lstm_hidden_size)
            if self.is_training and self.keep_prob < 1.0:
                cell = tf.nn.rnn_cell.DropoutWrapper(
                    cell,
                    input_keep_prob=self.keep_prob,
                    output_keep_prob=self.keep_prob)
            return cell
        # multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_lstm_layer)
        multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(
            [make_cell() for _ in range(self.num_lstm_layer)])
        outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
        return outputs, states

    def _softmax_layer(self, X):
        # X.shape = [batch_size, seq_len, lstm_hidden_size]
        with tf.variable_scope("softmax"):
            W_softmax = tf.get_variable(
                "W_softmax",
                shape=[self.lstm_hidden_size, self.word_vocab_size],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
            b_softmax = tf.get_variable(
                "b_softmax", shape=[self.word_vocab_size], dtype=tf.float32)
            # Do NOT use tf.nn.xw_plus_b or tf.matmul, otherwise tensorflow
            # would throw exceptions like:
            # "ValueError: Shape must be rank 2 but is rank 3..." because
            # rank of X is 3 and rank of W_softmax is 2.
            # logits.shape = [batch_size, seq_len, word_vocab_size]
            logits = tf.tensordot(X, W_softmax, axes=[[2], [0]]) + b_softmax
            return logits

    def train(self, loss, global_step):
        with tf.name_scope("train"):
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self.learning_rate)
            grads, vars = zip(*optimizer.compute_gradients(loss))
            if self.is_training:
                grads, _ = tf.clip_by_global_norm(grads, self.clip_norm)
            train_op = optimizer.apply_gradients(
                zip(grads, vars), global_step=global_step)
            return train_op

    def inference(self, char_embeddings):
        # char_embeddings.shape = [batch_size, seq_len, max_word_len, char_embed_size]
        # conv_in.shape = [new_batch_size, max_word_len], where
        # new_batch_size = batch_size * seq_len
        char_embeds = tf.reshape(
            char_embeddings, shape=[-1, self.max_word_len, self.char_embed_size])
        conv_in = tf.expand_dims(char_embeds, -1)
        # conv_out.shape = [new_batch_size, total_pooling_size],where
        # total_pooling_size = len(filter_widths) * num_filters
        conv_out = self._conv_layer(conv_in)
        highway = conv_out
        for i in range(self.num_highway_layer):
            highway = self._highway_layer(highway, "highway_%d" % i)
        # lstm_in.shape = [batch_size, seq_len, total_pooling_size]
        total_num_filter = sum(self.num_filters)
        lstm_in = tf.reshape(
            highway,
            shape=[-1, self.seq_len, total_num_filter])
        # lstm_out.shape = [batch_size, seq_len, lstm_hidden_size]
        lstm_out, _ = self._lstm_layer(lstm_in)
        logits = self._softmax_layer(lstm_out)
        return logits

    def loss(self, logits):
        # logits.shape = [batch_size, seq_len, word_vocab_size]
        with tf.name_scope("loss"):
            # loss.shape = [batch_size]
            loss = tf.contrib.seq2seq.sequence_loss(
                logits,
                self.input_y,
                tf.ones(shape=tf.shape(self.input_y)),
                average_across_timesteps=False,
                average_across_batch=True)
            cost = tf.reduce_sum(loss, axis=0)
            return cost
