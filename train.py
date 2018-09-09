import os
import numpy as np
import tensorflow as tf
import data_utils
import char_aware_model

tf.flags.DEFINE_string(
    "data_path", None, "where the training/test data is stored")
tf.flags.DEFINE_float(
    "learning_rate",
    1.0, "initial learning rate. It will be decayed during training.")
tf.flags.DEFINE_integer(
    "seq_len", 35, "number of timesteps to unroll for")
tf.flags.DEFINE_integer(
    "char_embed_size", 15, "dimensionality of character embeddings")
tf.flags.DEFINE_integer(
    "max_word_len", 65, "maximum word length")
tf.flags.DEFINE_multi_integer(
    "filter_widths", [1,2,3,4,5,6,7], "conv net kernel widths")
tf.flags.DEFINE_multi_integer(
    "num_filters", [50,100,150,200,200,200,200],
    "number of feature maps in the CNN")
tf.flags.DEFINE_integer("num_highway_layer", 2, "number of highway layers")
tf.flags.DEFINE_integer("num_lstm_layer", 2, "number of layers in the LSTM")
tf.flags.DEFINE_integer("lstm_hidden_size", 650, "size of LSTM internal state")
tf.flags.DEFINE_float("clip_norm", 5.0, "normalize gradients at")
tf.flags.DEFINE_float(
    "keep_prob", 0.5,
    "the probability of keeping weights in the dropout layer")
tf.flags.DEFINE_integer("batch_size", 20, "the batch size")
tf.flags.DEFINE_integer(
    "num_epochs", 50, "number of epochs for training")

FLAGS = tf.flags.FLAGS

def get_ops(char_model):
    char_embeddings = char_model.char_embedding()
    logits = char_model.inference(char_embeddings)
    loss = char_model.loss(logits)
    global_step = tf.get_variable(
        "global_step",
        dtype=tf.int32,
        initializer=tf.constant(0, dtype=tf.int32),
        trainable=False)
    train_op = char_model.train(loss, global_step)
    return train_op, loss

def calc_ppl(cost, seq_len):
    # Perplexity is defined as the exp of cross-entropy.
    return np.exp(cost / seq_len)

def run_epoch(char_model, session, train_op, loss, raw_data,
              id_word_map, char_id_map, batch_size, learning_rate):
    epoch_size = (len(raw_data) / batch_size - 1) / char_model.seq_len
    total_cost = 0.0
    total_len = 0.0
    for i, (X_batch, y_batch) in enumerate(
        data_utils.data_iterator(raw_data, batch_size, char_model.seq_len)):
        X_char_batch = data_utils.word_ids_to_char_ids(
            X_batch, id_word_map, char_id_map, FLAGS.max_word_len)
        if char_model.is_training:
            _, cost = session.run(
                [train_op, loss],
                feed_dict={
                    char_model.input_X: X_char_batch,
                    char_model.input_y: y_batch,
                    char_model.learning_rate: learning_rate})
        else:
            cost = session.run(
                loss,
                feed_dict={
                    char_model.input_X: X_char_batch,
                    char_model.input_y: y_batch})
        if i % (epoch_size / 10) == 10:
            print("Step %d, cost: %f" % (i, cost))
        total_cost += cost
        total_len += char_model.seq_len
    ppl = calc_ppl(total_cost, total_len)
    return ppl

def main(_):
    train_data_path = os.path.join(FLAGS.data_path, "ptb.train.txt")
    test_data_path = os.path.join(FLAGS.data_path, "ptb.test.txt")
    valid_data_path = os.path.join(FLAGS.data_path, "ptb.valid.txt")
    train_words, test_words, valid_words = data_utils.get_raw_words(
        train_data_path, test_data_path, valid_data_path)
    word_id_map, id_word_map = data_utils.build_word_id_map(train_words)
    train_data, test_data, valid_data = data_utils.get_raw_data(
        word_id_map, train_words, test_words, valid_words)
    char_id_map, _ = data_utils.build_char_id_map(train_words)
    word_vocab_size = len(word_id_map)
    char_vocab_size = len(char_id_map)
    with tf.Graph().as_default():
        with tf.variable_scope("model", reuse=None):
            char_model_train = char_aware_model.CharAwareModel(
                FLAGS.seq_len,
                char_vocab_size + 3,
                FLAGS.char_embed_size,
                FLAGS.max_word_len + 2,
                FLAGS.filter_widths,
                FLAGS.num_filters,
                FLAGS.num_highway_layer,
                FLAGS.num_lstm_layer,
                FLAGS.lstm_hidden_size,
                word_vocab_size,
                FLAGS.clip_norm,
                FLAGS.keep_prob,
                is_training=True)
            train_train_op, train_loss = get_ops(char_model_train)
        with tf.variable_scope("model", reuse=True):
            char_model_valid = char_aware_model.CharAwareModel(
                FLAGS.seq_len,
                char_vocab_size + 3,
                FLAGS.char_embed_size,
                FLAGS.max_word_len + 2,
                FLAGS.filter_widths,
                FLAGS.num_filters,
                FLAGS.num_highway_layer,
                FLAGS.num_lstm_layer,
                FLAGS.lstm_hidden_size,
                word_vocab_size,
                FLAGS.clip_norm,
                FLAGS.keep_prob,
                is_training=False)
            valid_train_op, valid_loss = get_ops(char_model_valid)
        with tf.variable_scope("model", reuse=True):
            char_model_test = char_aware_model.CharAwareModel(
                FLAGS.seq_len,
                char_vocab_size + 3,
                FLAGS.char_embed_size,
                FLAGS.max_word_len + 2,
                FLAGS.filter_widths,
                FLAGS.num_filters,
                FLAGS.num_highway_layer,
                FLAGS.num_lstm_layer,
                FLAGS.lstm_hidden_size,
                word_vocab_size,
                FLAGS.clip_norm,
                FLAGS.keep_prob,
                is_training=False)
            test_train_op, test_loss = get_ops(char_model_test)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            prev_valid_ppl = 0.0
            lr = FLAGS.learning_rate
            for epoch in range(FLAGS.num_epochs):
                print("Epoch %d" % epoch)
                train_ppl = run_epoch(
                    char_model_train, sess, train_train_op, train_loss,
                    train_data, id_word_map, char_id_map, FLAGS.batch_size, lr)
                valid_ppl = run_epoch(
                    char_model_valid, sess, valid_train_op, valid_loss,
                    valid_data, id_word_map, char_id_map, FLAGS.batch_size,
                    FLAGS.learning_rate)
                print("Train perplexity: %f, valid perplexity: %f" % (train_ppl, valid_ppl))
                if prev_valid_ppl - valid_ppl < 1.0:
                    lr /= 2.0
                    print("Halve learning rate: %f" % lr)
                prev_valid_ppl = valid_ppl
            test_ppl = run_epoch(
                char_model_test, sess, test_train_op, test_loss, test_data,
                id_word_map, char_id_map, FLAGS.batch_size, FLAGS.learning_rate)
            print("Test perplexity: %f" % test_ppl)

if __name__ == "__main__":
    tf.app.run()
    