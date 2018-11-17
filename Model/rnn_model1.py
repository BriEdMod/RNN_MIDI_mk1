import tensorflow as tf
import numpy as np
import time
import random
import argparse


class RNNNetwork:

    def __init__(self, input_size, batch_size, output_size, lstm_sz, n_layers, sess, learn_rate=0.01, name='rnn'):
        self.scope = name
        self.input_size = input_size
        self.lstm_sz = lstm_sz
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.output_size = output_size
        self.sess = sess
        self.learn_rate = tf.constant(learn_rate)
        self.lstm_last_state = np.zeros((self.n_layers * 2 * self.lstm_sz,))

        with tf.variable_scope(self.scope):
            self.X_input = tf.placeholder(tf.float32, shape=(None, None, self.input_size), name='X_input')
            self.lstm_init_val = tf.placeholder(tf.float32, shape=(None, self.n_layers * 2 * self.lstm_sz),
                                                name='lstm_init_val')

            self.lstm_cells = [
                tf.nn.rnn_cell.LSTMCell(self.lstm_sz, state_is_tuple=False
                                        ) for i in range(self.n_layers)
            ]
            self.lstm = tf.nn.rnn_cell.MultiRNNCell(self.lstm_cells, state_is_tuple=False)
            outputs, self.lstm_new_state = tf.nn.dynamic_rnn(cell=self.lstm, inputs=self.X_input, initial_state=self.lstm_init_val, dtype=tf.float32)
            self.rnn_out_weight = tf.Variable(tf.random_normal((self.lstm_sz, self.output_size), stddev=0.01))
            self.rnn_out_bias = tf.Variable(tf.random_normal((self.output_size,), stddev=0.01))
            outputs_reshape = tf.reshape(outputs, [-1, self.lstm_sz])
            network_output = tf.matmul(outputs_reshape, self.rnn_out_weight) + self.rnn_out_bias
            batch_time_shape = tf.shape(outputs)
            self.final_outputs = tf.reshape(tf.nn.softmax(network_output),
                                            (batch_time_shape[0], batch_time_shape[1], self.output_size))
            self.y_output = tf.placeholder(tf.float32, shape=(None, None, self.output_size))
            y_batch_long = tf.reshape(self.y_output, [-1, self.output_size])
            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=network_output, labels=y_batch_long))
            self.train_op = tf.train.AdamOptimizer(self.learn_rate, 0.9).minimize(self.cost)

    def run_step(self, x, init_zero_state=True):
        if init_zero_state:
            init_value = np.zeros((self.n_layers * 2 * self.lstm_sz))
        else:
            init_value = self.lstm_last_state
        output, next_lstm_state = self.sess.run(
            [self.final_outputs, self.lstm_new_state],
            feed_dict={
                self.X_input: [x], self.lstm_init_val: [init_value]
            }
        )

        self.lstm_last_state = next_lstm_state[0]
        return output[0][0]

    def train_batch(self, X_batch, y_batch):
        init_value = np.zeros((X_batch.shape[0], self.n_layers * 2 * self.lstm_sz))
        cost, _ = self.sess.run(
            [self.cost, self.train_op],
            feed_dict={
                self.X_input: X_batch, self.y_output: y_batch, self.lstm_init_val: init_value
            }
        )
        return cost


def embed_to_vocab(data_, vocab):
    data = np.zeros((len(data_), len(vocab)))
    cnt = 0
    for s in data_:
        v = [0.0] * len(vocab)
        v[vocab.index(s)] = 1.0
        data[cnt, :] = v
        cnt += 1
    return data


def decode_embed(array, vocab):
    return vocab[array.index(1)]


def load_data(input):
    data_ = ""
    with open(input, 'r') as f:
        data_ += f.read()
#    data_ = data_.lower()
    vocab = sorted(list(set(data_)))
    data = embed_to_vocab(data_, vocab)
    return data, vocab


def check_for_param(sess, saver, ckpt_folder):
    checkpoint = tf.train.get_checkpoint_state(ckpt_folder)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Loaded State")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default="shakespeare",
        help="Target text file"
    )
    parser.add_argument(
        "--test_prefix",
        type=str,
        default="A ",
        help="Prefix to train network"
    )
    parser.add_argument(
        "--ckpt_file",
        type=str,
        default="model",
        help="Checkpoint for model to load"
    )
    parser.add_argument(
        "--mode",
        default="talk",
        choices={"talk", "train"},
        help="Mode: talk or train"
    )
    args = parser.parse_args()

    input_file = args.input_file + '.txt'

    data, vocab = load_data('../Data/' + input_file)

    ckpt_folder = ('../Checkpoint/' + args.input_file + '/')
    ckpt_file = (args.ckpt_file + '.ckpt')
    final_ckpt_file = ckpt_folder + ckpt_file

    in_size = out_size = len(vocab)
    lstm_size = 256
    num_layer = 2
    batch_size = 64
    time_steps = 100

    NUM_TRAIN_BATCHES = 20000

    LEN_TEST_TEXT = 500

    TEST_PREFIX = args.test_prefix

    config = tf.ConfigProto()
    sess = tf.InteractiveSession(config=config)

    rn_net = RNNNetwork(
        input_size=in_size,
        lstm_sz=lstm_size,
        batch_size=batch_size,
        n_layers=num_layer,
        output_size=out_size,
        sess=sess,
        learn_rate=0.003,
        name='char_rnn_network'
    )

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())

    if args.mode == "train":

        check_for_param(sess, saver, ckpt_folder)
        last_time = time.time()
        X_batch = np.zeros((batch_size, time_steps, in_size))
        y_batch = np.zeros((batch_size, time_steps, in_size))
        possible_batch_ids = range(data.shape[0] - time_steps - 1)

        for i in range(NUM_TRAIN_BATCHES):
            batch_id = random.sample(possible_batch_ids, batch_size)

            for j in range(time_steps):
                ind1 = [k + j for k in batch_id]
                ind2 = [k + j + 1 for k in batch_id]

                X_batch[:, j, :] = data[ind1, :]
                y_batch[:, j, :] = data[ind2, :]

            cst = rn_net.train_batch(X_batch, y_batch)

            if (i % 100) == 0:
                new_time = time.time()
                diff = new_time - last_time
                last_time = new_time
                print("batch: ", i, " loss: ", cst, 100 / diff)

                saver.save(sess, final_ckpt_file)

    if args.mode == "talk":
        saver.restore(sess, final_ckpt_file)

        TEST_PREFIX = TEST_PREFIX.lower()
        for i in range(len(TEST_PREFIX)):
            out = rn_net.run_step(embed_to_vocab(TEST_PREFIX[i], vocab), i == 0)

        print("Sentence:")
        gen_str = TEST_PREFIX
        for i in range(LEN_TEST_TEXT):
            # Sample character from the network according to the generated
            # output probabilities.
            element = np.random.choice(range(len(vocab)), p=out)
            gen_str += vocab[element]
            out = rn_net.run_step(embed_to_vocab(vocab[element], vocab), False)

        print(gen_str)


if __name__ == "__main__":
    main()