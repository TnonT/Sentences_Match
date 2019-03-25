# coding = utf-8

import tensorflow as tf
import numpy as np


def collect_final_step_of_lstm(selflstm_representation, lengths):
    # lstm_representation: [batch_size, passsage_length, dim]
    # lengths: [batch_size]
    lengths = tf.maximum(lengths, tf.ones_like(lengths, dtype=tf.int32))

    batch_size = tf.shape(lengths)[0]
    batch_num = tf.range(0, limit=batch_size)
    indices = tf.stack((batch_num, lengths-1), axis=1)
    result = tf.gather_nd(selflstm_representation, indices, name='last-forward_lstm')
    return result  # [batch_size, dim]

class LSTM:

    def __init__(self, args, word_vec, iter_num, is_training=True):
        self.args = args
        self.create_placeholder()
        self.create_model_graph(args, word_vec,  iter_num, is_training)

    def create_placeholder(self):
        self.sentence1_lengths = tf.placeholder(tf.int32, [None])  # [batch_size]
        self.sentence2_lengths = tf.placeholder(tf.int32, [None])  # [batch_size]
        self.truth = tf.placeholder(tf.int64, [None])  # [batch_size]
        self.sentence1_words = tf.placeholder(tf.int32, [None, None])  # [batch_size, sentence1_len]
        self.sentence2_words = tf.placeholder(tf.int32, [None, None])  # [batch_size, sentence1_len]


    def create_feed_dict(self, sentences1, sentences2, sen1_len, sen2_len, truth, is_training=False): # 此函数待定
        feed_dict = {
            self.sentence1_lengths: sen1_len,
            self.sentence2_lengths: sen2_len,
            self.truth: truth,
            self.sentence1_words: sentences1,
            self.sentence2_words: sentences2,
        }
        return feed_dict



    def create_model_graph(self, args, word_vec, iter_num, is_training=True):

        # global_step
        global_step = tf.Variable(0, trainable=False)

        # ======= sentence embedding  =========

        print('Build Embedding.........')

        self.word_embeddings = tf.get_variable('embedding', trainable=self.args.word_vec_trainable,
                                               initializer=tf.constant(word_vec.word2vecs), dtype=tf.float32)
        print('word embedding lookup.........')

        # [batch_size, sentence1_len, word_dim]
        self.sentence1_emb = tf.nn.embedding_lookup(self.word_embeddings, self.sentence1_words)

        # [batch_size, sentence1_len, word_dim]
        self.sentence2_emb = tf.nn.embedding_lookup(self.word_embeddings, self.sentence2_words)

        if is_training:
            self.sentence1_emb = tf.nn.dropout(self.sentence1_emb, args.dropout_rate)
            self.sentence2_emb = tf.nn.dropout(self.sentence2_emb, args.dropout_rate)


        print('lstm  .........')
        # ======= LSTM  ============
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(args.lstm_units)
        # lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=1-args.dropout_rate)
        batch_size = tf.shape(self.sentence1_emb)[0]
        init_state = lstm_cell.zero_state(batch_size, tf.float32)

        with tf.variable_scope('sentence_rnn') as scope:
            # sen1_outputs : [batch_size, sentence1_len, lstm_units]
            # sen1_fin_state : [batch_size, lstm_units]
            sen1_outputs, sen1_fin_state = tf.nn.dynamic_rnn(lstm_cell, self.sentence1_emb, self.sentence1_lengths,
                                                         initial_state=init_state, dtype=tf.float32)
            scope.reuse_variables()

            # sen2_outputs : [batch_size, sentence1_len, lstm_units]
            # sen2_fin_state : [batch_size, lstm_units]
            sen2_outputs, sen2_fin_state = tf.nn.dynamic_rnn(lstm_cell, self.sentence2_emb, self.sentence2_lengths,
                                                          initial_state=init_state, dtype=tf.float32)

        # 单纯 公里 
        # 取出输出的最后时刻状态来训练
        sen1_output = collect_final_step_of_lstm(sen1_outputs, self.sentence1_lengths)  # [batch_size, dim]
        sen2_output = collect_final_step_of_lstm(sen2_outputs, self.sentence2_lengths)  # [batch_size, dim]

        # with tf.variable_scope('sentence2_rnn', reuse=True):


        #  RNN最后时刻output来计算相似度
        # simlarity = tf.reduce_sum(tf.multiply(sen1_fin_state, sen2_fin_state), axis=-1)  # [batch_size]
        # logits = tf.nn.sigmoid(tf.reshape(simlarity, [-1]))

        # RNN最后时刻output拼接起来通过全连接预测的
        # id lstm
        # fin_state_concat = tf.concat([sen1_fin_state[1], sen2_fin_state[1]], axis=1)
        # if basic rnn
        fin_state_concat = tf.concat([sen1_output, sen2_output], axis=1)

        with tf.name_scope('fc1_layer'):
            w = tf.get_variable('fc1_w', shape=[args.lstm_units*2, 100], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
            b = tf.get_variable('fc1_b', shape=[100], dtype=tf.float32, initializer=tf.constant_initializer(0))

            fc1_outputs = tf.nn.relu(tf.nn.xw_plus_b(fin_state_concat, w, b, name='fc1'))


        with tf.name_scope('fc2_layer'):
            w = tf.get_variable('fc2_w', shape=[100, 50], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
            b = tf.get_variable('fc2_b', shape=[50], dtype=tf.float32, initializer=tf.constant_initializer(0))
            fc2_outputs = tf.nn.relu(tf.nn.xw_plus_b(fc1_outputs, w, b, name='fc2'))

        with tf.name_scope('fc3_layer'):
            w = tf.get_variable('fc3_w', shape=[50, 2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
            b = tf.get_variable('fc3_b', shape=[2], dtype=tf.float32, initializer=tf.constant_initializer(0))
            fc3_outputs = tf.nn.relu(tf.nn.xw_plus_b(fc2_outputs, w, b, name='fc3'))

        # 不需要fc3_outputs进行softmax处理, tf.nn.sparse_softmax_cross_entropy_with_logits会处理
        logits = fc3_outputs

        print("equal...........")
        predict = tf.equal(tf.argmax(logits, axis=1), self.truth)
        self.accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

        # RNN最后时刻output来计算相似度的loss
        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.reshape(self.truth, [-1]), logits=logits))

        # RNN最后时刻output拼接起来通过全连接预测的
        # losses = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.truth, logits=logits), axis=-1)
        # self.loss = tf.reduce_mean(losses)
        # self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.truth, logits=logits))
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.truth, logits=logits))
        # 衰减学习率

        # if not is_training: return

        learning_rate = tf.train.exponential_decay(
            args.learning_rate,  # base learning rate
            global_step,
            iter_num,
            args.learning_rate_decay

        )

        optimizer = tf.train.AdamOptimizer(learning_rate)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # optimizer = tf.train.AdadeltaOptimizer(learning_rate)

        self.train_op = optimizer.minimize(self.loss)















