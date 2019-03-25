# coding = utf-8

import tensorflow as tf
import numpy as np

def get_weight(name, shape, lamb):
    var = tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                          initializer=tf.truncated_normal_initializer(stddev=0.01))
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lamb)(var))  # 把正则化加入集合losses集合中
    return var

def collect_final_step_of_lstm(lstm_representation, lengths):
    # lstm_representation: [batch_size, passsage_length, dim]
    # lengths: [batch_size]
    lengths = tf.maximum(lengths, tf.ones_like(lengths, dtype=tf.int32))

    batch_size = tf.shape(lengths)[0]
    batch_num = tf.range(0, limit=batch_size)
    indices = tf.stack((batch_num, lengths - 1), axis=1)
    result = tf.gather_nd(lstm_representation, indices, name='last-forward_lstm')
    return result  # [batch_size, dim]


class ESIM_ATT(object):

    def __init__(self, args, word_vec, iter_num, is_training=True):
        self.args = args
        self.create_placeholder()
        self.create_model_graph(args, word_vec, iter_num, is_training)

    def create_placeholder(self):
        self.sentence1_lengths = tf.placeholder(tf.int32, [None])  # [batch_size]
        self.sentence2_lengths = tf.placeholder(tf.int32, [None])  # [batch_size]
        self.truth = tf.placeholder(tf.int64, [None])  # [batch_size]
        self.sentence1_words = tf.placeholder(tf.int32, [None, None])  # [batch_size, sentence1_len]
        self.sentence2_words = tf.placeholder(tf.int32, [None, None])  # [batch_size, sentence1_len]

    def _create_feed_dict(self, sentences1, sentences2, sen1_len, sen2_len, truth):  # 此函数待定
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

        # regularizer
        regularizer = tf.contrib.layers.l2_regularizer(0.001)

        # initializer
        initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                             uniform=True, dtype=tf.float32)

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
        cell_fw = tf.nn.rnn_cell.LSTMCell(args.lstm_units)
        cell_bw = tf.nn.rnn_cell.LSTMCell(args.lstm_units)

        # lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=1-args.dropout_rate)
        batch_size = tf.shape(self.sentence1_emb)[0]
        init_state_fw = cell_fw.zero_state(batch_size, tf.float32)
        init_state_bw = cell_bw.zero_state(batch_size, tf.float32)

        with tf.variable_scope('sentence_rnn') as scope:
            # sen1_outputs : 2 × [batch_size, sentence1_len, rnn_units]
            # sen1_fin_state : [batch_size, rnn_units]
            sen1_outputs, sen1_fin_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.sentence1_emb,
                                                                           self.sentence1_lengths, init_state_fw,
                                                                           init_state_bw)
            scope.reuse_variables()

            # sen2_outputs : ([batch_size, sentence1_len, rnn_units],[batch_size, sentence1_len, rnn_units])
            # sen2_fin_state : ([batch_size, rnn_units],[batch_size, rnn_units])
            sen2_outputs, sen2_fin_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.sentence2_emb,
                                                                           self.sentence2_lengths, init_state_fw,
                                                                           init_state_bw)

        sen1_outputs = tf.concat(sen1_outputs, axis=2)  # [batch_size, sen1_max_len, 2*rnn_units]
        sen2_outputs = tf.concat(sen2_outputs, axis=2)  # [batch_size, sen2_max_len, 2*rnn_units]

        similarity_matrix = tf.matmul(sen1_outputs,
                                      tf.transpose(sen2_outputs, [0, 2, 1]))  # [batch_size, sen1_max_len, sen2_max_len]
        sen1_mask = tf.cast(tf.sequence_mask(self.sentence1_lengths), tf.float32)  # [batch_size, sen1_max_len]
        sen2_mask = tf.cast(tf.sequence_mask(self.sentence2_lengths), tf.float32)  # [batch_size, sen2_max_len]

        similarity_matrix = tf.multiply(similarity_matrix, tf.expand_dims(sen1_mask, axis=2))
        similarity_matrix = tf.multiply(similarity_matrix, tf.expand_dims(sen2_mask, axis=1))

        sen1_att_repre = tf.matmul(tf.nn.softmax(similarity_matrix),
                                   sen2_outputs)  # [batch_size, sen1_max_len, 2*rnn_units]
        sen2_att_repre = tf.matmul(tf.nn.softmax(tf.transpose(similarity_matrix, [0, 2, 1])),
                                   sen1_outputs)  # [batch_size, sen2_max_len, 2*rnn_units]

        # sen1_att_repre = tf.multiply(tf.expand_dims(sen1_mask, axis=2), sen1_att_repre)
        # sen2_att_repre = tf.multiply(tf.expand_dims(sen2_mask, axis=2), sen2_att_repre)

        sen1_o_minus_sen1_att = tf.subtract(sen1_outputs, sen1_att_repre)
        sen2_o_minus_sen2_att = tf.subtract(sen2_outputs, sen2_att_repre)

        sen1_o_mul_sen1_att = tf.multiply(sen1_outputs, sen1_att_repre)
        sen2_o_mul_sen2_att = tf.multiply(sen2_outputs, sen2_att_repre)


        sen1_repre = tf.concat([sen1_outputs, sen1_att_repre, sen1_o_minus_sen1_att, sen1_o_mul_sen1_att], axis=2)
        sen2_repre = tf.concat([sen2_outputs, sen2_att_repre, sen2_o_minus_sen2_att, sen2_o_mul_sen2_att], axis=2)


        with tf.variable_scope("Aggrete") as scope:
            cell_fw = tf.nn.rnn_cell.LSTMCell(args.lstm_units)
            cell_bw = tf.nn.rnn_cell.LSTMCell(args.lstm_units)

            init_state_fw = cell_fw.zero_state(batch_size, tf.float32)
            init_state_bw = cell_bw.zero_state(batch_size, tf.float32)

            # init_state = lstm_cell.zero_state(batch_size, tf.float32)
            sen1_att_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, sen1_repre,
                                                                           self.sentence1_lengths, init_state_fw,
                                                                           init_state_bw)
            scope.reuse_variables()
            sen2_att_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, sen2_repre,
                                                                  self.sentence2_lengths, init_state_fw,
                                                                  init_state_bw)

        sen1_att_outputs = tf.concat(sen1_att_outputs, axis=2)  # [batch_size, sen1_len, 2*lstm_units]
        sen2_att_outputs = tf.concat(sen2_att_outputs, axis=2)  # [batch_size, sen2_len, 2*lstm_units]

        sen1_att_outputs_max_pool = tf.reduce_max(sen1_att_outputs, axis=1)  # [batch_size, 2*lstm_units]
        sen2_att_outputs_max_pool = tf.reduce_max(sen2_att_outputs, axis=1)  # [batch_size, 2*lstm_units]
        sen1_att_outputs_avg_pool = tf.reduce_mean(sen1_att_outputs, axis=1)  # [batch_size, 2*lstm_units]
        sen2_att_outputs_avg_pool = tf.reduce_mean(sen2_att_outputs, axis=1)  # [batch_size, 2*lstm_units]

        concat_representation = tf.concat([sen1_att_outputs_avg_pool,
                                           sen1_att_outputs_max_pool,
                                           sen2_att_outputs_avg_pool,
                                           sen2_att_outputs_max_pool], axis=1)

        concat_dim = 8 * args.lstm_units

        with tf.variable_scope('fc1'):
            w1 = tf.get_variable('W1', [concat_dim, 256], dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.01), regularizer=regularizer)
            b1 = tf.get_variable('b1', [256], dtype=tf.float32, initializer=tf.constant_initializer(0))
            fc1_output = tf.nn.relu(tf.nn.xw_plus_b(concat_representation, w1, b1, name='fc1'))

        with tf.variable_scope('fc2'):
            w2 = tf.get_variable('W2', [256, 2], dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.01), regularizer=regularizer)
            b2 = tf.get_variable('b2', [2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
            fc2_output = tf.nn.relu(tf.nn.xw_plus_b(fc1_output, w2, b2, name='fc2'))

        # with tf.variable_scope('fc3'):
        #     w3 = tf.get_variable('W2', [64, 2], dtype=tf.float32,
        #                          initializer=tf.truncated_normal_initializer(stddev=0.01), regularizer=regularizer)
        #     b3 = tf.get_variable('b3', [2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
        #     fc3_output = tf.nn.relu(tf.nn.xw_plus_b(fc2_output, w3, b3, name='fc3'))

        logits = fc2_output

        print("equal...........")
        predict = tf.equal(tf.argmax(logits, axis=1), self.truth)
        self.accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

        tf.summary.scalar('correct_evaluate', self.accuracy)

        # regularizer
        # tv = tf.trainable_variables()
        # regularization_cost = 0.001 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])


        # RNN最后时刻output来计算相似度的loss
        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.reshape(self.truth, [-1]), logits=logits))

        # RNN最后时刻output拼接起来通过全连接预测的
        # losses = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.truth, logits=logits), axis=-1)
        # self.loss = tf.reduce_mean(losses)
        # self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.truth, logits=logits))
        # ses_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.truth, logits=logits)) + regularization_cost
        #
        # tf.add_to_collection('losses', ses_loss)
        # self.loss = tf.add_n(tf.get_collection('losses'))
        # self.loss = tf.reduce_mean(
        #     tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.truth, logits=logits)) + regularization_cost

        # regularizer_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        # self.loss = tf.reduce_mean(
        #         tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.truth, logits=logits)) + regularizer_loss

        regularizer_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.truth, logits=logits))
        self.loss = loss + regularizer_loss

        tf.summary.scalar('loss', self.loss)

        self.summary_ops = tf.summary.merge_all()

        # if not is_training: return
        # 衰减学习率
        # learning_rate = tf.train.exponential_decay(
        #     args.learning_rate,  # base learning rate
        #     global_step,
        #     iter_num,
        #     args.learning_rate_decay
        #
        # )

        optimizer = tf.train.AdamOptimizer(args.learning_rate)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # optimizer = tf.train.AdadeltaOptimizer(learning_rate)

        self.train_op = optimizer.minimize(self.loss)


