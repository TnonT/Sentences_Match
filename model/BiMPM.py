# -*- coding = utf-8 -*-

# @author:黑白
# @contact:1808132036@qq.com
# @time:19-3-25上午10:15
# @file:BiMPM.py

import tensorflow as tf

class BIMPM(object):
    def __init__(self, args):
        self.args = args

    def _create_palceholder(self):
        self.sen1 = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.sen2 = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.sen1_len = tf.placeholder(dtype=tf.int32, shape=[None])
        self.sen2_len = tf.placeholder(dtype=tf.int32, shape=[None])
        self.truth = tf.placeholder(dtypr=tf.int32, shape=[None])

    def _create_feed_dict(self, sen1, sen2, sen1_len, sen2_len, truth):
        '''

        :param sen1: with shape [batch_size, max_sen1_len]
        :param sen2: with shape [batch_size, max_sen2_len]
        :param sen1_len: with shape [batch_size]
        :param sen2_len: with shape [batch_size]
        :param truth: with shape [batch_size]
        :return: feed_dict
        '''
        feed_dict = {
            self.sen1: sen1,
            self.sen2: sen2,
            self.sen1_len: sen1_len,
            self.sen2_len: sen2_len,
            self.truth: truth
        }
        return feed_dict

    def _biLSTM(self, inputs, inputs_len, scope, reuse=False):
        """

        :param scope: variable_scope
        :param inputs: with shape[batch_size, sen_len, dim]
        :param inputs_len: with shape[batch_szie]
        :param reuse: weather reuse the variable
        :return: outputs with shape[batch_size, sen_len, 2*args.lstm_units]
        """
        with tf.variable_scope(scope, reuse=reuse):
            cell = LSTMCell(self.args.lstm_units)
            drop_cell = lambda: DropoutWrapper(cell, output_keep_prob=self.args.dropout_rate)
            cell_fw, cell_bw = drop_cell(), drop_cell()
            batch_size = tf.shape(inputs)[0]
            init_state_fw = cell_fw.zero_state(batch_size, tf.float32)
            init_state_bw = cell_bw.zero_state(batch_size, tf.float32)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                         cell_bw=cell_bw,
                                                         inputs=inputs,
                                                         sequence_length=inputs_len,
                                                         initial_state_fw=init_state_fw,
                                                         initial_state_bw=init_state_bw,
                                                         dtype=tf.float32
                                                         )
            return tf.concat(outputs, axis=2)

    def _multi_perspective_cosline(self, sena, senb, matrix_name, k):
        with tf.variable_scope(matrix_name):
            W = tf.get_variable(name=matrix_name, dtype=tf.float32, shape=[tf.shape(sena)[2], k],
                                initializer=tf.contrib.layers.xavier_initializer())
            sena = tf.mat


    def _match(self, sena, senb, lena, lenb, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            with tf.variable_scope('Full_Match', reuse=reuse):
                # Full Match
                sena_full = sena  # [batch, sena_len, hidden_size*2]
                senb_full = senb[:, -1, :]  # [batch, 1, hidden_size*2]
                full_similarity = tf.matmul(sena_full, senb_full, transpose_b=True)  # [batch, lena, 1]




    def _create_model_graph(self, word_vec, iter_num, l2_lambda):
        # ==============    Embedding Layer    =========================
        with tf.device('/cpu:0'):
            self.word_embeddings = tf.get_variable('embedding', trainable=self.args.word_vec_trainable,
                                                   initializer=tf.constant(word_vec.word2vecs), dtype=tf.float32)
            sen1_emb = tf.nn.embedding_lookup(self.word_embeddings, self.sen1)
            sen2_emb = tf.nn.embedding_lookup(self.word_embeddings, self.sen2)

        # =============     Context Representation Layer    =====================
        sen1 = self._biLSTM(sen1_emb, self.sen1_len, 'biLSTM')
        sen2 = self._biLSTM(sen2_emb, self.sen2_len, 'biLSTM', reuse=True)

        # =============     Matching Layer   =====================
