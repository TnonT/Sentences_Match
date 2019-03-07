# -*- coding = utf-8 -*-

# @author:黑白
# @contact:1808132036@qq.com
# @time:19-1-5下午3:49
# @file:mul_perspective_att.py
# @Tensorflow: v.18

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper

class Mul_Att(object):

    def __init__(self, args, word_vec, iter_num):
        self.args = args
        self._create_placeholder()

        self._create_model_graph(word_vec, iter_num)

    def _create_placeholder(self):
        self.sen1 = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.sen2 = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.sen1_len = tf.placeholder(dtype=tf.int32, shape=[None])
        self.sen2_len = tf.placeholder(dtype=tf.int32, shape=[None])
        self.truth = tf.placeholder(dtype=tf.int64, shape=[None])


    def _create_feed_dict(self, sen1, sen2, sen1_len, sen2_len, truth):
        """

        :param sen1: with shape[batch_size, max_sen1_len]
        :param sen2: with shape[batch_size, max_sen2_len]
        :param sen1_len: with shape[batch_size]
        :param sen2_len: with shape[batch_size]
        :param truth: with shape[batch_size]
        :return: feed_dict
        """
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


    def _mul_perspective_att(self, inputs1, inputs2, len1, len2, perspective_num, scope):
        """

        :param inputs1:
        :param inputs2:
        :param len1:
        :param len2:
        :param perspective_num:
        :param scope:
        :return:
        mul_inputs: [batch_size, max(len1), perspective_num*self.args.lstm_units]


        mul_att_inputs:[batch_size, persperctive_num, max(len), ars.lstm_units]

        mul_att_outputs:[batch_size, len, 2*arg.lstm_units]

        """
        with tf.variable_scope(scope):
            mul_inputs1 = tf.layers.dense(inputs1, perspective_num * self.args.lstm_units, activation=tf.nn.relu)
            mul_inputs1 = tf.reshape(mul_inputs1, (-1, tf.shape(mul_inputs1)[1], perspective_num, self.args.lstm_units))
            mul_inputs1 = tf.transpose(mul_inputs1, [0, 2, 1, 3])

            mul_inputs2 = tf.layers.dense(inputs2, perspective_num * self.args.lstm_units, activation=tf.nn.relu)
            mul_inputs2 = tf.reshape(mul_inputs2, (-1, tf.shape(mul_inputs2)[1], perspective_num, self.args.lstm_units))
            mul_inputs2 = tf.transpose(mul_inputs2, [0, 2, 1, 3])

            similary_matrix = tf.matmul(mul_inputs1, mul_inputs2, transpose_b=True)  # [bs, heads, len1, len2]

            inputs1_mask = tf.expand_dims(tf.expand_dims(tf.cast(tf.sequence_mask(len1), tf.float32), axis=1), axis=-1)
            inputs2_mask = tf.expand_dims(tf.expand_dims(tf.cast(tf.sequence_mask(len2), tf.float32), axis=1), axis=2)

            similary_matrix = tf.multiply(similary_matrix, inputs1_mask)
            similary_matrix = tf.multiply(similary_matrix, inputs2_mask)

            mul_att_inputs1 = tf.matmul(similary_matrix, mul_inputs2)
            mul_att_inputs2 = tf.matmul(similary_matrix, mul_inputs1, transpose_a=True)

            # mul_att_inputs1 = tf.concat(tf.split(mul_att_inputs1, axis=2), axis=1)
            # mul_att_inputs2 = tf.concat(tf.split(mul_att_inputs2, axis=2), axis=1)

            mul_att_output1_max = tf.reduce_max(mul_att_inputs1, axis=1)
            mul_att_output2_max = tf.reduce_max(mul_att_inputs2, axis=1)
            mul_att_output1_mean = tf.reduce_mean(mul_att_inputs1, axis=1)
            mul_att_output2_mean = tf.reduce_mean(mul_att_inputs2, axis=1)

            mul_att_output1 = tf.concat([mul_att_output1_max, mul_att_output1_mean], axis=2)
            mul_att_output2 = tf.concat([mul_att_output2_max, mul_att_output2_mean], axis=2)

            return mul_att_output1, mul_att_output2


    def _compositionBlock(self, sen1_inputs, sen2_inputs, scope):
        with tf.variable_scope(scope):
            sen1_outputs = self._biLSTM(sen1_inputs, self.sen1_len, scope='biLSTM', reuse=False)
            sen2_outputs = self._biLSTM(sen2_inputs, self.sen2_len, scope='biLSTM', reuse=True)

            sen1_outputs_avg = tf.reduce_mean(sen1_outputs, axis=1)
            sen2_outputs_avg = tf.reduce_mean(sen2_outputs, axis=1)
            sen1_outputs_max = tf.reduce_max(sen1_outputs, axis=1)
            sen2_outputs_max = tf.reduce_max(sen2_outputs, axis=1)

            merge_repre = tf.concat([sen1_outputs_avg, sen1_outputs_max, sen2_outputs_avg, sen2_outputs_max], axis=1)

            return merge_repre




    def _create_model_graph(self, word_vec, iter_num, l2_lambda=0.001):
        # ============   Embedding Layer  ===============
        with tf.device('/cpu:0'):
            self.word_embeddings = tf.get_variable('embedding', trainable=self.args.word_vec_trainable,
                                                   initializer=tf.constant(word_vec.word2vecs), dtype=tf.float32)
        sen1_emb = tf.nn.embedding_lookup(self.word_embeddings, self.sen1)
        sen2_emb = tf.nn.embedding_lookup(self.word_embeddings, self.sen2)


        # ===========   biLSTM Layer  ==================
        sen1 = self._biLSTM(sen1_emb, self.sen1_len, 'biLSTM')
        sen2 = self._biLSTM(sen2_emb, self.sen2_len, 'biLSTM', reuse=True)

        # ===========  mul_perspective_layer  ====================
        sen1, sen2 = self._mul_perspective_att(sen1, sen2, self.sen1_len, self.sen2_len, self.args.perspective_num, 'mul_perspective')

        # =========== composition layer  ========================
        merge_repre = self._compositionBlock(sen1, sen2, 'composition')

        with tf.variable_scope('feedforward_layer'):
            initializer = tf.random_normal_initializer(0.0, 0.1)

            with tf.variable_scope('feed_forward_layer1'):
                inputs = tf.nn.dropout(merge_repre, self.args.dropout_rate)
                outputs = tf.layers.dense(inputs, 256, tf.nn.relu, kernel_initializer=initializer)
            with tf.variable_scope('feed_forward_layer2'):
                outputs = tf.nn.dropout(outputs, self.args.dropout_rate)
                results = tf.layers.dense(outputs, 2, tf.nn.tanh, kernel_initializer=initializer)

        # ========================================================
        self.logits = results

        with tf.variable_scope('acc'):
            predict = tf.equal(tf.argmax(self.logits, axis=1), self.truth)
            accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))
            # label_pred = tf.argmax(self.logits, 1, name='label_pred')
            # label_true = tf.argmax(self.truth, 1, name='label_truth')
            # corrected_pred = tf.equal(tf.cast(label_pred, tf.int32), tf.cast(label_true, tf.int32))
            # accuracy = tf.reduce_mean(tf.cast(corrected_pred, tf.float32), name='Accuracy')
            self.accuracy = accuracy

        with tf.variable_scope('cost'):
            # losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.truth, logits=self.logits)
            # loss = tf.reduce_mean(losses, name='loss_val')
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.truth, logits=self.logits))
            weights = [v for v in tf.trainable_variables() if ('w' in v.name) or ('kernel' in v.name)]
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * l2_lambda
            loss += l2_loss
            self.loss = loss

        with tf.name_scope('training'):
            # global_step
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(
                self.args.learning_rate,  # base learning rate
                global_step,
                iter_num,
                self.args.learning_rate_decay

            )

            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_op = optimizer.minimize(self.loss)




