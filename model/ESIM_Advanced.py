# -*- coding = utf-8 -*-

# @author:黑白
# @contact:1808132036@qq.com
# @time:19-1-2下午7:22
# @file:aa.py

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper



class ESIM(object):
    def __init__(self, args, word_vec, iter_num):
        self.args = args
        self._create_placeholder()

        # model operation
        self.logits = self._logits_op(word_vec)
        self.loss = self._loss_op()
        self.accuracy = self._acc_op()
        self.train_op = self._train_op(iter_num)

        tf.add_to_collection('train_mini', self.train_op)


    def _create_placeholder(self):
        self.sen1_lengths = tf.placeholder(tf.int32, [None])  # [batch_size]
        self.sen2_lengths = tf.placeholder(tf.int32, [None])  # [batch_size]
        self.truth = tf.placeholder(tf.int64, [None])  # [batch_size]
        self.sen1_words = tf.placeholder(tf.int32, [None, None])  # [batch_size, sentence1_len]
        self.sen2_words = tf.placeholder(tf.int32, [None, None])  # [batch_size, sentence1_len]

    def _create_feed_dict(self, sentences1, sentences2, sen1_len, sen2_len, truth):  # 此函数待定
        feed_dict = {
            self.sen1_lengths: sen1_len,
            self.sen2_lengths: sen2_len,
            self.truth: truth,
            self.sen1_words: sentences1,
            self.sen2_words: sentences2,
        }
        return feed_dict


    def _inputEmbeddingBlock(self, word_vec):
        with tf.device('/cpu:0'):
            self.word_embeddings = tf.get_variable('embedding', trainable=self.args.word_vec_trainable,
                                                   initializer=tf.constant(word_vec.word2vecs), dtype=tf.float32)
        self.sen1_emb = tf.nn.embedding_lookup(self.word_embeddings, self.sen1_words)
        self.sen2_emb = tf.nn.embedding_lookup(self.word_embeddings, self.sen2_words)


    def _biLSTMBlock(self, inputs, inputs_lenths, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            # cell_fw & cell_bw
            lstm_cell = LSTMCell(self.args.lstm_units)
            drop_lstm_cell = lambda: DropoutWrapper(lstm_cell, output_keep_prob=self.args.dropout_rate)
            lstm_cell_fw, lstm_cell_bw = drop_lstm_cell(), drop_lstm_cell()
            batch_size = tf.shape(inputs)[0]
            init_state_fw = lstm_cell_fw.zero_state(batch_size, tf.float32)
            init_state_bw = lstm_cell_bw.zero_state(batch_size, tf.float32)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw,
                                                         cell_bw=lstm_cell_bw,
                                                         inputs=inputs,
                                                         sequence_length=inputs_lenths,
                                                         initial_state_fw=init_state_fw,
                                                         initial_state_bw=init_state_bw,
                                                         dtype=tf.float32
                                                        )

            return tf.concat(outputs, axis=2)


    def _localInferenceBlock(self, sen1_repre, sen2_repre, scope):
        with tf.variable_scope(scope):
            similarity_matrix = tf.matmul(sen1_repre, tf.transpose(sen2_repre, [0, 2, 1])) #[batch_size, sen1maxlen, sen2maxlen]
            sen1_mask = tf.cast(tf.sequence_mask(self.sen1_lengths), tf.float32)  # [batch_size, sen1_max_len]
            sen2_mask = tf.cast(tf.sequence_mask(self.sen2_lengths), tf.float32)  # [batch_size, sen2_max_len]

            similarity_matrix = tf.multiply(similarity_matrix, tf.tile(tf.expand_dims(sen1_mask, axis=2), [1, 1, tf.shape(sen2_mask)[-1]]))
            similarity_matrix = tf.multiply(similarity_matrix, tf.tile(tf.expand_dims(sen2_mask, axis=1), [1, tf.shape(sen1_mask)[-1], 1]))

            sen1_att_repre = tf.matmul(tf.nn.softmax(similarity_matrix),
                                       sen2_repre)  # [batch_size, sen1_max_len, 2*rnn_units]
            sen2_att_repre = tf.matmul(tf.nn.softmax(tf.transpose(similarity_matrix, [0, 2, 1])),
                                       sen1_repre)  # [batch_size, sen2_max_len, 2*rnn_units]

            sen1_diff_repre = tf.subtract(sen1_repre, sen1_att_repre)
            sen2_diff_repre = tf.subtract(sen2_repre, sen2_att_repre)

            sen1_mul_repre = tf.multiply(sen1_repre, sen1_att_repre)
            sen2_mul_repre = tf.multiply(sen2_repre, sen2_att_repre)

            sen1_all_repre = tf.concat([sen1_repre, sen1_att_repre, sen1_diff_repre, sen1_mul_repre], axis=2)
            sen2_all_repre = tf.concat([sen2_repre, sen2_att_repre, sen2_diff_repre, sen2_mul_repre], axis=2)

            return sen1_all_repre, sen2_all_repre

    def _compositionBlock(self, sen1_inputs, sen2_inputs, scope):
        with tf.variable_scope(scope):
            sen1_outputs = self._biLSTMBlock(sen1_inputs, self.sen1_lengths, scope='biLSTM', reuse=False)
            sen2_outputs = self._biLSTMBlock(sen2_inputs, self.sen2_lengths, scope='biLSTM', reuse=True)

            sen1_outputs_avg = tf.reduce_mean(sen1_outputs, axis=1)
            sen2_outputs_avg = tf.reduce_mean(sen2_outputs, axis=1)
            sen1_outputs_max = tf.reduce_max(sen1_outputs, axis=1)
            sen2_outputs_max = tf.reduce_max(sen2_outputs, axis=1)

            merge_repre = tf.concat([sen1_outputs_avg, sen1_outputs_max, sen2_outputs_avg, sen2_outputs_max], axis=1)

            return merge_repre

    def _feedForwardBlock(self, inputs, scope):
        """
        :param inputs: shape[batch_size, 4*2*args.lstm_units]
        :return:
        """
        with tf.variable_scope(scope):
            initializer = tf.random_normal_initializer(0.0, 0.1)

            with tf.variable_scope('feed_forward_layer1'):
                inputs = tf.nn.dropout(inputs, self.args.dropout_rate)
                outputs = tf.layers.dense(inputs, 256, tf.nn.relu, kernel_initializer=initializer)
            with tf.variable_scope('feed_forward_layer2'):
                outputs = tf.nn.dropout(outputs, self.args.dropout_rate)
                results = tf.layers.dense(outputs, 2, tf.nn.tanh, kernel_initializer=initializer)
                return results



    def _logits_op(self, word_vec):
        print('============= sentence embedding start ====================')
        self._inputEmbeddingBlock(word_vec)
        print('============= sentence embedding end ======================')

        print("============  BiLSTM Start ======================")
        sen1_repre = self._biLSTMBlock(self.sen1_emb, self.sen1_lengths, 'biLSTM', reuse=False)
        sen2_repre = self._biLSTMBlock(self.sen2_emb, self.sen2_lengths, 'biLSTM', reuse=True)
        print("============  BiLSTM End ======================")

        print('============== local inference start =================')
        sen1_outputs, sen2_outputs = self._localInferenceBlock(sen1_repre, sen2_repre, scope='local_inference')
        print('============== local inference end =================')

        print('============== merge layer start  =====================')
        mergre_repre = self._compositionBlock(sen1_outputs, sen2_outputs, 'merger')
        print('============== merge layer start=====================')

        print('============== feed forward layer start  ==================')
        logits = self._feedForwardBlock(mergre_repre, 'feed_forward')
        print('============== feed forward layer end  ==================')
        return logits

    def _loss_op(self, l2_lambda=0.001):
        with tf.variable_scope('cost'):
            # losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.truth, logits=self.logits)
            # loss = tf.reduce_mean(losses, name='loss_val')
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.truth, logits=self.logits))
            weights = [v for v in tf.trainable_variables() if('w' in v.name) or ('kernel' in v.name)]
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * l2_lambda
            loss += l2_loss
            return loss

    def _acc_op(self):
        with tf.variable_scope('acc'):
            predict = tf.equal(tf.argmax(self.logits, axis=1), self.truth)
            accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))
            # label_pred = tf.argmax(self.logits, 1, name='label_pred')
            # label_true = tf.argmax(self.truth, 1, name='label_truth')
            # corrected_pred = tf.equal(tf.cast(label_pred, tf.int32), tf.cast(label_true, tf.int32))
            # accuracy = tf.reduce_mean(tf.cast(corrected_pred, tf.float32), name='Accuracy')
            return accuracy

    def _train_op(self, iter_num):
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
            train_op = optimizer.minimize(self.loss)
            return train_op


