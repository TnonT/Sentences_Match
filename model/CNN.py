# -*- coding = utf-8 -*-

# @author:黑白
# @contact:1808132036@qq.com
# @time:19-1-10下午3:43
# @file:CNN.py


###################################################################################

需要将所有句子的长度处理到相同长度

##################################################################################

import tensorflow as tf

class CNN_ATT(object):
    def __init__(self, args, word_vec):
        self.args = args

        self._create_model_graph(word_vec)

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

    def _Conv1D(self, inputs, filters, kerner_size, strides, activation, use_bias=True, scope=None):
        """

        :param inputs: with shape [bacth_size, sen_len, dim]
        :param filters:
        :param kerner_size:
        :param strides:
        :param activation:
        :param use_bias:
        :param scope:
        :return: conv1d : with shape [batch_size, sen_len-kernerl_size+1, filter]
        """
        with tf.variable_scope(scope):
            conv1d = tf.layers.conv1d(inputs=inputs,
                                      filters=filters,
                                      kernel_size=kerner_size,
                                      strides=strides,
                                      activation=activation,
                                      use_bias=use_bias)

            return conv1d

    def _Max_Pool1D(self, inputs, pool_size, strides, padding='valid', scope=None):
        """

        :param inputs: [batch_size, len, filter]
        :param pool_size:
        :param strides:
        :param padding:
        :param scope:
        :return: maxpool1d : [batch_size, ]
        """
        with tf.variable_scope(scope):
            maxpool1d = tf.layers.max_pooling1d(inputs=inputs,
                                                pool_size=pool_size,
                                                strides=strides,
                                                padding=padding)
            return maxpool1d

    def _create_model_graph(self, word_vec):
        # Embedding
        embedding = tf.get_variable(name='embedding', dtype=tf.float32, initializer=tf.constant(word_vec.word2vecs))
        sen1_emb = tf.nn.embedding_lookup(embedding, self.sen1_words)
        sen2_emb = tf.nn.embedding_lookup(embedding, self.sen2_words)

        # conv1
        sen1_conv1 = self._Conv1D(sen1_emb, filters=5, kerner_size=2, strides=1, activation=tf.nn.relu, use_bias=True, scope='sen1_conv1')
        sen2_conv1 = self._Conv1D(sen2_emb, filters=5, kerner_size=2, strides=1, activation=tf.nn.relu, use_bias=True, scope='sen2_conv1')

        # pool1
        sen1_maxpool1 = self._Max_Pool1D(inputs=sen1_conv1, pool_size=2, strides=1, scope='sen1_pool1')
        sen2_maxpool1 = self._Max_Pool1D(inputs=sen2_conv1, pool_size=2, strides=1, scope='sen2_pool1')

        # conv2
        sen1_conv2 = self._Conv1D(sen1_maxpool1, filters=3, kerner_size=2, strides=1, activation=tf.nn.relu, use_bias=True, scope='sen1_conv2')
        sen2_conv2 = self._Conv1D(sen2_maxpool1, filters=3, kerner_size=2, strides=1, activation=tf.nn.relu, use_bias=True, scope='sen2_conv2')

        # pool2
        sen1_maxpool2 = self._Max_Pool1D(inputs=sen1_conv2, pool_size=2, strides=2, scope='sen1_pool2')
        sen2_maxpool2 = self._Max_Pool1D(inputs=sen2_conv2, pool_size=2, strides=2, scope='sen2_pool2')

        # conv3
        sen1_conv3 = self._Conv1D(sen1_maxpool2, filters=3, kerner_size=3, strides=1, activation=tf.nn.relu, use_bias=True, scope='sen1_conv3')
        sen2_conv3 = self._Conv1D(sen2_maxpool2, filters=3, kerner_size=3, strides=1, activation=tf.nn.relu, use_bias=True, scope='sen2_conv3')

        # pool3
        sen1_maxpool3 = self._Max_Pool1D(inputs=sen1_conv3, pool_size=2, strides=2, scope='sen1_pool3')
        sen2_maxpool3 = self._Max_Pool1D(inputs=sen2_conv3, pool_size=2, strides=2, scope='sen2_pool3')

