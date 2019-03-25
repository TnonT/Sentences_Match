#! -*- coding=utf-8 -*-

from ..utils.utils_transformer import get_token_embeddings, multi_head_attention, ff
import tensorflow as tf

class Transformer(object):
    def __init__(self, args, word_vec):
        self.args = args
        self.vocab_size = len(word_vec)



    def _create_placeholder(self):
        self.sen1 = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sen1_word')
        self.sen2 = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sen2_word')
        self.sen1_len = tf.placeholder(dtype=tf.int32, shape=[None], name='sen1_len')
        self.sen2_len = tf.placeholder(dtype=tf.int32, shape=[None], name='sen2_len')
        self.truth = tf.placeholder(dtype=tf.int64, shape=[None], name='ground truth')

    def _create_feedd_ict(self, sentence1, sentence2, sen1_len, sen2_len, truth):
        feeddict = {
            self.sen1: sentence1,
            self.sen2: sentence2,
            self.sen1_len: sen1_len,
            self.sen2_len: sen2_len,
            self.truth: truth
        }
        return feeddict

    def _position_embedding(self, inputs, position_size):
        """

        :param inputs: [batch_szie, seq_len, word_size]
        :param position_size:
        :return: position_embedding: [batch_size, seq_len, position_size]
        """
        batch_size, seq_len = tf.shape(inputs)[0], tf.shape(inpits)[1]
        position_j = 1. / tf.pow(10000., \
                                 2 * tf.range(position_size / 2, dtype=tf.float32 \
                                              ) / position_size)
        position_j = tf.expand_dims(position_j, 0)
        position_i = tf.range(tf.cast(seq_len, tf.float32), dtype=tf.float32)
        position_i = tf.expand_dims(position_i, 1)
        position_ij = tf.matmul(position_i, position_j)
        position_ij = tf.concat([tf.cos(position_ij), tf.sin(position_ij)], 1)
        position_embedding = tf.expand_dims(position_ij, 0) \
                             + tf.zeros((batch_size, seq_len, position_size))
        return position_embedding

    def _Dense(self, inputs, outputs_size):
        inputs_size = int(inputs.shape[-1])
        W = tf.get_variable(name='w', shape=[inputs_size, outputs_size], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
        outputs = tf.matmul(tf.reshape(inputs, [-1, inputs_size]), W)
        outputs = tf.reshape(outputs, tf.concat(tf.shape(inputs)[:-1], [outputs_size], axis=0))
        return outputs


    def _self_attention(self, Q, K, V, nb_head, size_per_head, Q_len=None, V_len=None):
        Q = self._Dense(Q, nb_head * size_per_head)
        Q = tf.reshape(Q, (-1, tf.shape(Q)[1], nb_head, size_per_head))
        Q = tf.transpose(Q, [0, 2, 1, 3])

        K = self._Dense(K, nb_head * size_per_head)
        K = tf.reshape(K, (-1, tf.shape(K)[1], nb_head, size_per_head))
        K = tf.transpose(K, [0, 2, 1, 3])

        V = self._Dense(V, nb_head * size_per_head)
        V = tf.reshape(V, (-1, tf.shape(V)[1], nb_head, size_per_head))
        V = tf.transpose(V, [0, 2, 1, 3])

        # Attention
        A = tf.matmul(Q, K, transpose_b=True)
        A = tf.nn.softmax(A)

        O = tf.matmul(A, V)
        O = tf.transpose(O, [0, 2, 1, 3])
        O = tr.reshape(O, [-1, tf.shape(O)[1], nb_head*size_per_head])
        return O





    def _transformer_enc(self, inputs, vocab_size, num_units, training=True):
        """

        :param inputs:
        :param word_vec:
        :return: [batch_size, sen_len, 512]
        """
        # Embedding
        embeddings = tf.get_variable("word_embeding",
                                     dtype=tf.float32,
                                     shape=(vocab_size, num_units),
                                     initializer=tf.contrib.layers.xavier_initializer())
        sen_word_emb = tf.nn.embedding_lookup(embeddings, inputs)
        sen_pos_emb = self._position_embedding(sen_word_emb, self.args.position_size)
        sen_emb = sen_word_emb + sen_pos_emb
        sen_emb = tf.layers.dropout(sen_emb, self.args.dropout_rate, training=training)

        # Block
        for i in range(self.args.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                # self-attention
                sen_emb = multi_head_attention(queries=sen_emb,
                                               keys=sen_emb,
                                               values=sen_emb,
                                               num_heads=self.args.num_heads,
                                               dropout_rate=self.args.dropout_rate,
                                               training=training,
                                               causality=False)
                # feed forward
                sen_emb = ff(sen_emb, num_units)
        return sen_emb



    def _create_model_graph(self, word_vec, traing=True):
        # Transformer Encoder
        vocab_size = len(word_vec)
        num_units = [self.args.d_ff, self.args.d_model]
        with tf.name_scope("transformer encoder"):
            sen1 = self._transformer_enc(self.sen1, vocab_size, num_units, training)  # [b_s, sen1_len, 512]
            sen2 = self._transformer_enc(self.sen2, vocab_size, num_units, training)

        #





