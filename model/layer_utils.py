# coding = utf-8

import tensorflow as tf

eps = 1e-6
def cosine_distance(y1, y2):
    # y1 = [... , a, 1, d]
    # y2 = [... , 1, a, d]
    cosine_numberator = tf.reduce_sum(tf.multiply(y1, y2), axis=-1)
    y1_norm = tf.sqrt(tf.reduce_max(tf.reduce_sum(tf.square(y1), axis=-1), eps))
    y2_norm = tf.sqrt(tf.reduce_max(tf.reduce_sum(tf.square(y2), axis=-1), eps))
    return cosine_numberator / y1_norm / y2_norm

def cal_relevancy_matrix(sen1, sen2): # the shape of both are all [batch_size, sen_len, feature_dim].
    sen1_repre_tmp = tf.expand_dims(sen1, axis=1) # [batch_size, 1, sen1_len, feature_dim]
    sen2_repre_tmp = tf.expand_dims(sen2, axis=2) # [batch_size, sen1_len, 1, feature_dim]
    relevancy_matrix = cosine_distance(sen1_repre_tmp, sen2_repre_tmp)  # [batch_size, sen1_len, sen2_len]
    return relevancy_matrix

def mask_relevancy_matrix(relevancy_matrix, sen1_mask, sen2_mask):
    # relevancy_matrix: [batch_size, passage_len, question_len]
    # sen1_mask: [batch_size, sen1_len]
    # sen2_mask: [batch_size, sen2_len]
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(sen2_mask, axis=1))
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(sen1_mask, axis=2))
    return relevancy_matrix

def multi_perspective_match(feature_dim, repres1, repres2, is_training=True, dropout_rate=0.2, options=None,
                            scope_name='mp-match', reuse=False):
    '''

    :param repres1: [batch_size, len, feature_dim]
    :param repres2: [batch_size, len, feature_dim]
    :return:
    '''
    innput_shape = tf.shape(repres1)


def macth_sen_with_sen(sen1, sen2, sen1_len, sen2_len, sen1_mask, sen2_mask, input_dim, scope='word_match_forward',
                       with_full_match=True,
                       ):
    sen1 = tf.multiply(sen1, tf.expand_dims(sen1_mask, axis=-1))  # [batch_size, sen1_len, feature_dim]
    sen2 = tf.multiply(sen2, tf.expand_dims(sen2_mask, axis=-1))  # [batch_size, sen1_len, feature_dim]
    all_sen1_aware_rep = []
    dim = 0

    with tf.variable_scope(scope):
        relevancy_matrix = cal_relevancy_matrix(sen1, sen2) # [batch_size, passage_len, question_len]
        relevancy_matrix = mask_relevancy_matrix(relevancy_matrix, sen1_mask, sen2_mask)
        all_sen1_aware_rep.append(tf.reduce_max(relevancy_matrix, axis=2, keep_dims=True))
        dim += 2

        if with_full_match:







def bilateral_match_func(sen1, sen2, sen1_len, sen2_len, sen1_mask, sen2_mask, input_dim, is_training):
    sen1_aware_representation = []
    sen1_aware_dim = 0
    sen2_aware_representation = []
    sen2_aware_dim = 0

    # ====  Match sen1 with sen2  ======



