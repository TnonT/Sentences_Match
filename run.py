#coding = utf-8

from __future__ import print_function
from utils import Vocab, DataStream
from model.LSTM import LSTM
from model.BIdirection_Attention import LSTM_Attention
from model.ESIM import ESIM_ATT
import numpy as np
import tensorflow as tf
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import datetime
import os

whole_padding = False




def train(sess, args, model, data, epoch):
    # if not ps.path.exist(args.vocab_path):
    #     os.mkdir(args.vocab_path)
    total_loss = 0
    total_acc = 0
    iter_num = 0

    if whole_padding:
        print("load numpy data....")
        question1 = np.load('../data/quora/question1.npy')
        question2 = np.load('../data/quora/question2.npy')
        q1_length = np.load('../data/quora/q1_length.npy')
        q2_length = np.load('../data/quora/q2_length.npy')
        labels = np.load('../data/quora/labels.npy')
        print("end load numpy data....")

        pbar = tqdm(range(3000))
        for i in pbar:
        # for i in range(3000):
            pbar.set_description("Epoch {} ".format(epoch + 1))
            start_index = i * 64
            end_index = start_index + 64

            feed_dict = model.create_feed_dict(question1[start_index:end_index], question2[start_index: end_index],
                            q1_length[start_index:end_index], q2_length[start_index:end_index], labels[start_index:end_index])
            _, loss = sess.run([model.train_op ,model.loss], feed_dict=feed_dict)
            total_loss += loss

    else:
        iter_num = data.train_num // args.batch_size # =6190
        ptbr = tqdm(range(iter_num))
        for batch_num in ptbr:

            ptbr.set_description("Epoch {} ".format(epoch + 1))
            sentences1, sentences2, sen1_len, sen2_len, truth = data.get_batch(batch_num, data.train_question1,
                                                                                data.train_question2, data.train_labels)
            feed_dict = model.create_feed_dict(sentences1, sentences2, sen1_len, sen2_len, truth)

            _, loss, acc = sess.run([model.train_op, model.loss, model.accuracy], feed_dict=feed_dict)

            total_loss += loss
            total_acc += acc
            # tf.summary.scalar('Loss', loss)
            # tf.summary.scalar('Accuracy', acc)
            # merged = tf.summary.merge_all()
            # logdir = '../tensorboard/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'
            # writer = tf.summary.FileWriter(logdir, sess.graph)
            # writer.add_summary()


    return total_loss / iter_num, total_acc / iter_num




def main(args):
    # Loading word2vec, build vocabulary
    vocab = Vocab(args.embedding_type, args.embedding_path)
    if whole_padding:
        data = None
    else:
        print("Loading Data........")
        data = DataStream(args, vocab)
        print("End Loading.........")

    # for
    train_losses = []
    train_accs = []
    dev_losses = []
    dev_accs = []



    #定义初始化函数 tensorflow v1.8
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    print("初始化模型...")
    # train model
    with tf.name_scope('training'):
        with tf.variable_scope('model', reuse=None, initializer=initializer):
            # train_model = LSTM(args, vocab, data.train_num, True)
            train_model = LSTM_Attention(args, vocab, data.train_num, True)
            # train_model = ESIM_ATT(args, vocab, data.train_num, True)

    # eval model
    with tf.name_scope('eval'):
        with tf.variable_scope('model', reuse=True, initializer=initializer):
            # dev_model = LSTM(args, vocab, data.train_num, False)
            dev_model = LSTM_Attention(args, vocab, data.train_num, False)
            # dev_model = ESIM_ATT(args, vocab, data.train_num, False)

    print("结束初始化...")

    # # saver
    # saver = tf.train.Saver()

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        tf.global_variables_initializer().run()

        # Dev
        sen1_dev, sen2_dev, sen1_dev_len, sen2_dev_len, dev_truth = data.get_dev_data()
        print('sen1_dev_len max_len is {}.'.format(max(sen1_dev_len)))
        print('sen2_dev_len max_len is {}.'.format(max(sen2_dev_len)))
        feed_dict = dev_model.create_feed_dict(sen1_dev, sen2_dev, sen1_dev_len, sen2_dev_len, dev_truth)

        dev_loss, dev_acc = sess.run([dev_model.loss, dev_model.accuracy], feed_dict=feed_dict)
        print("Before training, Dev loss is {}".format(dev_loss))
        print("Before training, Dev acc is {}".format(dev_acc))

        for epoch in range(args.num_epoch):
            train_loss, train_acc = train(sess, args, train_model, data, epoch)
            # print('train loss is {}.'.format(train_loss))
            # print('train acc is {}.'.format(train_acc))

            print(f'Train loss is {train_loss:10.5f}')
            print(f'Train acc is {train_acc:10.4f}')

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            # # Save the model
            # saver.save(sess, '../ckpt/model.ckpt')

            # Dev
            sen1_dev, sen2_dev, sen1_dev_len, sen2_dev_len, dev_truth = data.get_dev_data()
            feed_dict = dev_model.create_feed_dict (sen1_dev, sen2_dev, sen1_dev_len, sen2_dev_len, dev_truth)

            dev_loss, dev_acc = sess.run([dev_model.loss, dev_model.accuracy], feed_dict=feed_dict)
            # print("Dev loss is {}".format(dev_loss))
            # print("Dev acc is {}".format(dev_acc))
            print(f'Dev loss is {dev_loss:10.5f}')
            print(f'Dev acc is {dev_acc:10.4f}')
            dev_losses.append(dev_loss)
            dev_accs.append(dev_acc)

        # Test
        sen1_test, sen2_test, sen1_test_len, sen2_test_len, test_truth = data.get_test_data()
        feed_dict = dev_model.create_feed_dict(sen1_test, sen2_test, sen1_test_len, sen2_test_len, test_truth)
        test_loss, test_acc = sess.run([dev_model.loss, dev_model.accuracy], feed_dict=feed_dict)
        # print("Test loss is {}".format(test_loss))
        # print("Test acc is {}".format(test_acc))
        print(f'Test loss is {test_loss:10.5f}')
        print(f'Test acc is {test_acc:10.4f}')

        # mal
        plt.title("learn curve")
        epochs = range(args.num_epoch)
        plt.plot(epochs, train_losses, label='train_loss')
        plt.plot(epochs, dev_losses, label='test_loss')

        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("loss")

        plt.show() 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', type=str, default='data/quora/train.csv', help='Training data path!')
    parser.add_argument('--dev_ratio', type=float, default=0.05, help='dev ratio!')
    parser.add_argument('--test_ratio', type=float, default=0.05, help='test ratio!')

    # parser.add_argument('--vocab_path', type=str, help='vocabulary path') # No need

    parser.add_argument('--embedding_type', type=str, default='glove', help='glove or word2vector')
    parser.add_argument('--embedding_path', type=str, default='embedding/glove.6B.300d.txt', help='word embedding!')
    parser.add_argument('--log_dir', type=str, default='log/', help='log dir')

    parser.add_argument('--batch_size', type=int, default=64, help='batch size!')

    parser.add_argument('--word_vec_trainable', type=bool, default=True, help='whether word embedding can be trained')
    parser.add_argument('--num_epoch', type=int, default=30, help='epochs')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='learning rate!')
    parser.add_argument('--learning_rate_decay', type=float, default=0.8, help='dropout rate!')
    parser.add_argument('--dropout_rate', type=float, default=0.25, help='dropout rate!')

    parser.add_argument('--lstm_units', type=int, default=64, help='lstm units')

    # ====== multi-perspective-rnn-att
    parser.add_argument('--multi_perspective', type=int, default=3, help='num of perspective')

    args = parser.parse_args()


    # run
    main(args)
