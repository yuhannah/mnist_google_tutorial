# encoding: UTF-8
#
# train3: 有 5 层的密集神经网络
#
# 将图像平铺成一维向量，全连接下一隐藏层，激活函数为 ReLU，输出层 softmax
#
# X = [batch, 784], W1 = [784, 200], b1 = [200], W2 = [200, 100], b2 = [100],
# W3 = [100, 60], b3 = [60], W4 = [60, 30], b4 = [30], W5 = [30, 10], b5 = [10], Y = [batch, 10]
#
# tag: GD, Adam, minibatch, softmax

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mnist import input_data
import math

# import time


print("Tensorflow version " + tf.__version__)
tf.set_random_seed(1)


def train2(layers, learning_rate=0.005, minibatch_size=100, iterations=2000 + 1):
    # Use local mnist dataset (60k for train, 10k for test)
    mnist = input_data.read_data_sets("mnist", one_hot=True, reshape=False)

    # input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
    X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="x_placeholder")
    # correct answers will go here
    Y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_placeholder")
    # step for variable learning rate
    iter = tf.placeholder(tf.int32)
    # # Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
    # pkeep = tf.placeholder(tf.float32)
    # train/test selector for batch normalisation
    tst = tf.placeholder(tf.bool)

    # five layers and their number of neurons (tha last layer has 10 softmax neurons)
    L = 200
    M = 100
    N = 60
    P = 30
    Q = 10

    W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1))  # 784 = 28 * 28
    S1 = tf.Variable(tf.ones([L]))
    O1 = tf.Variable(tf.zeros([L]))
    W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
    S2 = tf.Variable(tf.ones([M]))
    O2 = tf.Variable(tf.zeros([M]))
    W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
    S3 = tf.Variable(tf.ones([N]))
    O3 = tf.Variable(tf.zeros([N]))
    W4 = tf.Variable(tf.truncated_normal([N, P], stddev=0.1))
    S4 = tf.Variable(tf.ones([P]))
    O4 = tf.Variable(tf.zeros([P]))
    W5 = tf.Variable(tf.truncated_normal([P, Q], stddev=0.1))
    B5 = tf.Variable(tf.zeros([Q]))

    def batchnorm(Ylogits, Offset, Scale, is_test, iteration):
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.998,
                                                           iteration)  # adding the iteration prevents from averaging across non-existing iterations
        bnepsilon = 1e-5
        mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_averages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, Offset, Scale, bnepsilon)
        return Ybn, update_moving_averages

    def no_batchnorm(Ylogits, Offset, Scale, is_test, iteration):
        return Ylogits, tf.no_op()

    # The model
    XX = tf.reshape(X, [-1, 784])

    Y1l = tf.matmul(XX, W1)
    Y1bn, update_ema1 = batchnorm(Y1l, O1, S1, tst, iter)
    Y1 = tf.nn.sigmoid(Y1bn)

    Y2l = tf.matmul(Y1, W2)
    Y2bn, update_ema2 = batchnorm(Y2l, O2, S2, tst, iter)
    Y2 = tf.nn.sigmoid(Y2bn)

    Y3l = tf.matmul(Y2, W3)
    Y3bn, update_ema3 = batchnorm(Y3l, O3, S3, tst, iter)
    Y3 = tf.nn.sigmoid(Y3bn)

    Y4l = tf.matmul(Y3, W4)
    Y4bn, update_ema4 = batchnorm(Y4l, O4, S4, tst, iter)
    Y4 = tf.nn.sigmoid(Y4bn)

    Ylogits = tf.matmul(Y4, W5) + B5
    Y = tf.nn.softmax(Ylogits)

    update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4)

    # Loss is defined as cross entropy between the prediction and the real value
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_,
                                                            name="lossFunction")
    cross_entropy = tf.reduce_mean(cross_entropy) * minibatch_size

    # accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

    # training, learning rate = 0.005
    # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, name="gradDescent")
    # train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, name="adam")
    # training step
    # the learning rate is: # 0.0001 + 0.003 * (1/e)^(step/2000)), i.e. exponential decay from 0.003->0.0001
    lr = 0.0001 + tf.train.exponential_decay(0.003, iter, 2000, 1 / math.e)
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    # init
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # save data
    train_indexes = []
    train_costs = []
    train_accuracies = []
    test_indexes = []
    test_costs = []
    test_accuracies = []

    # Feed the next batch and run the training
    for i in range(iterations):
        # training on batches of 100 images with 100 labels
        batch_X, batch_Y = mnist.train.next_batch(minibatch_size)

        # compute training values
        if i % 10 == 0:
            acc, cost = sess.run([accuracy, cross_entropy], feed_dict={X: batch_X, Y_: batch_Y, iter: i, tst: False})
            train_indexes.append(i)
            train_costs.append(cost)
            train_accuracies.append(acc)
            print(str(i) + ": accuracy:" + str(acc) + " loss: " + str(cost))

        # compute test values
        if i % 50 == 0:
            acc, cost = sess.run([accuracy, cross_entropy],
                                 feed_dict={X: mnist.test.images, Y_: mnist.test.labels, tst: True})
            test_indexes.append(i)
            test_costs.append(cost)
            test_accuracies.append(acc)
            print(str(i) + ": ********* epoch " + str(
                i * minibatch_size // mnist.train.images.shape[0] + 1) + " ********* test accuracy:" + str(
                acc) + " test loss: " + str(cost))

        # the backpropagation training step
        sess.run([train_step, update_ema], feed_dict={X: batch_X, Y_: batch_Y, iter: i, tst: False})

    print("max test accuracy: " + str(max(test_accuracies)))

    lrate = sess.run(lr, feed_dict={X: mnist.test.images, Y_: mnist.test.labels, iter: iterations - 1, tst: False})
    # plot train and test costs and accuracies
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(np.squeeze(train_indexes), np.squeeze(train_accuracies), label="train_accuracy")
    plt.plot(np.squeeze(test_indexes), np.squeeze(test_accuracies), label="test_accuracy")
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(lrate))
    plt.subplot(122)
    plt.plot(np.squeeze(train_indexes), np.squeeze(train_costs), label="train_costs")
    plt.plot(np.squeeze(test_indexes), np.squeeze(test_costs), label="test_costs")
    plt.legend()
    plt.ylabel('costs')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(lrate))
    # plt.show()
    plt.savefig(
        "output/learning_rate " + str(lrate) + " iterations " + str(iterations) + " bn max test accuracy " + str(
            max(test_accuracies)) + " .png")

    sess.close()


if __name__ == "__main__":
    # for learning_rate in [0.5, 0.05, 0.005]:
    layers = [784, 200, 100, 60, 30, 10]
    # 开始时间
    # start_time = time.clock()
    train2(layers, learning_rate=0.005, minibatch_size=100, iterations=10000 + 1)
    # 结束时间
    # end_time = time.clock()
    # 计算时差
    # print("CPU的执行时间 = " + str(end_time - start_time) + " 秒")