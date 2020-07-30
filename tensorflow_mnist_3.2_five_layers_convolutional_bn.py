# encoding: UTF-8
#
# train4: 有 5 层的卷积神经网络
#
# 将图像平铺成一维向量，全连接下一隐藏层，激活函数为 ReLU，输出层 softmax
#
# X = [batch, 28, 28, 1], W1 = [5, 5, 1, 6], b1 = [6], W2 = [5, 5, 4, 12], b2 = [12],
# W3 = [4, 4, 8, 24], b3 = [24], W4 = [7*7*12, 200], b4 = [200], W5 = [200, 10], b5 = [10], Y = [batch, 10]
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


def train4(minibatch_size=100, iterations=2000 + 1):
    # Use local mnist dataset (60k for train, 10k for test)
    mnist = input_data.read_data_sets("mnist", one_hot=True, reshape=False)

    # input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
    X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="x_placeholder")
    # correct answers will go here
    Y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_placeholder")
    # step for variable learning rate
    iter = tf.placeholder(tf.int32)
    # Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
    pkeep = tf.placeholder(tf.float32)
    pkeep_conv = tf.placeholder(tf.float32)
    # test flag for batch norm
    tst = tf.placeholder(tf.bool)

    def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999,
                                                           iteration)  # adding the iteration prevents from averaging across non-existing iterations
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_averages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_averages

    def no_batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
        return Ylogits, tf.no_op()

    def compatible_convolutional_noise_shape(Y):
        noiseshape = tf.shape(Y)
        noiseshape = noiseshape * tf.constant([1, 0, 0, 1]) + tf.constant([0, 1, 1, 0])
        return noiseshape

    # three convolutional layers with their channel counts, and a
    # fully connected layer (tha last layer has 10 softmax neurons)
    K = 6  # first convolutional layer output depth
    L = 12  # second convolutional layer output depth
    M = 24  # third convolutional layer
    N = 200  # fully connected layer

    W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))  # 5x5 patch, 1 input channel, K output channels
    B1 = tf.Variable(tf.ones([K]) / 10)
    W2 = tf.Variable(tf.truncated_normal([4, 4, K, L], stddev=0.1))
    B2 = tf.Variable(tf.ones([L]) / 10)
    W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
    B3 = tf.Variable(tf.ones([M]) / 10)

    W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
    B4 = tf.Variable(tf.ones([N]) / 10)
    W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
    B5 = tf.Variable(tf.ones([10]) / 10)

    # The model
    stride = 1  # output is 28x28
    Y1l = tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME')
    Y1bn, update_ema1 = batchnorm(Y1l, tst, iter, B1, convolutional=True)
    Y1r = tf.nn.relu(Y1bn)
    Y1 = tf.nn.dropout(Y1r, pkeep_conv)#, compatible_convolutional_noise_shape(Y1r))
    stride = 2  # output is 14x14
    Y2l = tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME')
    Y2bn, update_ema2=batchnorm(Y2l,tst,iter,B2,convolutional=True)
    Y2r=tf.nn.relu(Y2bn)
    Y2=tf.nn.dropout(Y2r,pkeep_conv)
    stride = 2  # output is 7x7
    Y3l=tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME')
    Y3bn,update_ema3=batchnorm(Y3l,tst,iter,B3,convolutional=True)
    Y3r=tf.nn.relu(Y3bn)
    Y3=tf.nn.dropout(Y3r,pkeep_conv)

    # reshape the output from the third convolution for the fully connected layer
    YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

    Y4l=tf.matmul(YY, W4)
    Y4bn,update_ema4=batchnorm(Y4l,tst,iter,B4)
    Y4r=tf.nn.relu(Y4bn)
    Y4 = tf.nn.dropout(Y4r, pkeep)
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
            acc, cost = sess.run([accuracy, cross_entropy], feed_dict={X: batch_X, Y_: batch_Y, iter: i, tst: False, pkeep: 1.0, pkeep_conv: 1.0})
            train_indexes.append(i)
            train_costs.append(cost)
            train_accuracies.append(acc)
            print(str(i) + ": accuracy:" + str(acc) + " loss: " + str(cost))

        # compute test values
        if i % 50 == 0:
            acc, cost = sess.run([accuracy, cross_entropy],
                                 feed_dict={X: mnist.test.images, Y_: mnist.test.labels, tst: True, pkeep: 1.0, pkeep_conv: 1.0})
            test_indexes.append(i)
            test_costs.append(cost)
            test_accuracies.append(acc)
            print(str(i) + ": ********* epoch " + str(
                i * minibatch_size // mnist.train.images.shape[0] + 1) + " ********* test accuracy:" + str(
                acc) + " test loss: " + str(cost))

        # the backpropagation training step
        sess.run(train_step, {X: batch_X, Y_: batch_Y, tst: False, iter: i, pkeep: 0.75, pkeep_conv: 1.0})
        sess.run(update_ema, {X: batch_X, Y_: batch_Y, tst: False, iter: i, pkeep: 1.0, pkeep_conv: 1.0})

    print("max test accuracy: " + str(max(test_accuracies)))

    lrate = sess.run(lr, feed_dict={X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1.0, iter: iterations - 1})
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
    outfig = "output/learning_rate " + str(lrate) + " iterations " + str(iterations) + " bn dropout max test accuracy " + str(
        max(test_accuracies)) + " .png"
    plt.savefig(outfig)
    print("out figure's name: ", outfig)

    sess.close()


if __name__ == "__main__":
    # 开始时间
    # start_time = time.clock()
    train4(minibatch_size=100, iterations=10000 + 1)
    # 结束时间
    # end_time = time.clock()
    # 计算时差
    # print("CPU的执行时间 = " + str(end_time - start_time) + " 秒")
