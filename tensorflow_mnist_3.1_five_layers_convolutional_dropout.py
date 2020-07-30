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
    step = tf.placeholder(tf.int32)
    # Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
    pkeep = tf.placeholder(tf.float32)

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
    Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
    stride = 2  # output is 14x14
    Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
    stride = 2  # output is 7x7
    Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

    # reshape the output from the third convolution for the fully connected layer
    YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

    Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
    YY4 = tf.nn.dropout(Y4, pkeep)
    Ylogits = tf.matmul(YY4, W5) + B5
    Y = tf.nn.softmax(Ylogits)

    # Loss is defined as cross entropy between the prediction and the real value
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_,
                                                            name="lossFunction")
    cross_entropy = tf.reduce_mean(cross_entropy) * minibatch_size

    # accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

    # training step
    # the learning rate is: # 0.0001 + 0.003 * (1/e)^(step/2000)), i.e. exponential decay from 0.003->0.0001
    lr = 0.0001 + tf.train.exponential_decay(0.003, step, 2000, 1 / math.e)
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
            acc, cost = sess.run([accuracy, cross_entropy], feed_dict={X: batch_X, Y_: batch_Y, pkeep: 1.0, step: i})
            train_indexes.append(i)
            train_costs.append(cost)
            train_accuracies.append(acc)
            print(str(i) + ": accuracy:" + str(acc) + " loss: " + str(cost))

        # compute test values
        if i % 50 == 0:
            acc, cost = sess.run([accuracy, cross_entropy],
                                 feed_dict={X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1.0})
            test_indexes.append(i)
            test_costs.append(cost)
            test_accuracies.append(acc)
            print(str(i) + ": ********* epoch " + str(
                i * minibatch_size // mnist.train.images.shape[0] + 1) + " ********* test accuracy:" + str(
                acc) + " test loss: " + str(cost))

        # the backpropagation training step
        sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y, pkeep: 0.75, step: i})

    print("max test accuracy: " + str(max(test_accuracies)))

    lrate = sess.run(lr, feed_dict={X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1.0, step: iterations - 1})
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
    outfig = "output/learning_rate " + str(lrate) + " iterations " + str(iterations) + " dropout max test accuracy " + str(
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
