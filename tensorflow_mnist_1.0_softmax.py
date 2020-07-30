# encoding: UTF-8
#
# train1: 有 10 个单元的单层神经网络
#
# 将图像平铺成一维向量，全连接 softmax 的 10 个单元
#
# X = [batch, 784], W = [784, 10], b = [10], Y = [batch, 10]
#
# tag: GD, minibatch, softmax, learning_rate

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mnist import input_data

print("Tensorflow version " + tf.__version__)
tf.set_random_seed(1)


def train1(learning_rate=0.005, minibatch_size=100, iterations=2000 + 1):
    # Use local mnist dataset (60k for train, 10k for test)
    mnist = input_data.read_data_sets("mnist", one_hot=True, reshape=False)

    # input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
    X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="x_placeholder")
    # correct answers will go here
    Y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_placeholder")

    # weights W[784, 10]   784=28*28
    W = tf.Variable(tf.zeros([784, 10]), name="weights_variable")
    # biases b[10]
    b = tf.Variable(tf.zeros([10]), name="bias_variable")

    # flatten
    XX = tf.reshape(X, shape=[-1, 784])

    # The model
    Ylogits = tf.matmul(XX, W) + b
    Y = tf.nn.softmax(Ylogits)

    # Loss is defined as cross entropy between the prediction and the real value
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_, name="lossFunction")
    cross_entropy = tf.reduce_mean(cross_entropy) * minibatch_size

    # accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

    # training, learning rate = 0.005
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, name="gradDescent")

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
            acc, cost = sess.run([accuracy, cross_entropy], feed_dict={X: batch_X, Y_: batch_Y})
            train_indexes.append(i)
            train_costs.append(cost)
            train_accuracies.append(acc)
            print(str(i) + ": accuracy:" + str(acc) + " loss: " + str(cost))

        # compute test values
        if i % 50 == 0:
            acc, cost = sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
            test_indexes.append(i)
            test_costs.append(cost)
            test_accuracies.append(acc)
            print(str(i) + ": ********* epoch " + str(
                i * minibatch_size // mnist.train.images.shape[0] + 1) + " ********* test accuracy:" + str(
                acc) + " test loss: " + str(cost))

        # the backpropagation training step
        sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y})

    print("max test accuracy: " + str(max(test_accuracies)))

    # plot train and test costs and accuracies
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(np.squeeze(train_indexes), np.squeeze(train_accuracies), label="train_accuracy")
    plt.plot(np.squeeze(test_indexes), np.squeeze(test_accuracies), label="test_accuracy")
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.subplot(122)
    plt.plot(np.squeeze(train_indexes), np.squeeze(train_costs), label="train_costs")
    plt.plot(np.squeeze(test_indexes), np.squeeze(test_costs), label="test_costs")
    plt.legend()
    plt.ylabel('costs')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate))
    # plt.show()
    plt.savefig(
        "output/learning_rate " + str(learning_rate) + " iterations " + str(iterations) + " max test accuracy " + str(
            max(test_accuracies)) + " .png")

    sess.close()


if __name__ == "__main__":
    # for learning_rate in [0.5, 0.05, 0.005]:
    train1(learning_rate=0.005, minibatch_size=100, iterations=10000 + 1)
