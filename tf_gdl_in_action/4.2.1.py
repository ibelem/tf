# 经典损失函数
# 交叉熵
# 均方误差损失函数

import tensorflow as tf

x1 = tf.constant([[1.0, 2.0], [2.0, 1.0]])
x2 = tf.constant([[2.0, 3.0], [3.0, 1.0]])

a1 = tf.matmul(x1, x2)
a2 = x1 * x2
# 1 2         2 3
# 2 1         3 1
# matmul
# 8 5
# 7 7
# *
# 2 6
# 6 1

v = tf.constant([[1.,2.], [3.,4.]])
# cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_)
# mse = tf.reduce_mean(tf.square(y_ - y))

with tf.Session() as sess:
    print(sess.run(a1))
    print(sess.run(a2))
    print(sess.run(tf.reduce_mean(v)))