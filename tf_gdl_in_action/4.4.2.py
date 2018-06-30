# 过拟合问题

# 正则化 regularization
# L1 and L2

import tensorflow as tf

# w = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
# y = tf.matmul(x, w)
#
# loss = tf.reduce_mean(tf.square(y_ - y)) +
#     tf.contrib.layers.l2_regularizer(lamda)(w)

weights = tf.constant([[1., -2.], [-3., 4.]])

with tf.Session() as sess:
    # L1 (|1|+|-2|+|-3|+|4| x 0.5 = 5
    # 0.5 为正则化项权重
    print(sess.run(tf.contrib.layers.l1_regularizer(.5)(weights)))
    # L2 (1^2+(-2)^2+(-3)^2+4^2)/2 x 0.5 = 7.5
    print(sess.run(tf.contrib.layers.l2_regularizer(.5)(weights)))