# 神经网络样例程序

import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1), name='w1')
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1), name='w2')

# 定义 placeholder 为存数据的地方, 维度不一定需定义
# 维度确定并给出，降低出错概率
x = tf.placeholder(tf.float32, shape=(1,2), name='input')

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(y, feed_dict={x: [[0.7, 0.9]]}))

x2 = tf.placeholder(tf.float32, shape=(3, 2), name='input2')

a2 = tf.matmul(x2, w1)
y2 = tf.matmul(a2, w2)

with tf.Session() as sess:
    init_op2 = tf.global_variables_initializer()
    sess.run(init_op2)
    print(sess.run(y2, feed_dict={x2: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))