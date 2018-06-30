# 自定义损失函数
# loss = tf.reduce_sum(tf.where(tf.greater(v1, v2), (v1 - v2) * a, (v2 - v1) * b))


import tensorflow as tf
from numpy.random import RandomState

x1 = tf.constant([[1.0, 2.0], [2.0, 2.0]])
x2 = tf.constant([[2.0, 3.0], [3.0, 1.0]])

a1 = tf.greater(x1, x2)
b1 = tf.where(tf.greater(x1, x2), x1, x2)

with tf.Session() as sess:
    print(sess.run(a1))
    print(sess.run(b1))


batch_size = 8

# 两个输入节点
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')

# 回归问题一般只有一个输出节点
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 单层NN前向传播过程，简单加权和
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

loss_less = 10
loss_more = 1
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_)*loss_more, (y_- y)*loss_less))

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# 随机数生成模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# 回归的正确值为两个输入的和加上一个随机量
# 随机量目的是加入不可预测的噪音 否则不同损失函数的意义就不大
# 因为不同损失函数都会在能完全预测正确时最低
# 一般噪音为一个均值为0的小量，设为 -0.05 ~ 0.05 的随机数

Y = [[x1 + x2 + rdm.rand()/10.0-0.05] for (x1, x2) in X]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(w1.get_shape())
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        sess.run(train_step, feed_dict={ x: X[start:end], y_: Y[start:end]})
        print(sess.run(w1))

# y = x1 + x2 = 1.02x1 +1.04x2 表明倾向于预测多一点
