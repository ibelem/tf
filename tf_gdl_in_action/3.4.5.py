# 完整神经网络样例程序

## 神经网络训练步骤
## 1. 定义神经网络的结构和前向传播的输出结果 得到预测值, 并对比预测值和真实值的差距
## 2. 定义损失函数以及选择反向传播优化的算法
## 3. 生成会话并在训练数据上反复运行反向传播优化算法

import tensorflow as tf
# Numpy 生成模拟数据集
from numpy.random import RandomState

# 训练数据 batch 大小
batch_size = 8

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1), name='w1')
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1), name='w2')

# shape 的一个维度使用 None 可方便使用不打的 batch 大小
# 训练时需要将数据分成较小 batch
# 测试时可一次性使用全部数据
# 数据集较小时比较方便测试
# 数据集较大时，大量数据放入一个 batch 可能导致内存溢出
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 定义神经网络前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 损失函数和反向传播过程
# 交叉熵 H(p, q) = -∑p(x)logq(x)

# y_ 正确结果
# y 预测结果
# tf.clip_by_value 限定张量数值在一个范围内 1.0x10^-10 ~ 1.0
# * 矩阵元素直接相乘，不同于矩阵乘法 tf.matmul()
# 平均交叉熵
cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
# cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_)
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 随机数生成模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# 定义规则给出样本的标签
# x1 + x2 < 1 为正样本, 其他为负样本
# 此处 0 为负样本, 1  为正样本
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))

    # print(8%128)  8
    # print(16%128) 16

    # 训练轮数
    STEPS = 5000
    for i in range(STEPS):
        # 每次选取 batch_size 个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)

        # 选取样本训练神经网络并更新参数
        sess.run(train_step,
                 feed_dict={
                     x: X[start:end],
                     y_: Y[start:end]
                 })
        if i % 1000 == 0:
            # 每隔一段时间在所有数据的交叉熵并输出
            total_cross_entropy = sess.run(
                cross_entropy, feed_dict={
                    x: X, y_: Y
                }
            )
            # 结果随训练进行，交叉熵逐渐变小，说明预测结果与真实结果差距越小
            print("After %d training step(s), cross entropy on all data is %g" %(i, total_cross_entropy))
    print(sess.run(w1))
    print(sess.run(w2))