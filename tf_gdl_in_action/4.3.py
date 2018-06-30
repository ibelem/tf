# 损失函数 梯度下降算法 优化单个参数取值
# 反向传播算法给出一个高效方式在所有参数上使用梯度下降算法

## 神经网络训练步骤
## 1. 定义神经网络的结构和前向传播的输出结果 得到预测值, 并对比预测值和真实值的差距
## 2. 定义损失函数以及选择反向传播优化的算法 对每一个参数的梯度, 依据梯度和学习率使用梯度下降更新每一个参数
## 3. 生成会话并在训练数据上反复运行反向传播优化算法

# 随机梯度下降散发 (优化所有训练数据的损失函数的计算)

# 折中 每次计算一小部分训练函数的损失函数，称为一个 batch
#每次在一个 batch 可大大减小收敛所需迭代次数，接近梯度下降效果

import tensorflow as tf
from numpy.random import RandomState

batch_size = n

# 每次读取一小部分数据作为当前训练数据来执行反向传播算法
x = tf.placeholder(tf.float32, shape=(batch_size, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(batch_size, 1), name='y-input')

# 定义NN结构和优化算法
loss = ...
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    ...
    STEPS = 5000
    for i in range(STEPS):
        # 准备 batch_size 个训练数据
        current_X, current_Y = ...
        sess.run(train_step, feed_dict={ x: current_X, y_: current_X})