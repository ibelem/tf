# 过拟合问题

# 正则化 regularization
# L1 and L2

import tensorflow as tf

# 获取一层NN边的权重, 并将此权重的L2 正则化损失加入名称为 losses 的集合中
def get_weight(shape, lamda):
    var = tf.Variable(tf.random_normal(shape), dtype = tf.float32)
    # add_to_collection 函数将新生成变量的 L2 正则化损失项加入集合
    # 这个哈数的第一个参数 'losses' 是集合名字, 第二个参数是要加入这个集合的内容
    tf.add_to_collection(
        'losses',
        tf.contrib.layers.l2_regularizer(lamda)(var)
    )
    return var


batch_size = 8

# 两个输入节点
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')

# 回归问题一般只有一个输出节点
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 定义每一层网络中节点个数
layer_dimension = [2, 10, 10, 10, 1]
# 神经网络的层数
n_layers = len(layer_dimension)

# 这个变量维护前向传播最深层节点，开始时为输入层
cur_layer = x
# 当前层节点个数
in_dimension = layer_dimension[0]

# 通过一个循环生成 5 层全连接NN结果
for i in range(1, n_layers):
    # layer_dimension[i] 为下一层节点个数
    out_dimension = layer_dimension[i]
    # 生成当前层中权重的变量，将该变量 L2 正则化损失加入计算图上的集合
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    # 使用 ReLu 激活函数
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    # 进入下一层之前将下一层节点数更新为当前层节点个数
    in_dimension = layer_dimension[i]

# 定义nn前向传播同事将所有 L2 正则化损失加入了图上的集合
# 这里只需计算刻画模型在训练数据上表现的损失函数
mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))

# 将均方误差损失函数加入损失集合
tf.add_to_collection('losses', mse_loss)

# get_collection 返回一个列表，是所有这个集合中的元素
# 这些元素是损失函数的不同部分，加起来得到最终的损失函数

loss = tf.add_n(tf.get_collection('losses'))
