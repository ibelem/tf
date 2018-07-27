# 5.1

# t10k-images-idx3-ubyte.gz	 Training Data Image
# t10k-labels-idx1-ubyte.gz	 Training Data Result
# train-images-idx3-ubyte.gz Testing Data Image
# train-labels-idx1-ubyte.gz Testing Data Result

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.WARN)

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("MNIST_data/", tf.one_hot)
# print(mnist.train.num_examples)

INPUT_NODE = 784  #28x28 输入层节点数 == 图片像素数
OUTPUT_NONE = 10  #输出层节点数, 0~9

'''配置神经网络的参数'''
LAYER1_NODE = 500   #隐藏层节点数
BATCH_SIZE = 100    # 数字越小训练过程越接近随机梯度下降，数字越大越接近梯度下降

# 模型相关的参数
LEARNING_RATE_BASE = 0.8  #基础学习率
LEARNING_RATE_DECAY = 0.99  #学习率的衰减率
REGULARIZATION_RATE = 0.0001    #描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000   #训练轮数
MOVING_AVERAGE_DECAY = 0.99 #滑动平均衰减率

def inference(input_tensor, avg_class, reuse=False):
    with tf.variable_scope('layer1', reuse=tf.AUTO_REUSE):
        weights = tf.get_variable('weights', [INPUT_NODE, LAYER1_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [INPUT_NODE, LAYER1_NODE],
                                 initializer=tf.constant_initializer(0.0))
        if avg_class:
            layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights)) + avg_class.average(biases))
        else:
            layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2', reuse=tf.AUTO_REUSE):
        weights = tf.get_variable('weights', [LAYER1_NODE, OUTPUT_NONE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [OUTPUT_NONE],
                                 initializer=tf.constant_initializer(0.0))
        if avg_class:
            layer2 = tf.nn.relu(tf.matmul(layer1, avg_class.average(weights)) + avg_class.average(biases))
        else:
            layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)

    return layer2

# '''定义一个辅助函数，用于计算神经网络的前向结果
# ReLU 激活函数的三层全连接神经网络, 加入隐藏层实现多层网络结构
# 通过 ReLU 实现非线性
# 其中参数avg_classs是用于计算参数平均值的类
# 这样方便在测试时使用滑动平均模型'''
# def inference(input_tensor, avg_class, w1, b1, w2, b2):
#     '''
#     :param input_tensor: 输入
#     :param avg_class: 用于计算参数平均值的类
#     :param w1: 第一层权重
#     :param b1: 第一层偏置
#     :param w2: 第二层权重
#     :param b2: 第二层偏置
#     :return: 返回神经网络的前向结果
#     '''
#     if avg_class == None:
#         layer1 = tf.nn.relu(tf.matmul(input_tensor, w1) + b1)
#         # 计算损失函数时会一并计算 softmax 函数, 因为此处无需加入激活函数
#         # 不加入 softmax 不影响预测结果, 因为预测时使用的是不同类别对应节点输出值相对大小
#         # 没有 softmax 层对最后分类结果计算没有影响
#         # 于是最后计算整个神经网络前向传播时可不加入最后的 softmax 层
#         return tf.matmul(layer1, w2) + b2
#     else:
#         # 使用滑动平均类计算参数的滑动平均值
#         layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(w1)) + avg_class.average(b1))
#         return tf.matmul(layer1, avg_class.average(w2)) + avg_class.average(b2)

def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')  #维度可以自动算出，也就是样本数
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NONE], name='y-input')
    # 生成隐藏层的参数
    # w1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))    #一种正态的随机数
    # b1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层的参数
    # w2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NONE], stddev=0.1))
    # b2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NONE]))
    # 计算不含滑动平均类的前向传播结果
    y = inference(x, None)
    # y = inference(x, None, w1, b1, w2, b2)
    # 定义训练轮数
    global_step = tf.Variable(0, trainable=False)   #一般训练轮数的变量指定为不可训练的参数
    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    # 给定训练轮数的变量可以加快训练早期变量的更新速度
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 在所有代表神经网络的参数的变量上使用滑动平均，其他辅助变量就不需要了
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # 计算使用滑动平均的前向结果
    # 滑动平均不会改变变量本身取值，而是维护一个影子变量来记录其滑动平均，需要明确调用 average 函数
    average_y = inference(x, variable_averages)
    # average_y = inference(x, variable_averages, w1, b1, w2, b2)
    # 计算交叉熵及其平均值
    # 交叉熵为刻画预测之及真实值之间差距的损失函数
    # sparse_softmax_cross_entropy_with_logits 分类问题只有一个正确答案(本例0~9)时使用此函数加速交叉熵计算
    # y 不包括　softmax 层的前向传播结果; 正确答案是一个长度为10的一维数组, tf.argmax　为训练数据的正确答案对应的类别编号
    # 这里tf.argmax(y_,1)表示在“行”这个维度上张量最大元素的索引号
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 损失函数的计算 + L2 正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE) #正则化损失函数

    w1 = tf.get_variable('layer1/weights', [INPUT_NODE, LAYER1_NODE])
    w2 = tf.get_variable('layer2/weights', [LAYER1_NODE, OUTPUT_NONE])
    regularization = regularizer(w1) + regularizer(w2)  #模型的正则化损失
    loss = cross_entropy_mean + regularization  #总损失函数=交叉熵损失和正则化损失的和
    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, #　基础学习率, 随着迭代的进行, 更新变量时使用的学习率在此基础上递减
        global_step,
        mnist.train.num_examples / BATCH_SIZE,  #过完所有训练数据需要的迭代次数
        LEARNING_RATE_DECAY,
        staircase=True
    )
    # 优化损失函数，用梯度下降法来优化, 此处的损失函数包含了交叉熵损失和L2正则化损失
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 反向传播更新参数和更新每一个参数的滑动平均值
    # 为一次完成多个操作, TF 提供　tf.control_dependencies 及　tf.group 两种机制
    # 下面两行等价于　train_op = tf.group(train_step, variable_averages_op)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
    # 计算正确率
    # tf.argmax(average_y, 1)　计算没一个样例的预测答案
    # average_y 为　batch_size * 10 的二维数组, 每一行表示一个样例的前向传播结果
    # 第二个参数 1 表示选取最大值的操作仅在第一个维度中进行
        # 每一行选取最大值对应下标, 得到结果是一个长度为　batch 的一维数组
        # 该一维数组中的值表示没一个样例对应的数字识别结果
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    # 转换数值类型为浮点型, 然后计算平均值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        # 测试数据在训练时不可见, 仅作为模型优劣的评价标准
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        # 循环的训练神经网络
        for i in range(TRAINING_STEPS):
            # 复杂神经网络模型中, 太大的　batch 会导致计算时间过长甚至发生内存溢出错误
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            # print("xs.shape: ", xs.shape)
            # print("ys.shape", ys.shape)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
        # 训练结束后在测试数据上检测模型正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s), test accuracy using average model is %g " % (TRAINING_STEPS, test_acc))

def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    print("====== MNIST data downloaded ======")
    print("MNIST training data size: ", mnist.train.num_examples)
    print("MNIST validation data size: ", mnist.validation.num_examples)
    print("MNIST testing data size: ", mnist.test.num_examples)
    print("MNIST training data images[0]: ", mnist.train.images[0])
    print("MNIST training data label[0]: ", mnist.train.labels[0])
    train(mnist)
    tf.logging.set_verbosity(old_v)

if __name__ == '__main__':
    tf.app.run()
