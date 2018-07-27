import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference

BATCH_SIZE = 100    # 数字越小训练过程越接近随机梯度下降，数字越大越接近梯度下降
LEARNING_RATE_BASE = 0.8  #基础学习率
LEARNING_RATE_DECAY = 0.99  #学习率的衰减率
REGULARIZATION_RATE = 0.0001    #描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000   #训练轮数
MOVING_AVERAGE_DECAY = 0.99 #滑动平均衰减率

MODEL_SAVE_PATH = '/home/belem/github/tf/'
MODEL_NAME = 'mnist55.ckpt'


def train(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')  #维度可以自动算出，也就是样本数
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NONE], name='y-input')

    # 损失函数的计算 + L2 正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE) #正则化损失函数
    # 计算不含滑动平均类的前向传播结果
    y = mnist_inference.inference(x, regularizer)

    # 定义训练轮数
    global_step = tf.Variable(0, trainable=False)   #一般训练轮数的变量指定为不可训练的参数
    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    # 给定训练轮数的变量可以加快训练早期变量的更新速度
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 在所有代表神经网络的参数的变量上使用滑动平均，其他辅助变量就不需要了
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算交叉熵及其平均值
    # 交叉熵为刻画预测之及真实值之间差距的损失函数
    # sparse_softmax_cross_entropy_with_logits 分类问题只有一个正确答案(本例0~9)时使用此函数加速交叉熵计算
    # y 不包括　softmax 层的前向传播结果; 正确答案是一个长度为10的一维数组, tf.argmax　为训练数据的正确答案对应的类别编号
    # 这里tf.argmax(y_,1)表示在“行”这个维度上张量最大元素的索引号
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
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

    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 循环的训练神经网络
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is is %g " % (step, loss_value))
        saver.save(
            sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step
        )

def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    print("====== MNIST data downloaded ======")
    train(mnist)

if __name__ == '__main__':
    tf.app.run()