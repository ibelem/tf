import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import mnist_train

# 每 10 秒加载一次最新模型　并在测试数据上测试最新模型正确率
EVAL_INTERVAL_SECS = 20

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        # 测试时正则化损失设为 None
        y = mnist_inference.inference(x, None)

        # 计算正确率
        # tf.argmax(y, 1)　计算没一个样例的预测答案
        # y 为　batch_size * 10 的二维数组, 每一行表示一个样例的前向传播结果
        # 第二个参数 1 表示选取最大值的操作仅在第一个维度中进行
        # 每一行选取最大值对应下标, 得到结果是一个长度为　batch 的一维数组
        # 该一维数组中的值表示没一个样例对应的数字识别结果
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # 转换数值类型为浮点型, 然后计算平均值
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
        # 给定训练轮数的变量可以加快训练早期变量的更新速度
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    print('ckpt.model_checkpoint_path', ckpt.model_checkpoint_path)
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print('After %s training step(s), validation accuracy = %g' % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()
