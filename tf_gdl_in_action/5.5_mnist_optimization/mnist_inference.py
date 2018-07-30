import tensorflow as tf

INPUT_NODE = 784  #28x28 输入层节点数 == 图片像素数
OUTPUT_NODE = 10  #输出层节点数, 0~9
LAYER1_NODE = 500   #隐藏层节点数

def get_wight_variable(shape, regularizer):
    weights = tf.get_variable(
        "weights", shape,
        initializer=tf.truncated_normal_initializer(stddev=0.1)
    )
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights

def inference(input_tensor, regularizer):
    # 隐藏层
    with tf.variable_scope('layer1'):
        weights = get_wight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable('biases', [LAYER1_NODE],
                                 initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    # 输出层
    with tf.variable_scope('layer2'):
        weights = get_wight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable('biases', [OUTPUT_NODE],
                                 initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2