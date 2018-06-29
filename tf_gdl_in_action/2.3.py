# Also refer to https://github.com/caicloud/tensorflow-tutorial/tree/master/Deep_Learning_with_TensorFlow/1.4.0

# Just disables the warning, doesn't enable AVX/FMA
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

a = tf.constant([1.0, 2.0], name='a')
b = tf.constant([3.0, 4.0], name='b')


# mat 2x3, 矩阵元素均值为0, 标准差为2
weights = tf.Variable(tf.random_normal([2, 3], stddev=2))

w2 = tf.Variable(tf.truncated_normal([2, 3], stddev=2))

d1 = tf.zeros([2, 3], tf.int32)
d2 = tf.ones([2, 3], tf.int32)
d3 = tf.fill([2, 3], 9)

print(d1)
print(d2)
print(d3)

biases = tf.Variable(tf.zeros([3]))
print(biases)

w2 = tf.Variable(weights.initialized_value())
w3 = tf.Variable(weights.initialized_value() * 2.0)

print(w3)

result = a + b
result2 = tf.add(a, b, name='add')
print(result)
print(result2)

# 张量维度
print(result.get_shape())
print(weights.get_shape())


with tf.Session() as sess:
    print(sess.run(result))

sess = tf.Session()
with sess.as_default():
    print(result.eval())

sess = tf.Session()
print(sess.run(result))
print(result.eval(session=sess))

sess = tf.InteractiveSession()
print(result.eval())
sess.close()