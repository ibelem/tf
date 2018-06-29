# Also refer to https://github.com/caicloud/tensorflow-tutorial/tree/master/Deep_Learning_with_TensorFlow/1.4.0

# Just disables the warning, doesn't enable AVX/FMA
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

a = tf.constant([1.0, 2.0], name='a')
b = tf.constant([3.0, 4.0], name='b')

result = a + b
print(result)

with tf.Session() as sess:
    print(sess.run(result))
