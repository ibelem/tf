# import tensorflow as tf
#
# v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
# v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
# result = v1 + v2
#
# x = tf.constant([[1.0, 1.0]])
# w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1), name='w1')
# a = tf.matmul(x, w1, name='example-matmul')
#
# init_op = tf.global_variables_initializer()
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init_op)
#     print(sess.run(result))
#     print(sess.run(a))
#     saver.save(sess, '/home/belem/github/tf/model1.ckpt')
#     # model1.ckpt.meta 计算图结构
#     # model1.ckpt TF 每个变量取值
#     # checkpoint 目录下模型文件列表


# import tensorflow as tf
#
# # 使用和模型代码中一样的方式来声明变量
# v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
# v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
# result = v1 + v2
#
# x = tf.constant([[1.0, 1.0]])
# w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1), name='w1')
# a = tf.matmul(x, w1, name='example-matmul')
#
# # init_op = tf.global_variables_initializer()
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     # sess.run(init_op)
#     saver.restore(sess, '/home/belem/github/tf/model1.ckpt')
#     print(sess.run(result))
#     print(sess.run(a))


import tensorflow as tf

saver = tf.train.import_meta_graph('/home/belem/github/tf/model1.ckpt.meta')

with tf.Session() as sess:
    saver.restore(sess, '/home/belem/github/tf/model1.ckpt')
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))
    print(sess.run(tf.get_default_graph().get_tensor_by_name("example-matmul:0")))