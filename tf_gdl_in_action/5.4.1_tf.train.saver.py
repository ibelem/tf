import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name="v")

for variables in tf.global_variables():
    print(variables.name)

ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.global_variables())

for variables in tf.global_variables():
    print(variables.name)

saver = tf.train.Saver()

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(tf.assign(v, 10))
    sess.run(maintain_averages_op)

    saver.save(sess, '/home/belem/github/tf/model5.4.1.ckpt')
    print(sess.run([v, ema.average(v)]))


# import tensorflow as tf
#
# v = tf.Variable(0, dtype=tf.float32, name='v')
# ema = tf.train.ExponentialMovingAverage(0.99)
# print(ema.variables_to_restore())
#
# saver = tf.train.Saver(ema.variables_to_restore())
#
# with tf.Session() as sess:
#     saver.restore(sess, '/home/belem/github/tf/model5.4.1.ckpt')
#     print(sess.run(v))

import tensorflow as tf

saver = tf.train.import_meta_graph('/home/belem/github/tf/model5.4.1.ckpt.meta')

with tf.Session() as sess:
    saver.restore(sess, '/home/belem/github/tf/model5.4.1.ckpt')
    print(sess.run(tf.get_default_graph().get_tensor_by_name("v:0")))
    print(sess.run(tf.get_default_graph().get_tensor_by_name("v/ExponentialMovingAverage:0")))