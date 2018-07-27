# import tensorflow as tf
# from tensorflow.python.framework import graph_util
#
# v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
# v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
# result = v1 + v2
#
# init_op = tf.global_variables_initializer()
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init_op)
#     # 导出当前图的　GraphDef 部分, 只需要这部分即可完成输入层到输出层计算过程
#     graph_def = tf.get_default_graph().as_graph_def()
#     output_graph_def = graph_util.convert_variables_to_constants(
#         sess, graph_def, ['add']
#     )
#     with tf.gfile.GFile("/home/belem/github/tf/model541b.pb", "wb") as f:
#         f.write(output_graph_def.SerializeToString())


import tensorflow as tf
from tensorflow.python.platform import gfile

with tf.Session() as sess:
    model_filename = "/home/belem/github/tf/model541b.pb"
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    result = tf.import_graph_def(graph_def, return_elements=["add:0"])
    print(sess.run(result))