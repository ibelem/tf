# import tensorflow as tf
# 
# v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
# v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
# 
# result1 = v1 + v2
# 
# saver = tf.train.Saver()
# 
# saver.export_meta_graph("/home/belem/github/tf/model542.ckpt.meta.json", as_text=True)
# 

import tensorflow as tf

reader = tf.train.NewCheckpointReader('/home/belem/github/tf/model5.4.1.ckpt')

all_variables = reader.get_variable_to_shape_map()
for variable_name in all_variables:
    print(variable_name, all_variables[variable_name])
print("Value for variables v is ", reader.get_tensor("v"))


