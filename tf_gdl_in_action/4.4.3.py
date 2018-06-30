# 滑动平均模型 使模型在测试数据上更健壮

import tensorflow as tf

# 定义变量用于计算滑动平均 初始值为0
# 所有需要计算滑动平均的变量必须为实数, 手动制定变量的类型为 float32

v1 = tf.Variable(0, dtype=tf.float32)

# step 模拟nn中迭代轮数, 用于动态控制衰减率
step = tf.Variable(0, trainable=False)

# 定义滑动平均类 初始化时给定衰减率 0.99 及控制衰减率的变量 step
ema = tf.train.ExponentialMovingAverage(0.99, step)

# 定义更新滑动平均的操作 给定一个列表，每次执行这个操作是更新列表变量
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 获取滑动平均后变量取值 初始化后 v1 及 v1 滑动平均都为 0
    print(sess.run([v1, ema.average(v1)]))

    # 更新变量 v1 的值到 5
    sess.run(tf.assign(v1, 5))
    # 更新 v1 滑动平均值 衰减率为 min(0.99, (1+step)/(10+step) = 0.1} = 0.1
    # v1 滑动平均被更新为 0.1x0 + 0.9x5 = 4.5
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))

    # 更新 step 值为 10000
    sess.run(tf.assign(step, 10000))
    # 更新 v1 值为 10
    sess.run(tf.assign(v1, 10))
    # 更新 v1 滑动平均值 衰减率为 min{0.99, (1+step)/(10+step) 约= 0.999} = 0.99
    # v1 滑动平均被更新为 0.99x4.5 + 0.01x10 = 4.555
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))

    # 再次更新滑动平均为 0.99x4.555 + 0.01x10 = 4.60945
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))