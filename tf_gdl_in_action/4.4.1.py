# 学习率设置
# 指数衰减法 tf.train.expoential_decay

import tensorflow as tf

decayed_learning_rate = \
    learning_rate * decay_rate ^ (global_step / decay_steps)

global_step = tf.Variable(0)

# 通过 exponential_decay 函数生成学习率

learing_rate = tf.train.exponential_decay(
    0.1, global_step, 100, 0.96, staircase=True
)

# 使用指数衰减的学习率, 在 minimize 函数中传入 global_step 将自动更新
# global_step 参数, 从而使得学习率也得到相应更新

learing_step = tf.train.GradientDescentOptimizer(learing_rate)\
    .minimize(...my loss..., global_step=global_step)