#encoding:utf-8

import tensorflow as tf

file = open("../input/train_data", "r")
input_x = []
input_y = []
for line in file:
    list = line.split()
    input_x.append([float(list[0]) / 1032, float(list[1]) / 7])
    input_y.append([float(list[2]) / 7226])
# 一些参数
training_epochs = 100000
batch_size = 1000
display_step = 10
example_num = len(input_x)
# tf Graph Input
x = tf.placeholder(tf.float32, [None, 2])
y = tf.placeholder(tf.float32, [None, 1])

# 设置模型参数变量w和b
W = tf.Variable(tf.random_uniform([2,1], -1.0, 1.0))
b = tf.Variable(tf.ones([1]))
global_step = tf.Variable(0)
theta = tf.Variable(tf.zeros([2, 1]))
theta0 = tf.Variable(tf.zeros([1, 1]))
pred = 1 / (1 + tf.exp(-tf.matmul(x, theta) + theta0))
learning_rate = tf.train.exponential_decay(1e-2,global_step,decay_steps=example_num,decay_rate=0.98,staircase=True)
# learning_rate = 0.05
# 损失函数用mean squared error
cost = tf.reduce_mean(-y * tf.log(pred) - (1 - y) * tf.log(1 - pred))
tf.summary.scalar('learnging_rate', learning_rate)
tf.summary.scalar('cost', cost)
# 梯度下降优化
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step=global_step)
# 初始化所有变量
init = tf.initialize_all_variables()


def data_batch(input_x, input_y, cnt, batch_size):
    temp_x = input_x[cnt * batch_size:(cnt + 1) * batch_size]
    temp_y = input_y[cnt * batch_size:(cnt + 1) * batch_size]
    return temp_x,temp_y
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    # 合并到Summary中
    merged = tf.summary.merge_all()
    # 选定可视化存储目录
    writer = tf.summary.FileWriter("../log", sess.graph)
    for epoch in range(training_epochs):
        avg_cost = 0.
        #total_batch = int(len(input_x)/batch_size)
        # 每一轮迭代total_batches
        #for i in range(total_batch):
        #batch_xs, batch_ys = data_batch(input_x, input_y, i, batch_size)
        batch_xs, batch_ys = input_x[:1000], input_y[:1000]
            # print batch_xs
            # print batch_ys
            # 使用batch data训练数据
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                      y: batch_ys})
            # 将每个batch的损失相加求平均
            #avg_cost += c / total_batch
        avg_cost += c
        # 每一轮打印损失
        if (epoch+1) % display_step == 0:
            result = sess.run(merged, feed_dict={x: input_x, y: input_y})  # merged也是需要run的
            writer.add_summary(result, epoch+1)
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

    print "Optimization Finished!"

    # 模型预测
    # 对1032个数据预测准确率
    x = input_x[1000:1032]
    y = input_y[1000:1032]
    pred = tf.matmul(x, W) + b
    cost = tf.square((pred - y) * 7226) / len(x)
    cost_temp = cost.eval()
    total_cost = 0
    for num in cost_temp:
        temp = str(num).strip("[]").strip()
        total_cost += int(float(temp))
    print total_cost
    test_file = open("../input/test_data","r")

    test_x = []
    test_y = []
    for line in test_file:
        list = line.split()
        test_x.append([float(list[0]) / 1579, float(list[1]) / 7])
    test_y = (tf.matmul(test_x, W) + b) * 7226
    res = test_y.eval()
    out_file = open("../output/out", "w")
    index = 1032
    for num in res:
        temp = str(num).strip("[]").strip()
        out_file.write(str(index) + '\t' + str(int(float(temp)))+ '\n')
        index += 1
    # sess.run(tf.Print(test_y, [test_y], summarize=548))
