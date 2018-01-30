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
training_epochs = 1000
batch_size = 100
display_step = 1
example_num = len(input_x)
# tf Graph Input
x = tf.placeholder(tf.float32, [None, 2])
y = tf.placeholder(tf.float32, [None, 1])

# 设置模型参数变量w和b
W = tf.Variable(tf.random_uniform([2,1], -1.0, 1.0))
b = tf.Variable(tf.ones([1]))
global_step = tf.Variable(0)
pred = tf.matmul(x, W) + b
learning_rate = tf.train.exponential_decay(1e-2,global_step,decay_steps=example_num/batch_size,decay_rate=0.98,staircase=True)


# 损失函数用mean squared error
cost = tf.reduce_mean(tf.square(pred - y))
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
        total_batch = int(len(input_x)/batch_size)
        # 每一轮迭代total_batches
        for i in range(total_batch):
            batch_xs, batch_ys = data_batch(input_x, input_y, i, batch_size)
            # print batch_xs
            # print batch_ys
            # 使用batch data训练数据
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                      y: batch_ys})
            # 将每个batch的损失相加求平均
            avg_cost += c / total_batch
        # 每一轮打印损失
        if (epoch+1) % display_step == 0:
            result = sess.run(merged, feed_dict={x: input_x, y: input_y})  # merged也是需要run的
            writer.add_summary(result, epoch+1)
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

    print "Optimization Finished!"

    # 模型预测
    # tf.argmax(pred,axis=1)是预测值每一行最大值的索引，这里最大值是概率最大
    # tf.argmax(y,axis=1)是真实值的每一行最大值得索引，这里最大值就是1
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    #
    # 对3000个数据预测准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print "Accuracy:", accuracy.eval({x:input_x[:1000], y: input_y[:1000]})

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
