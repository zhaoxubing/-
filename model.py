# 1、导入包
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import data_processing

# 2、准备数据
data = data_processing.load_data(download=False)
new_data = data_processing.covert2onehot(data)
# print(data)
# print(new_data)
new_data = new_data.values.astype(np.float32)
# print("前",new_data)
np.random.shuffle(new_data)
# print("后",new_data)
sep = int(0.7 * len(new_data))
train_data = new_data[:sep]
test_data = new_data[sep:]

# 3、搭建网络模型 build network
tf_input = tf.placeholder(tf.float32, [None, 25], "input")
tfx = tf_input[:, :21]
tfy = tf_input[:, 21:]

L1 = tf.layers.dense(tfx, 128, tf.nn.relu, name="L1")
L2 = tf.layers.dense(L1, 128, tf.nn.relu, name="L2")
out = tf.layers.dense(L2, 4, name="L3")
prediction = tf.nn.softmax(out, name="pred")

loss = tf.losses.softmax_cross_entropy(onehot_labels=tfy, logits=out)
accuracy = tf.metrics.accuracy(labels=tf.argmax(tfy, axis=1), predictions=tf.argmax(out, axis=1), )[1]

train_opt = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

# 4、训练
sess = tf.Session()

sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
accuracies, steps = [], []
losses, prediction_s, outs = [], [], []
for t in range(4001):
    batch_index = np.random.randint(len(train_data), size=32)
    sess.run(train_opt, feed_dict={tf_input: train_data[batch_index]})
    if t % 50 == 0:
        acc_, pred_, loss_, out_ = sess.run([accuracy, prediction, loss, out, ], {tf_input: test_data})
        accuracies.append(acc_)
        steps.append(t)
        losses.append(loss_)
        prediction_s.append(pred_)
        outs.append(out_)
        print("Step: %i" % t, "| Accurate: %.2f" % acc_, "| Loss: %.2f" % loss_, )

        # visualize testing
        ax1.cla()
        for c in range(4):
            # print(sum((np.argmax(pred_, axis=1) == c)))
            # print(np.argmax(pred_, axis=1))
            # print(sum((np.argmax(test_data[:, 21:], axis=1) == c)))
            # print(np.argmax(test_data[:, 21:], axis=1))
            bp = ax1.bar(c + 0.1, height=sum((np.argmax(pred_, axis=1) == c)), width=0.2, color='red')
            bt = ax1.bar(c - 0.1, height=sum((np.argmax(test_data[:, 21:], axis=1) == c)), width=0.2, color='blue')
        ax1.set_xticks(range(4), ["accepted", "good", "unaccepted", "very good"])
        # handles[bp,bt] 为bp,bt 创建图列
        ax1.legend(handles=[bp, bt], labels=["prediction", "target"])

        ax1.set_ylim((0, 400))

        ax2.cla()
        ax2.plot(steps, accuracies, label="accuracy")
        ax2.set_ylim(ymax=1)
        ax2.set_ylabel("accuracy")
        plt.pause(0.01)
# print(accuracies)
# print(losses)
# print(np.array(prediction_s).shape)
# print("prediction\n", prediction_s[0])
# print(np.array(outs).shape)
# print("out\n", outs[0])
plt.ioff()
plt.show()
