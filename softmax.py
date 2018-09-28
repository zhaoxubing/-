import tensorflow as tf
import numpy as np


A = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
B = [[1, 3, 4], [2, 4, 8]]

print(B)
B = np.array(B)
print(B.shape)
print(A)
A = np.array(A)
print(A.shape)
# 1表示横轴，方向从左到右；0表示纵轴，方向从上到下

with tf.Session() as sess:
    print(sess.run(tf.nn.softmax(A)))
    print(sess.run(tf.argmax(A, axis=0)))
    print(sess.run(tf.argmax(B, axis=0)))
    print(sess.run(tf.argmax(B, axis=1)))
