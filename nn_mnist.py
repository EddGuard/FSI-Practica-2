import gzip
import _pickle as cPickle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
f.close()

train_x, train_y = train_set

train_y = one_hot(train_y, 10)

valid_x, valid_y = valid_set

valid_y = one_hot(valid_y, 10)

test_x, test_y = test_set

test_y = one_hot(test_y, 10)

# ---------------- Visualizing some element of the MNIST dataset --------------




# TODO: the neural net!!
x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 20)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(20)) * 0.1)

#W12 = tf.Variable(np.float32(np.random.rand(20, 15)) * 0.1)
#b12 = tf.Variable(np.float32(np.random.rand(15)) * 0.1)

#W13 = tf.Variable(np.float32(np.random.rand(9, 11)) * 0.1)
#b13 = tf.Variable(np.float32(np.random.rand(11)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(20, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
#h2 = tf.nn.sigmoid(tf.matmul(h,W12) + b12)

## h = tf.matmul(x, W1) + b1  # Try this!
#h3 = tf.nn.sigmoid(tf.matmul(h2,W13) + b13)
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

#y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print("----------------------")
print("   Start training...  ")
print("----------------------")

batch_size = 20

error_list, error_train_list= [], []

epoch, error, error_anterior = 0, 5. , 100.
while np.absolute(error_anterior - error) > 0.0001 * (np.absolute(error_anterior) + 1.):

    error_anterior = error


    for jj in range(len(train_x) // batch_size):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        error_train = sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    error = sess.run(loss, feed_dict={x: valid_x, y_: valid_y})

    error_list.append(error)
    error_train_list.append(error_train)

    print("Validation #: ", epoch, "\n Error || Error anterior \n", error, "||", error_anterior)

    epoch += 1

    print
    #result = sess.run(y, feed_dict={x: batch_xs})
    #for b, r in zip(batch_ys, result):
    #    print b, "-->", r
    #print "----------------------------------------------------------------------------------"

plt.plot(error_list)
plt.ylabel('some numbers')
plt.show()

print("----------------------")
print("   Start Test...  ")
print("----------------------")

error_total, error = 0., 0.
test_res = sess.run(y, feed_dict={x: test_x})

for b, r in zip(test_y, test_res):
    if np.argmax(b) != np.argmax(r):
        error += 1
    error_total += 1
porcentaje = error / error_total * 100.0
print("% Error: ", porcentaje,"% Exito", (100.0 - porcentaje), "%")

plt.ylabel('Errores')
plt.xlabel('Epocas')
tr_handle, = plt.plot(error_train_list)
vl_handle, = plt.plot(error_list)