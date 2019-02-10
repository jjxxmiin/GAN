import tensorflow as tf
import numpy as np
import scipy.misc as sm
#import cv2

img_shape = [64,64,3]
batch_size = 64
total_epoch = 10
total_batch = 100
learning_rate = 0.0002


# batch normalization node
class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name,
                                            reuse=tf.AUTO_REUSE  # if tensorflow vesrion < 1.4, delete this line
                                            )


#4x4 filter
G_W1 = tf.Variable(tf.truncated_normal([4, 4, 1024, 100], stddev=0.02), name="G_W1")
#batch-norm
G_bn1 = batch_norm(name="G_bn1")

#4x4 filter
G_W2 = tf.Variable(tf.truncated_normal([4, 4, 512, 1024], stddev=0.02), name='G_W2')
#batch-norm
G_bn2 = batch_norm(name="G_bn2")

#4x4 filter
G_W3 = tf.Variable(tf.truncated_normal([4, 4, 256, 512], stddev=0.02), name='G_W3')
#batch-norm
G_bn3 = batch_norm(name="G_bn3")

#4x4 filter
G_W4 = tf.Variable(tf.truncated_normal([4, 4, 128, 256], stddev=0.02), name='G_W4')
#batch-norm
G_bn4 = batch_norm(name="G_bn4")

#4x4 filter
G_W5 = tf.Variable(tf.truncated_normal([4, 4, 3, 128], stddev=0.02), name='G_W5')


 #4x4 filter
D_W1 = tf.Variable(tf.truncated_normal([4, 4, 3, 128], stddev=0.02), name='D_W1')

 #4x4 filter
D_W2 = tf.Variable(tf.truncated_normal([4, 4, 128, 256], stddev=0.02), name='D_W2')
#batch-norm
D_bn2 = batch_norm(name="D_bn2")

#4x4 filter
D_W3 = tf.Variable(tf.truncated_normal([4, 4, 256, 512], stddev=0.02), name='D_W3')
#batch_norm
D_bn3 = batch_norm(name="D_bn3")

#4x4 filter
D_W4 = tf.Variable(tf.truncated_normal([4, 4, 512, 1024], stddev=0.02), name='D_W4')
#batch_norm
D_bn4 = batch_norm(name="D_bn4")

#4x4 filter
D_W5 = tf.Variable(tf.truncated_normal([4, 4, 1024, 1], stddev=0.02), name='D_W5')

D_var_list = [D_W1, D_W2, D_W3, D_W4, D_W5]
G_var_list = [G_W1, G_W2, G_W3, G_W4, G_W5]


def generate(z):
    # generate

    # 1x1x100
    input_ = tf.reshape(z, [batch_size, 1, 1, 100])

    # 1x1x100 -> 4x4x1024
    layer_1 = tf.nn.conv2d_transpose(input_
                                     , G_W1
                                     , output_shape=[batch_size, 4, 4, 1024]
                                     , strides=[1, 4, 4, 1])

    # activation function
    layer_1 = tf.nn.relu(G_bn1(layer_1))

    # 4x4x1024 -> 8x8x512
    layer_2 = tf.nn.conv2d_transpose(layer_1
                                     , G_W2
                                     , output_shape=[batch_size, 8, 8, 512]
                                     , strides=[1, 2, 2, 1])

    # activation function
    layer_2 = tf.nn.relu(G_bn2(layer_2))

    # 8x8x512 -> 16x16x256
    layer_3 = tf.nn.conv2d_transpose(layer_2
                                     , G_W3
                                     , output_shape=[batch_size, 16, 16, 256]
                                     , strides=[1, 2, 2, 1])

    # activation function
    layer_3 = tf.nn.relu(G_bn3(layer_3))

    # 16x16x256 -> 32x32x128
    layer_4 = tf.nn.conv2d_transpose(layer_3
                                     , G_W4
                                     , output_shape=[batch_size, 32, 32, 128]
                                     , strides=[1, 2, 2, 1])

    # activation function
    layer_4 = tf.nn.relu(G_bn4(layer_4))

    # 32x32x128 -> 64x64x3
    layer_5 = tf.nn.conv2d_transpose(layer_4
                                     , G_W5
                                     , output_shape=[batch_size, 64, 64, 3]
                                     , strides=[1, 2, 2, 1])

    output_ = tf.nn.tanh(layer_5)

    return output_

def sample_generator(noise_z,batch_size):
    noise_z = np.array(noise_z).reshape([batch_size,100])

    Z = tf.placeholder(tf.float32,[batch_size,100])
    # 1x1x100
    input_ = tf.reshape(Z, [batch_size, 1, 1, 100])

    # 1x1x100 -> 4x4x1024
    layer_1 = tf.nn.conv2d_transpose(input_
                                     , G_W1
                                     , output_shape=[batch_size, 4, 4, 1024]
                                     , strides=[1, 4, 4, 1])

    # activation function
    layer_1 = tf.nn.relu(G_bn1(layer_1))

    # 4x4x1024 -> 8x8x512
    layer_2 = tf.nn.conv2d_transpose(layer_1
                                     , G_W2
                                     , output_shape=[batch_size, 8, 8, 512]
                                     , strides=[1, 2, 2, 1])

    # activation function
    layer_2 = tf.nn.relu(G_bn2(layer_2))

    # 8x8x512 -> 16x16x256
    layer_3 = tf.nn.conv2d_transpose(layer_2
                                     , G_W3
                                     , output_shape=[batch_size, 16, 16, 256]
                                     , strides=[1, 2, 2, 1])

    # activation function
    layer_3 = tf.nn.relu(G_bn3(layer_3))

    # 16x16x256 -> 32x32x128
    layer_4 = tf.nn.conv2d_transpose(layer_3
                                     , G_W4
                                     , output_shape=[batch_size, 32, 32, 128]
                                     , strides=[1, 2, 2, 1])

    # activation function
    layer_4 = tf.nn.relu(G_bn4(layer_4))

    # 32x32x128 -> 64x64x3
    layer_5 = tf.nn.conv2d_transpose(layer_4
                                     , G_W5
                                     , output_shape=[batch_size, 64, 64, 3]
                                     , strides=[1, 2, 2, 1])

    output_ = tf.nn.tanh(layer_5)

    generated_samples = sess.run(output_, feed_dict={Z: noise_z})
    generated_samples = (generated_samples + 1.) / 2.
    return generated_samples


def save_visualization(X, nh_nw, save_path='./vis/sample.jpg'):
    nh, nw = nh_nw
    h, w = X.shape[1], X.shape[2]
    img = np.zeros((h * nh, w * nw, 3))

    for n, x in enumerate(X):
        j = int(n / nw)
        i = int(n % nw)
        img[j * h:j * h + h, i * w:i * w + w, :] = x

    sm.imsave(save_path, img)

def discriminate(img):
    # 64x64x3 -> 32x32x128
    layer_1 = tf.nn.conv2d(img
                           , D_W1
                           , strides=[1, 2, 2, 1]
                           , padding='SAME')

    # activation function
    layer_1 = tf.nn.leaky_relu(layer_1, alpha=0.2)

    # 32x32x128 -> 16x16x256
    layer_2 = tf.nn.conv2d(layer_1
                           , D_W2
                           , strides=[1, 2, 2, 1]
                           , padding='SAME')

    # activation function
    layer_2 = tf.nn.leaky_relu(D_bn2(layer_2), alpha=0.2)

    # 16x16x256 -> 8x8x512
    layer_3 = tf.nn.conv2d(layer_2
                           , D_W3
                           , strides=[1, 2, 2, 1]
                           , padding='SAME')

    # activation function
    layer_3 = tf.nn.leaky_relu(D_bn3(layer_3), alpha=0.2)

    # 8x8x512 -> 4x4x1024
    layer_4 = tf.nn.conv2d(layer_3
                           , D_W4
                           , strides=[1, 2, 2, 1]
                           , padding='SAME')

    # activation function
    layer_4 = tf.nn.leaky_relu(D_bn4(layer_4), alpha=0.2)

    # 4x4x1024 -> 1x1x1
    layer_5 = tf.nn.conv2d(layer_4, D_W5, strides=[1, 4, 4, 1], padding='SAME')
    layer_5 = tf.reshape(layer_5, [batch_size, 1])

    # percentage
    output_ = tf.nn.sigmoid(layer_5)

    return output_


noise = tf.placeholder(tf.float32,[batch_size,100])
img_real = tf.placeholder(tf.float32,[batch_size]+img_shape)

img_fake = generate(noise)

#image real,fake inspection
d_real = discriminate(img_real)
d_fake = discriminate(img_fake)

#d_cost want d_real to get bigger
#g_cost want d_fake to get smaller
d_cost = -tf.reduce_mean(tf.log(d_real) + tf.log(1 - d_fake) + 1e-10)
#g_cost want d_fake to get bigger
g_cost = -tf.reduce_mean(tf.log(d_fake) + 1e-10)

#train
d_train = tf.train.AdamOptimizer(learning_rate=learning_rate ,beta1=0.5).minimize(d_cost ,var_list = D_var_list)
g_train = tf.train.AdamOptimizer(learning_rate=learning_rate ,beta1=0.5).minimize(g_cost ,var_list = G_var_list)


import Fail_code.reading as df

r = df.reader('C://Users//woals//Git_store//mygan//celeba//',batch_size, 1)

n_noise = 100
noise_test = np.random.normal(size=(64,100))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(total_epoch):
        for step in range(total_batch):
            # next batch 작성
            batch_x = r.next_batch()
            batch_x = batch_x * (2.0 / 255.0) - 1

            # batch_x = cv2.resize(batch_x,(64,64), interpolation = cv2.INTER_AREA)

            z = np.random.normal(size=(batch_size, n_noise))

            _, loss_val_D = sess.run([d_train, d_cost],
                                     feed_dict={img_real: batch_x, noise: z})

            _, loss_val_G = sess.run([g_train, g_cost],
                                     feed_dict={noise: z})

            print('Epoch: [', epoch, '/', total_epoch, '], ', 'Step: [', step, '/', total_batch, '], D_loss: ',
                  loss_val_D, ', G_loss: ', loss_val_G)

    # test code 작성
            if step == 0 or (step + 1) % 10 == 0:
                generated_samples = sample_generator(noise_test, batch_size=64)
                savepath = 'C://Users//woals//Git_store//mygan//' + 'output_' + 'EP' + str(epoch).zfill(3) + "_Batch" + str(step).zfill(6) + '.jpg'
                save_visualization(generated_samples,(14,14),savepath)

