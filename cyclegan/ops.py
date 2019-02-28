# GAN OPS
# Loss,Activation,Normalization,Layer

import tensorflow as tf
import tensorflow.contrib as tf_contrib

weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
# weight_init = tf.truncated_normal(mean=0.0,stddev=0.2)
weight_regularizer = None

# ===================================== layer =================================================
def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, scope='conv'):
    with tf.variable_scope(scope):
        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='CONSTANT')
            # 0000000
            # 0012300
            # 0078900
            # 0000000

        if pad_type == 'reflect' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')
            # 9878987
            # 3212321
            # 9878987
            # 3212321

        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias)

        return x

def deconv(x, channels, kernel=3, stride=2, use_bias=True, scope='deconv_0') :
    with tf.variable_scope(scope):
        x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                       kernel_size=kernel, kernel_initializer=weight_init,
                                       kernel_regularizer=weight_regularizer,
                                       strides=stride, use_bias=use_bias, padding='SAME')

        return x

# ===================================== Activation ================================================
def lrelu(x, alpha=0.1):
    #tf.nn.leaky_relu(x,alpha=alpha)
    return tf.maximum(x, alpha*x)

def relu(x):
    #tf.nn.relu(x)
    return tf.maximum(x,0)

def sigmoid(x):
    # tf.nn.sigmoid(x)
    return 1 / (1+tf.exp(-x))

def tanh(x):
    return tf.tanh(x)

# Normalization
def instance_norm(x,scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True,
                                           scale=True,
                                           scope=scope)

def batch_norm(x,scope='batch_norm'):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-05,
                                        scale=True,
                                        is_training=True,
                                        scope=scope,
                                        reuse=tf.AUTO_REUSE  # if tensorflow vesrion < 1.4, delete this line
                                        )

# ===================================== Residual block ================================================
def resblock(x,filter,scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            y = conv(x,filter,kernel=3,stride=1,pad=1,pad_type='reflect',use_bias=True)
            y = instance_norm(x)
            y = relu(x)

        with tf.variable_scope('res2'):
            y = conv(y,filter,kernel=3,stride=1,pad=1,pad_type='reflect',use_bias=True)
            y = instance_norm(x)

        return x + y

# ===================================== Loss ================================================
# d_cost real=up fake=down
def d_cost(real,fake):
    real_cost = tf.reduce_mean(tf.squared_difference(real,1.0))
    fake_cost = tf.reduce_mean(tf.square(fake))

    return real_cost + fake_cost

# g_cost fake up
def g_cost(fake):
    fake_cost = tf.reduce_mean(tf.squared_difference(fake, 1.0))

    return fake_cost

def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss

def L2_loss(x, y):
    loss = tf.reduce_mean(tf.pow((x - y),2))

    return loss

