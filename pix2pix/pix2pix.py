import tensorflow as tf
import reader

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
                                            reuse=tf.AUTO_REUSE     # if tensorflow vesrion < 1.4, delete this line
                                            )

class pix2pix:
    def __init__(self,
                 img_shape,
                 batch_size,
                 learning_rate = 0.0002,
                 keep_prop = 0.5):
        # hyper parameter
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.keep_prop = keep_prop
        # generate Variable(encoder-decoder)
        # ENCODING
        self.G_W1 = tf.Variable(tf.truncated_normal([4,4,3,64],stddev=0.2), name="G_W1")

        self.G_W2 = tf.Variable(tf.truncated_normal([4,4,64,128],stddev=0.2),name="G_W2")
        self.G_bn2 = batch_norm(name="G_bn2")

        self.G_W3 = tf.Variable(tf.truncated_normal([4,4,128,256],stddev=0.2),name="G_W3")
        self.G_bn3 = batch_norm(name="G_bn3")

        self.G_W4 = tf.Variable(tf.truncated_normal([4, 4, 256, 512], stddev=0.2), name="G_W4")
        self.G_bn4 = batch_norm(name="G_bn4")

        self.G_W5 = tf.Variable(tf.truncated_normal([4, 4, 512, 512], stddev=0.2), name="G_W5")
        self.G_bn5 = batch_norm(name="G_bn5")

        self.G_W6 = tf.Variable(tf.truncated_normal([4, 4, 512, 512], stddev=0.2), name="G_W6")
        self.G_bn6 = batch_norm(name="G_bn6")

        self.G_W7 = tf.Variable(tf.truncated_normal([4, 4, 512, 512], stddev=0.2), name="G_W7")
        self.G_bn7 = batch_norm(name="G_bn7")

        self.G_W8 = tf.Variable(tf.truncated_normal([4, 4, 512, 512], stddev=0.2), name="G_W8")
        self.G_bn8 = batch_norm(name="G_bn8")

        # DECODING
        self.G_W9 = tf.Variable(tf.truncated_normal([4, 4, 512, 512], stddev=0.2), name="G_W9")
        self.G_bn9 = batch_norm(name="G_bn9")

        self.G_W10 = tf.Variable(tf.truncated_normal([4, 4, 512, 512+512], stddev=0.2), name="G_W10")
        self.G_bn10 = batch_norm(name="G_bn10")

        self.G_W11 = tf.Variable(tf.truncated_normal([4, 4, 512, 512+512], stddev=0.2), name="G_W11")
        self.G_bn11 = batch_norm(name="G_bn11")

        self.G_W12 = tf.Variable(tf.truncated_normal([4, 4, 512, 512+512], stddev=0.2), name="G_W12")
        self.G_bn12 = batch_norm(name="G_bn12")

        self.G_W13 = tf.Variable(tf.truncated_normal([4, 4, 256, 512+512], stddev=0.2), name="G_W13")
        self.G_bn13 = batch_norm(name="G_bn13")

        self.G_W14 = tf.Variable(tf.truncated_normal([4, 4, 128, 256+256], stddev=0.2), name="G_W14")
        self.G_bn14 = batch_norm(name="G_bn14")

        self.G_W15 = tf.Variable(tf.truncated_normal([4, 4, 64, 128+128], stddev=0.2), name="G_W15")
        self.G_bn15 = batch_norm(name="G_bn15")

        self.G_W16 = tf.Variable(tf.truncated_normal([4, 4, 3, 64+64], stddev=0.2), name="G_W16")
        self.G_bn16 = batch_norm(name="G_bn16")

        # discriminate
        self.D_W1 = tf.Variable(tf.truncated_normal([4, 4, 3, 64],stddev=0.02), name='D_W1')
        self.D_bn1 = batch_norm(name="D_bn1")

        self.D_W2 = tf.Variable(tf.truncated_normal([4, 4, 64, 128], stddev=0.02), name='D_W2')
        self.D_bn2 = batch_norm(name="D_bn2")

        self.D_W3 = tf.Variable(tf.truncated_normal([4, 4, 128, 256], stddev=0.02), name='D_W3')
        self.D_bn3 = batch_norm(name="D_bn3")

        self.D_W4 = tf.Variable(tf.truncated_normal([4, 4, 256, 512], stddev=0.02), name='D_W4')
        self.D_bn4 = batch_norm(name="D_bn4")

        self.D_W5 = tf.Variable(tf.truncated_normal([4, 4, 512, 1], stddev=0.02), name='D_W5')

        self.G_var_list = [
            self.G_W1,
            self.G_W2,
            self.G_W3,
            self.G_W4,
            self.G_W5,
            self.G_W6,
            self.G_W7,
            self.G_W8,
            self.G_W9,
            self.G_W10,
            self.G_W11,
            self.G_W12,
            self.G_W13,
            self.G_W14,
            self.G_W15,
            self.G_W16
        ]

        self.D_var_list = [
            self.D_W1,
            self.D_W2,
            self.D_W3,
            self.D_W4,
            self.D_W5
        ]

        def generate(self,img):
            # 256x256x3 -> 128x128x64
            # padding
            # SAME : appropriate pad
            # VALID : zero pad
            layer_1 = skip_1 = tf.nn.conv2d(img,self.G_W1,strides=[1,2,2,1],padding='SAME')
            layer_1 = tf.nn.leaky_relu(layer_1,alpha=0.2)

            layer_2 = tf.nn.conv2d(layer_1,self.G_W2,strides=[1,2,2,1],padding='SAME')
            layer_2 = skip_2 = self.G_bn2(layer_2)
            layer_2 = tf.nn.leaky_relu(layer_2,alpha=0.2)

            layer_3 = tf.nn.conv2d(layer_2,self.G_W3,strides=[1,2,2,1],padding='SAME')
            layer_3 = skip_3 = self.G_bn3(layer_3)
            layer_3 = tf.nn.leaky_relu(layer_3,alpha=0.2)

            layer_4 = tf.nn.conv2d(layer_3, self.G_W4, strides=[1, 2, 2, 1], padding='SAME')
            layer_4 = skip_4 = self.G_bn4(layer_4)
            layer_4 = tf.nn.leaky_relu(layer_4, alpha=0.2)

            layer_5 = tf.nn.conv2d(layer_4, self.G_W5, strides=[1, 2, 2, 1], padding='SAME')
            layer_5 = skip_5 = self.G_bn5(layer_5)
            layer_5 = tf.nn.leaky_relu(layer_5, alpha=0.2)

            layer_6 = tf.nn.conv2d(layer_5, self.G_W6, strides=[1, 2, 2, 1], padding='SAME')
            layer_6 = skip_6 = self.G_bn6(layer_6)
            layer_6 = tf.nn.leaky_relu(layer_6, alpha=0.2)

            layer_7 = tf.nn.conv2d(layer_6, self.G_W7, strides=[1, 2, 2, 1], padding='SAME')
            layer_7 = skip_7 = self.G_bn7(layer_7)
            layer_7 = tf.nn.leaky_relu(layer_7, alpha=0.2)

            layer_8 = tf.nn.conv2d(layer_7, self.G_W8, strides=[1, 2, 2, 1], padding='SAME')
            layer_8 = self.G_bn8(layer_8)
            layer_8 = tf.nn.relu(layer_8)

            # 1x1x512 -> 2x2x512
            layer_9 = tf.nn.conv2d_transpose(layer_8,
                                             self.G_W9,
                                             output_shape=[self.batch_size,2,2,512],
                                             strides=[1,2,2,1])
            layer_9 = tf.nn.dropout(self.G_bn9(layer_9), keep_prob=self.keep_prob)
            layer_9 = tf.nn.relu(layer_9)
            layer_9 = tf.concat([layer_9,skip_7], axis=3)

            layer_10 = tf.nn.conv2d_transpose(layer_9,
                                             self.G_W10,
                                             output_shape=[self.batch_size, 2, 2, 512],
                                             strides=[1, 2, 2, 1])
            layer_10 = tf.nn.dropout(self.G_bn10(layer_10), keep_prob=self.keep_prob)
            layer_10 = tf.nn.relu(layer_10)
            layer_10 = tf.concat([layer_10, skip_6], axis=3)

            layer_11 = tf.nn.conv2d_transpose(layer_10,
                                              self.G_W11,
                                              output_shape=[self.batch_size, 8, 8, 512],
                                              strides=[1, 2, 2, 1])  # [?,4,4,512+512] -> [?,8,8,512]
            layer_11 = tf.nn.dropout(self.G_bn11(layer_11), keep_prob=self.keep_prob)
            layer_11 = tf.nn.relu(layer_11)
            layer_11 = tf.concat([layer_11, skip_5], axis=3)

            layer_12 = tf.nn.conv2d_transpose(layer_11,
                                              self.G_W12,
                                              output_shape=[self.batch_size, 16, 16, 512],
                                              strides=[1, 2, 2, 1])  # [?,8,8,512+512] -> [?,16,16,512]
            layer_12 = self.G_bn12(layer_12)
            layer_12 = tf.nn.relu(layer_12)
            layer_12 = tf.concat([layer_12, skip_4], axis=3)

            layer_13 = tf.nn.conv2d_transpose(layer_12,
                                              self.G_W13,
                                              output_shape=[self.batch_size, 32, 32, 256],
                                              strides=[1, 2, 2, 1])  # [?,16,16,512+512] -> [?,32,32,256]
            layer_13 = self.G_bn13(layer_13)
            layer_13 = tf.nn.relu(layer_13)
            layer_13 = tf.concat([layer_13, skip_3], axis=3)

            layer_14 = tf.nn.conv2d_transpose(layer_13,
                                              self.G_W14,
                                              output_shape=[self.batch_size, 64, 64, 128],
                                              strides=[1, 2, 2, 1])  # [?,32,32,256+256] -> [?,64,64,128]
            layer_14 = self.G_bn14(layer_14)
            layer_14 = tf.nn.relu(layer_14)
            layer_14 = tf.concat([layer_14, skip_2], axis=3)

            layer_15 = tf.nn.conv2d_transpose(layer_14,
                                              self.G_W15,
                                              output_shape=[self.batch_size, 128, 128, 64],
                                              strides=[1, 2, 2, 1])  # [?,64,64,128+128] -> [?,128,128,64]
            layer_15 = self.G_bn15(layer_15)
            layer_15 = tf.nn.relu(layer_15)
            layer_15 = tf.concat([layer_15, skip_1], axis=3)

            layer_16 = tf.nn.conv2d_transpose(layer_15, self.G_W16,
                                         output_shape=[self.batch_size, 256, 256, 3],
                                         strides=[1, 2, 2, 1])  # [?,128,128,64+64] -> [?,256,256,3]
            output_ = tf.nn.tanh(layer_16)

            return output_

        def discriminate(self,img,target):
            # img : real_img
            # target : trans_img
            img_concat = tf.concat([img,target],axis=3) # channel concat

            layer_1 = tf.nn.conv2d(img_concat,self.D_W1,strides=[1,2,2,1],padding='SAME')
            layer_1 = tf.nn.leaky_relu(self.D_bn1(layer_1),alpha=0.2)

            layer_2 = tf.nn.conv2d(layer_1,self.D_W2,strides=[1,2,2,1],padding='SAME')
            layer_2 = tf.nn.leaky_relu(self.D_bn2(layer_2),alpha=0.2)

            layer_3 = tf.nn.conv2d(layer_2,self.D_W3,strides=[1,2,2,1],padding='SAME')
            layer_3 = tf.nn.leaky_relu(self.D_bn3(layer_3),alpha=0.2)

            layer_4 = tf.pad(layer_3,[[0,0],[1,1],[1,1],[0,0]],mode='CONSTANT')
            layer_4 = tf.nn.conv2d(layer_4,self.D_W4,strides=[1,1,1,1],padding='VALID')
            layer_4 = tf.nn.leaky_relu(self.D_bn4(layer_4),alpha=0.2)

            layer_5 = tf.pad(layer_4,[[0,0],[1,1],[1,1],[0,0]],mode='CONSTANT')
            layer_5 = tf.nn.conv2d(layer_5,self.D_W5,strides=[1,1,1,1],padding='VALID')

            output_ = tf.nn.sigmoid(layer_5)

            return output_

        


















