import tensorflow as tf
from reader import reader
import scipy.misc
from pix2pix import pix2pix
import cv2
import numpy as np

def split_image(img):
    tmp = np.split(img,2,axis=2)
    img_A = tmp[0]
    img_B = tmp[1]

    return img_A,img_B

# input_normalization
def input_normalization(img):
    return img * (2.0 / 255.0) - 1

#init
total_epoch = 5
resize_shape = [256,512,3]
image_shape = [256,256,3]
batch_size = 128
learning_rate = 0.0002
EPS = 1e-12

# pix2pix model
model = pix2pix(img_shape=resize_shape,batch_size=batch_size,learning_rate=learning_rate)

# dataset
data = reader(dir_name="G:/dataset/maps/train/",batch_size=batch_size,resize=resize_shape)

# totalbatch
total_batch = data.total_batch

# input image
input_img = tf.placeholder(tf.float32, [batch_size] + image_shape)
# target_image
target_img = tf.placeholder(tf.float32, [batch_size] + image_shape)

# generate image
gen_img = model.generate(input_img)

# discriminate
# (input vs target)
# (input vs gen)
d_real = model.discriminate(input_img, target_img)
d_fake = model.discriminate(input_img, gen_img)

# discriminate cost
d_cost = -tf.reduce_mean(tf.log(d_real + EPS) + tf.log(1 - d_fake + EPS))

# generate cost
g_cost_GAN = -tf.reduce_mean(tf.log(d_fake + EPS))
g_cost_L1 = tf.reduce_mean(tf.abs(target_img - gen_img))
g_cost = g_cost_GAN + g_cost_L1 * 100

d_train = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(d_cost, var_list=model.D_var_list)
g_train = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(g_cost, var_list=model.G_var_list)

# Variable
global_epoch = tf.Variable(0,trainable=False,name='global_step')
# new Variable
global_epoch_decrease = tf.assign(global_epoch,tf.add(global_epoch,1))

with tf.Session() as sess:
    # model save
    saver = tf.train.Saver(tf.global_variables())

    # call model
    ckpt = tf.train.get_checkpoint_state('./save_dir')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    epoch = sess.run(global_epoch)

    while True:
        if epoch == total_epoch:
            break
        for step in range(total_batch):
            batch = data.next_batch()
            real, target = split_image(batch)

            real = input_normalization(real)
            target = input_normalization(input)

            _, loss_val_D = sess.run([d_train,d_cost],feed_dict={input_img:real,
                                                             target_img:target})
            _, loss_val_GAN, loss_val_L1 = sess.run([g_train,g_cost_GAN,g_cost_L1],feed_dict={input_img:real,
                                                                                      target_img:target})
            print('Epoch: [', epoch, '/', total_epoch, '], ', 'Step: [', step, '/', total_batch,
                  '], D_loss: ',
                  loss_val_D, ', GAN_loss: ', loss_val_GAN, ', L1_loss: ', loss_val_L1)

            if step == 0 or (step + 1) % 100 == 0:



'''
batch = data.next_batch()

print(batch.shape)

scipy.misc.imsave("./test.jpg",batch[0])
'''





