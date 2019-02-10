import tensorflow as tf
import reader
import numpy as np
from DCGAN import DCGAN

model = DCGAN()

noise = tf.placeholder(tf.float32, [model.batch_size, model.noise_n])
# train image
img_real = tf.placeholder(tf.float32, [model.batch_size] + model.img_shape)
# generate image
img_fake = model.generate(noise)

# image real,fake inspection
d_real = model.discriminate(img_real)
d_fake = model.discriminate(img_fake)

# d_cost want d_real to get bigger
# g_cost want d_fake to get smaller
d_cost = -tf.reduce_mean(tf.log(d_real) + tf.log(1 - d_fake))
# g_cost want d_fake to get bigger
g_cost = -tf.reduce_mean(tf.log(d_fake))

# train
d_train = tf.train.AdamOptimizer(learning_rate=model.learning_rate, beta1=0.5).minimize(d_cost,
                                                                                       var_list=model.D_var_list)
g_train = tf.train.AdamOptimizer(learning_rate=model.learning_rate, beta1=0.5).minimize(g_cost,
                                                                                       var_list=model.G_var_list)

r = reader.reader('C://Users//woals//Git_store//dataset//celeba//', model.batch_size, (64, 64))

noise_test = np.random.normal(size=(model.batch_size, model.noise_n))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(model.total_epoch):
        for step in range(r.total_batch):
            # next batch 작성
            batch_x = r.next_batch()
            batch_x = batch_x * (2.0 / 255.0) - 1

            # batch_x = cv2.resize(batch_x,(64,64), interpolation = cv2.INTER_AREA)

            z = np.random.normal(size=(model.batch_size, model.noise_n))

            _, loss_val_D = sess.run([d_train, d_cost],
                                     feed_dict={img_real: batch_x, noise: z})

            _, loss_val_G = sess.run([g_train, g_cost],
                                     feed_dict={noise: z})

            print('Epoch: [', epoch, '/', model.total_epoch, '], ', 'Step: [', step, '/', r.total_batch,
                  '], D_loss: ',
                  loss_val_D, ', G_loss: ', loss_val_G)

            if step == 0 or (step + 1) % 100 == 0:
                generated_samples = model.test_generator(noise_test, model.batch_size, sess)
                savepath = 'C://Users//woals//Git_store//dataset//dcgan_result//test_'+str(epoch)+'_'+str(step)+'.jpg'
                reader.batch_visualization(generated_samples, (14, 14), savepath)