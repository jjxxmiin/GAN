import os
import scipy.misc
import numpy as np
import time


class reader:
    def __init__(self, dir_name, batch_size,resize):
        self.dir_name = dir_name
        self.batch_size = batch_size
        self.resize = resize
        # file list
        self.file_list = os.listdir(self.dir_name)
        # total batch num
        self.leng = len(self.file_list)
        self.total_batch = self.leng // self.batch_size
        # index
        self.index = 0
        # shuffle
        np.random.shuffle(self.file_list)

    def getList(self):
        return self.file_list

    def getTotalNum(self):
        return len(self.file_list)

    def next_batch(self):
        if self.index == self.total_batch:
            np.random.shuffle(self.file_list)
            self.index = 0

        # image random choice
        batch = []

        file_list_batch = self.file_list[self.index * self.batch_size:(self.index + 1) * self.batch_size]
        self.index += 1

        # 6331번
        for file_name in file_list_batch:
            dir_n = self.dir_name + file_name
            img = scipy.misc.imread(dir_n)
            res = scipy.misc.imresize(img, self.resize)
            batch.append(res)

        return np.array(batch).astype(np.float32)


def batch_visualization(X,nh_nw,path):
    nh, nw = nh_nw
    h, w = X.shape[1], X.shape[2]
    img = np.zeros((h * nh, w * nw, 3))

    for n, x in enumerate(X):
        j = int(n / nw)
        i = int(n % nw)
        img[j * h:j * h + h, i * w:i * w + w, :] = x

    scipy.misc.imsave(path,img)


'''
dir_n = 'C://Users//woals//Git_store//mygan//celeba//'
batch_size = 64
resize = (64,64)

batch_image = reader(dir_n,batch_size,resize).next_batch()

batch_visualization(batch_image,(14,14))

print(batch_image[0].shape)
print(batch_image[0])
'''