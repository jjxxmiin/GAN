import os
import scipy.misc as sm
import numpy as np
import time

class reader:
    def __init__(self, dir_name, batch_size, resize):
        self.dir_name = dir_name
        self.batch_size = batch_size
        #file list
        self.file_list = os.listdir(self.dir_name)
        #total batch num
        self.leng = len(self.file_list)
        self.total_batch = self.leng // self.batch_size
        #index
        self.index = 0
        #shuffle
        np.random.shuffle(self.file_list)
        
    
    def getList(self):
        return self.file_list
    
    def getTotalNum(self):
        return len(self.file_list)
    
    def next_batch(self):
        if self.index == self.total_batch:
            np.random.shuffle(self.file_list)
            self.index = 0
        
        #image random choice
        batch = []
        
        file_list_batch = self.file_list[self.index*self.batch_size:(self.index+1)*self.batch_size]
        self.index += 1
        
        start = time.time()
        
        #6331번
        for file_name in file_list_batch:
                dir_n = self.dir_name + file_name
                img = sm.imread(dir_n)
                res = sm.imresize(img,(64,64))
                batch.append(res)
        
        end = time.time()
        
        return np.array(batch)

