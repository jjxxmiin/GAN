from tqdm import tqdm
import request
import os
import math
import zipfile

class dataset:
    def __init__(self,url,filename):
        self.url = url
        self.filename = filename
        self.current = '//'.join(os.getcwd().split('\\'))+'//'+filename

    def download(self):
        if os.path.exists(self.current):
            print('already existing file')
            return

        wrote = 0
        chunkSize = 1024
        r = request.get(self.url, stream=True)
        total_size = int(r.headers['Content-Length'])
        with open(self.filename, 'wb') as f:
            for data in tqdm(r.iter_content(chunkSize), total=math.ceil(total_size // chunkSize), unit='KB',
                             unit_scale=True):
                wrote = wrote + len(data)
                f.write(data)
        print('download success')
        return

    def upzip(self,savepath='.'):
        if os.path.isdir(self.current[:-4]):
            print('already existing folder')
            return

        with zipfile.ZipFile(self.filename,'r') as zf:
            zf.extractall(path=savepath)
            zf.close()
        print('unzip success')

'''
url = "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/vangogh2photo.zip"
filename = url.split("/")[-1]

database = dataset(url,filename)

flag = database.download()

database.upzip(savepath='.')
'''
