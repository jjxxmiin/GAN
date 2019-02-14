import tensorflow as tf
from reader import reader
import scipy.misc
from pix2pix import pix2pix

def split(img):
    img = reader.next_batch()

model = pix2pix()





