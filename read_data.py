import numpy as np
import struct
import csv



def read_data(file_name, classification=False):
    with open(file_name, mode='r') as file:
        csv_reader = csv.reader(file)
        csv_reader.__next__()
        data = np.array(list(csv_reader), dtype=np.float32)
    if classification:
        return data.T[0:2].T, data.T[2].T
    return data.T[0:1].T, data.T[1:2].T


def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        # Read the magic number, number of images, rows, and columns
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(f'Invalid magic number {magic} in MNIST image file: {filename}')
        
        # Read the image data
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
        
    return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        # Read the magic number and number of labels
        magic, num_labels = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError(f'Invalid magic number {magic} in MNIST label file: {filename}')
        
        # Read the label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        
    return labels
