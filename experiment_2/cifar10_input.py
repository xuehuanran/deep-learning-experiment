import os
import pickle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def unpickle(filename):
    with open(filename, 'rb') as fo:
        dict_data = pickle.load(fo, encoding='bytes')
    return dict_data


def distort(image, is_train=True):
    image = np.reshape(image, (3, 32, 32))
    image = np.transpose(image, (1, 2, 0))
    image = image.astype(float)
    return image


def shuffle(images, labels):
    perm = np.arange(len(labels))
    np.random.shuffle(perm)
    return np.asarray(images)[perm], np.asarray(labels)[perm]


def one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


# public interface load cifar10 data
def load_cifar10(is_train=True):
    if is_train:
        filenames = [ROOT + "/dataset/cifar10/cifar-10-batches-py/data_batch_%d" % j for j in range(1, 6)]
    else:
        filenames = [ROOT + "/dataset/cifar10/cifar-10-batches-py/test_batch"]
    images, labels = [], []
    for filename in filenames:
        cifar10 = unpickle(filename)
        for i in range(len(cifar10[b'labels'])):
            images.append(distort(cifar10[b'data'][i], is_train))
        labels += cifar10[b'labels']
    images, labels = shuffle(images, labels)
    images = images / 255.0
    labels = one_hot(labels, 10)
    return images, labels
