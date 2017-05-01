import pickle
import numpy as np

train_file = '../dataset/cifar100/cifar-100-python/train'
test_file = '../dataset/cifar100/cifar-100-python/test'
train_num = 50000
test_num = 10000
class_num = 100


def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


class Cifar100(object):
    pass


def read_cifar100():
    train_dict = unpickle(train_file)
    test_dict = unpickle(test_file)

    cifar100 = Cifar100()
    cifar100.train_images = train_dict[b'data']
    cifar100.test_images = test_dict[b'data']

    train_labels = np.zeros([train_num, class_num])
    test_labels = np.zeros([test_num, class_num])
    for i in range(train_num):
        label = train_dict[b'fine_labels'][i]
        train_labels[i][label] = 1
    for i in range(test_num):
        label = test_dict[b'fine_labels'][i]
        test_labels[i][label] = 1

    cifar100.train_labels = train_labels
    cifar100.test_labels = test_labels

    return cifar100
