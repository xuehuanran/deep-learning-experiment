import cifar10_input

train_images, train_labels = cifar10_input.load_cifar10(True)
test_images, test_labels = cifar10_input.load_cifar10(False)

print(train_images[0:50].shape)
print(train_labels.shape)

print(test_images.shape)
print(test_labels.shape)
