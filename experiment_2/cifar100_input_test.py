import cifar100_input

cifar100 = cifar100_input.read_cifar100()

print(cifar100.train_images.shape)
print(cifar100.train_labels.shape)

print(cifar100.test_images.shape)
print(cifar100.test_labels.shape)
