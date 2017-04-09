import matplotlib.pyplot as plt
import numpy as np

loss0 = np.load('output/loss0.npy')
loss1 = np.load('output/loss1.npy')
loss2 = np.load('output/loss2.npy')
loss3 = np.load('output/loss3.npy')

plt.figure(1)
curve0, = plt.semilogy(loss0, 'r')
curve1, = plt.semilogy(loss1, 'b')
curve2, = plt.semilogy(loss2, 'g')
curve3, = plt.semilogy(loss3, 'y')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right', handles=[curve0, curve1, curve2, curve3], labels=['Momentum', 'Modified_Momentum', 'Nesterov', 'Modified_Nesterov'])
plt.grid(True)
plt.show()
