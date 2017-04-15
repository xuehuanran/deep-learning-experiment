import matplotlib.pyplot as plt
import numpy as np

loss0 = np.load('output/loss0.npy')
loss1 = np.load('output/loss1.npy')

plt.figure(1)
curve0, = plt.semilogy(loss0, 'r')
curve1, = plt.semilogy(loss1, 'b')

plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right', handles=[curve0, curve1], labels=['Momentum', 'Modified_Momentum'])
plt.grid(True)
plt.show()
