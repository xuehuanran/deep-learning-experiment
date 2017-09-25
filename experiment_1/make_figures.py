import matplotlib.pyplot as plt
import numpy as np

loss0 = np.load('output/loss0.npy')
loss1 = np.load('output/loss1.npy')

data0 = []
data1 = []

for i in range(0, 1100 * 200, 1100):
    data0.append(loss0[i])
    data1.append(loss1[i])

x = []
for i in range(0, 1100 * 200, 1100):
    x.append(i / 1100)

plt.figure(1)
curve0, = plt.semilogy(x, data0, 'r')
curve1, = plt.semilogy(x, data1, 'b')

plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right', handles=[curve0, curve1], labels=['Momentum', 'Modified_Momentum'])
plt.grid(True)
plt.show()
