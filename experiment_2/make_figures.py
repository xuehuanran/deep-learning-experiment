import matplotlib.pyplot as plt
import numpy as np

points_num = 50
interval = 1

loss0 = np.load('output/momentum/loss.npy')
loss1 = np.load('output/momentum_modified/loss.npy')

data0 = []
data1 = []
for i in range(0, points_num, interval):
    data0.append(loss0[i])
    data1.append(loss1[i])
x = []
for i in range(0, points_num, interval):
    x.append(i / 1000)

plt.figure(1)
curve0, = plt.semilogy(x, data0, 'r')
curve1, = plt.semilogy(x, data1, 'b')

plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right', handles=[curve0, curve1], labels=['momentum', 'momentum_modified'])
plt.grid(True)
plt.show()
