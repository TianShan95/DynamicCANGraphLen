import re
import matplotlib.pyplot as plt


# epochs = []
lenCANS = []
q1_loss_list = []
acc_list = []
with open('/Users/aaron/Desktop/aaa/Rl_20220227_235728_multiDim_log 2/Rl_20220227_235728.log') as f:
    for line in f:
        # if 'epoch: 31' in line:
            # print(line)
        try:
            # print(line)
            epoch = int(re.search(r'epoch: (\d+)', line).group()[7:])
            lenCAN = int(re.search(r'len: (\d+)', line).group()[5:])
            q1_loss = float(re.findall(r'avg_Q1_loss: (\d+\.\d+);', line)[0])
            acc = float(re.findall(r'acc: (\d+\.\d+) ;', line)[0])
        except AttributeError:
            continue


        # epochs.append(epoch)
        lenCANS.append(lenCAN)
        q1_loss_list.append(q1_loss)
        acc_list.append(acc)
        # print(lenCANS)
        # print(epochs)
        # break


x = [i for i in range(len(q1_loss_list))]
# plt.plot(epochs, lenCANS)
# plt.plot(x, lenCANS)
plt.plot(x, acc_list)

print('over')
plt.show()