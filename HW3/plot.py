import matplotlib.pyplot as plt
import re

logs = ''''
 "{'rouge1': 9.4912, 'rouge2': 3.5683, 'rougeL': 9.3979, 'rougeLsum': 9.3989, 'train_loss': 5.781361905282793, 'epoch': 0, 'step': 5428}\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'rouge1': 12.131, 'rouge2': 4.3663, 'rougeL': 11.9479, 'rougeLsum': 11.9661, 'train_loss': 4.318649393192705, 'epoch': 1, 'step': 10856}\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'rouge1': 14.0804, 'rouge2': 4.7032, 'rougeL': 13.8274, 'rougeLsum': 13.8531, 'train_loss': 3.9090253229780765, 'epoch': 2, 'step': 16284}\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'rouge1': 15.5882, 'rouge2': 5.2699, 'rougeL': 15.426, 'rougeLsum': 15.4161, 'train_loss': 3.6867630803242446, 'epoch': 3, 'step': 21712}\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'rouge1': 16.4666, 'rouge2': 5.8185, 'rougeL': 16.2577, 'rougeLsum': 16.2436, 'train_loss': 3.5433681547070743, 'epoch': 4, 'step': 27140}\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'rouge1': 17.8164, 'rouge2': 6.0981, 'rougeL': 17.5278, 'rougeLsum': 17.5065, 'train_loss': 3.4317784376151437, 'epoch': 5, 'step': 32568}\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'rouge1': 17.6566, 'rouge2': 5.8658, 'rougeL': 17.4015, 'rougeLsum': 17.3709, 'train_loss': 3.3363174742078114, 'epoch': 6, 'step': 37996}\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'rouge1': 17.982, 'rouge2': 5.9127, 'rougeL': 17.7422, 'rougeLsum': 17.7217, 'train_loss': 3.2611750558446944, 'epoch': 7, 'step': 43424}\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'rouge1': 18.9385, 'rouge2': 6.2976, 'rougeL': 18.6602, 'rougeLsum': 18.6332, 'train_loss': 3.193918901413965, 'epoch': 8, 'step': 48852}\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'rouge1': 18.9934, 'rouge2': 6.3941, 'rougeL': 18.6893, 'rougeLsum': 18.7006, 'train_loss': 3.1332860284635227, 'epoch': 9, 'step': 54280}\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'rouge1': 19.1648, 'rouge2': 6.6142, 'rougeL': 18.877, 'rougeLsum': 18.8476, 'train_loss': 3.0817002118644066, 'epoch': 10, 'step': 59708}\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'rouge1': 19.0196, 'rouge2': 6.2345, 'rougeL': 18.7736, 'rougeLsum': 18.7642, 'train_loss': 3.0320031117584745, 'epoch': 11, 'step': 65136}\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'rouge1': 19.2921, 'rouge2': 6.3418, 'rougeL': 18.9581, 'rougeLsum': 18.9881, 'train_loss': 2.9860901013840273, 'epoch': 12, 'step': 70564}\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'rouge1': 19.6922, 'rouge2': 6.7439, 'rougeL': 19.412, 'rougeLsum': 19.4406, 'train_loss': 2.942390548717299, 'epoch': 13, 'step': 75992}\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'rouge1': 19.3774, 'rouge2': 6.5399, 'rougeL': 19.0788, 'rougeLsum': 19.0835, 'train_loss': 2.9029545513425754, 'epoch': 14, 'step': 81420}\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'rouge1': 19.7519, 'rouge2': 6.6949, 'rougeL': 19.4378, 'rougeLsum': 19.4534, 'train_loss': 2.870513893526621, 'epoch': 15, 'step': 86848}\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'rouge1': 20.2676, 'rouge2': 6.7198, 'rougeL': 19.93, 'rougeLsum': 19.9476, 'train_loss': 2.833943894793202, 'epoch': 16, 'step': 92276}\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'rouge1': 20.4829, 'rouge2': 6.745, 'rougeL': 20.1627, 'rougeLsum': 20.1738, 'train_loss': 2.803805211403832, 'epoch': 17, 'step': 97704}\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'rouge1': 20.3575, 'rouge2': 6.660, 'rougeL': 20.0443, 'rougeLsum': 20.0833, 'train_loss': 2.775142886134396, 'epoch': 18, 'step': 103132}\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'rouge1': 20.2985, 'rouge2': 6.6491, 'rougeL': 20.0076, 'rougeLsum': 19.9742, 'train_loss': 2.746022145011975, 'epoch': 19, 'step': 108560}\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'rouge1': 20.0878, 'rouge2': 6.7704, 'rougeL': 19.793, 'rougeLsum': 19.7888, 'train_loss': 2.7225772686187364, 'epoch': 20, 'step': 113988}\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'rouge1': 20.6063, 'rouge2': 7.0399, 'rougeL': 20.2787, 'rougeLsum': 20.3036, 'train_loss': 2.6969754268952655, 'epoch': 21, 'step': 119416}\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'rouge1': 21.0947, 'rouge2': 7.0581, 'rougeL': 20.7456, 'rougeLsum': 20.7439, 'train_loss': 2.677636682767594, 'epoch': 22, 'step': 124844}\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'rouge1': 20.6967, 'rouge2': 6.8172, 'rougeL': 20.388, 'rougeLsum': 20.4159, 'train_loss': 2.6522012596720708, 'epoch': 23, 'step': 130272}\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'rouge1': 20.6984, 'rouge2': 6.815, 'rougeL': 20.3453, 'rougeLsum': 20.3842, 'train_loss': 2.6374795979757737, 'epoch': 24, 'step': 135700}\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'rouge1': 20.6644, 'rouge2': 6.9661, 'rougeL': 20.3355, 'rougeLsum': 20.3592, 'train_loss': 2.6168753526275794, 'epoch': 25, 'step': 141128}\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'rouge1': 20.6882, 'rouge2': 6.764, 'rougeL': 20.3629, 'rougeLsum': 20.3707, 'train_loss': 2.602114470108696, 'epoch': 26, 'step': 146556}\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'rouge1': 20.6858, 'rouge2': 6.6541, 'rougeL': 20.3562, 'rougeLsum': 20.3582, 'train_loss': 2.585500313766581, 'epoch': 27, 'step': 151984}\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'rouge1': 21.1012, 'rouge2': 6.9369, 'rougeL': 20.79, 'rougeLsum': 20.7865, 'train_loss': 2.5748973781779663, 'epoch': 28, 'step': 157412}\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'rouge1': 20.8205, 'rouge2': 6.772, 'rougeL': 20.4703, 'rougeLsum': 20.493, 'train_loss': 2.5643171115051584, 'epoch': 29, 'step': 162840}\n"
          ]
{'rouge1': 20.8482, 'rouge2': 6.871, 'rougeL': 20.5821, 'rougeLsum': 20.622, 'train_loss': 2.5551408783161387, 'epoch': 30, 'step': 5428}
{'rouge1': 20.9713, 'rouge2': 6.8953, 'rougeL': 20.6669, 'rougeLsum': 20.6833, 'train_loss': 2.5446554252832536, 'epoch': 31, 'step': 10856}
{'rouge1': 21.0511, 'rouge2': 6.8832, 'rougeL': 20.7284, 'rougeLsum': 20.7519, 'train_loss': 2.5387866948576825, 'epoch': 32, 'step': 16284}
{'rouge1': 21.2065, 'rouge2': 6.925, 'rougeL': 20.9199, 'rougeLsum': 20.9464, 'train_loss': 2.532795444339536, 'epoch': 33, 'step': 21712}
{'rouge1': 21.0784, 'rouge2': 6.8636, 'rougeL': 20.7679, 'rougeLsum': 20.7995, 'train_loss': 2.5283467595684415, 'epoch': 34, 'step': 27140}          
'''

logs.replace('rougeLsum', 'rouge-Lsum')
r1_lst = [m.end() for m in re.finditer('rouge1', logs)]
r2_lst = [m.end() for m in re.finditer('rouge2', logs)]
rl_lst = [m.end() for m in re.finditer('rougeLsum', logs)]

r1_result = []
r2_result = []
rl_result = []

for r1, r2, rl in zip(r1_lst, r2_lst, rl_lst):
    r1_result.append(float(re.sub('[ ,}]', '', logs[r1+3:][:7])))
    r2_result.append(float(re.sub('[ ,}]', '', logs[r2+3:][:7])))
    rl_result.append(float(re.sub('[ ,}]', '', logs[rl+3:][:7])))

epoch = list(range(len(r1_result)))

plt.plot(epoch, r1_result, label = 'rouge-1', marker = 'o')
plt.yticks(range(round(min(r1_result)) - 1, round(max(r1_result)) + 2,1))
plt.xlabel('epoch')
plt.ylabel('rouge-1')
plt.legend()
plt.show()
plt.plot(epoch, r2_result, label = 'rouge-2', marker = 'o', color='green')
plt.yticks(range(round(min(r2_result)) - 1, round(max(r2_result)) + 2,1))
plt.xlabel('epoch')
plt.ylabel('rouge-2')
plt.legend()
plt.show()
plt.plot(epoch, rl_result, label = 'rouge-L', marker = 'o', color='orange')
plt.yticks(range(round(min(rl_result)) - 1, round(max(rl_result)) + 2,1))
plt.xlabel('epoch')
plt.ylabel('rouge-L')
plt.legend()
plt.show()

