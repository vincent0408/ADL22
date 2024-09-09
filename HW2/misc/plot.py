import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

loss = [20105.3359375,5471.0380859375,3968.02490234375,3288.822509765625,2946.62255859375]
no_train = [33982.8984375,31493.61328125,30668.8828125,30127.8046875,29795.220703125]
epochs = [1, 2, 3, 4, 5]
acc = [0.4280397295951843,0.785871684551239,0.8334236145019531,0.8587172031402588,0.870135486125946]
no_acc = [0.1802709996700287,0.1790785789489746,0.1769466996192932, 0.17842817306518555,0.17759710550308228]

plt.plot(epochs,acc, marker='o')  
#plt.plot(epochs,acc,label='Pretrained ', marker='o')  
#plt.plot(epochs,no_acc,label='Not Pretrained',marker='o')  
plt.legend()
plt.xlabel("Epochs")
plt.xticks(np.arange(1, 6))
plt.ylabel("EM")
plt.show()

plt.plot(epochs,loss, marker='o')  
#plt.plot(epochs,loss,label='Pretrained', marker='o')  
#plt.plot(epochs,no_train,label='Not Pretrained', marker='o')  
plt.legend()
plt.xlabel("Epochs")
plt.xticks(np.arange(1, 6))
plt.ylabel("Loss")
plt.show()


