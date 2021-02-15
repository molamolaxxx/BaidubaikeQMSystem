import re
import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl
import matplotlib.font_manager as font_manager

songTi = font_manager.FontProperties(fname='/usr/local/share/fonts/s/simhei.ttf')
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111)
# 字体

ax.set_title('多决策模型评分模式下的ROC曲线', fontsize=26, fontproperties=songTi)
plt.ylabel("准确率", fontproperties=songTi, fontsize=26)
plt.xlabel("训练迭代次数", fontproperties=songTi, fontsize=26)
plt.tick_params(labelsize=20)
#
# plt.ylim((0, 100))
# plt.xlim((0,3000))
# 画图
# plt.plot(x,y1, 'o-',color="green", markersize=3, label="BERT-CRF")
x = [300,600,900,1500,2100,2700]
y1 = [295,542,696,783,805,816]
y2 = [5,58,204,717,1295,1884]
plt.plot(x, y1, 'o-', color="blue", markersize=3, label="BERT+BiLSTM-CRF")
plt.plot(x, y2, 'o-', color="red", markersize=3, label="BERT+ConvEncoder-CRF")
# plt.plot(x, y4, 'o-', color="black", markersize=3, label="BERT+FCEncoder-CRF")
plt.xticks([0,300,600,900,1500,2100,2700])
plt.yticks(np.arange(0,200,2000))
plt.legend(prop= {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18,
        },loc='lower right')
plt.show()