import matplotlib.pyplot as plt
import numpy as np

#x=[2,4,6,8,10,12,14,16,18,20,22]
#y1=[0.706, 0.724, 0.732, 0.729, 0.720, 0.727, 0.735, 0.738, 0.732, 0.736, 0.741]
#y2=[0.709, 0.726, 0.717, 0.719, 0.731, 0.730, 0.726, 0.731, 0.745, 0.734, 0.742]
x=[2,4,6,8,10,12,14,16,18,20,22]
y1=[0.706, 0.724, 0.732, 0.729, 0.720, 0.727, 0.735, 0.738, 0.732, 0.736, 0.741]
y2=[0.709, 0.726, 0.717, 0.719, 0.731, 0.730, 0.726, 0.731, 0.745, 0.734, 0.742]

x_short=[2,4,6,8,14,16,20,22]
y1_short=[0.706, 0.724, 0.732, 0.729, 0.735, 0.738, 0.736, 0.741]
line=plt.plot(x_short, y1_short,'r',linewidth=3.0)
plt.xlabel('Number of Training Examples', fontsize=20)
plt.ylabel('Performance (F1)', fontsize=20)
plt.ylim(0.65, 0.80)
plt.xlim(2,22)
#plt.legend(['bug v.s. others'], loc='upper left')
plt.title('Learning Curve for Bug Label Classification', fontsize=20)
plt.xticks(x, ('20k', '40k', '60k', '80k', '100k','120k','140k','160k','180k','200k','220k') )
plt.show()

x_short=[2,6,8,12,14,16,20,22]
y2_short=[0.709, 0.717, 0.719, 0.730, 0.726, 0.731, 0.734, 0.742]
plt.plot(x_short, y2_short,'r',linewidth=3.0)
plt.xlabel('Number of Training Examples', fontsize=20)
plt.ylabel('Performance (F1)', fontsize=20)
plt.ylim(0.65, 0.80)
plt.xlim(2,22)
#plt.legend(['bug v.s. others'], loc='upper left')
plt.title('Learning Curve for Enhancement Label Classification', fontsize=20)
plt.xticks(x, ('20k', '40k', '60k', '80k', '100k','120k','140k','160k','180k','200k','220k') )
plt.show()