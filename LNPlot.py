import numpy as np
import scipy.stats as stats
import matplotlib.pylab as plt

Priorproject = np.genfromtxt('PriorProj_May7.csv',delimiter=',')
STAproject = np.genfromtxt('STAproject_May7.csv',delimiter=',')
Mode1project = np.genfromtxt('Mode1project_May7.csv',delimiter=',')
Mode1project = Mode1project*-1
Mode2project = np.genfromtxt('Mode2project_May7.csv',delimiter=',')
Mode2project = Mode2project*-1

plt.figure()
plt.hist(Priorproject, normed=True, color = 'gray')
plt.hist(STAproject, normed=True, color='k')
plt.hist(Mode1project,normed=True, color='b')
plt.hist(Mode2project,normed=True, color='g')
plt.xlabel('projection',fontsize=20)
plt.ylabel('density',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(('Prior','STA','Mode1','Mode2'),loc='upper right')
plt.tight_layout()
plt.show()

Prior_PDF = stats.kde.gaussian_kde(Priorproject)
STA_PDF = stats.kde.gaussian_kde(STAproject)
Mode1_PDF = stats.kde.gaussian_kde(Mode1project)
Mode2_PDF = stats.kde.gaussian_kde(Mode2project)

x = np.linspace(-20,20,1000)
plt.figure()
plt.plot(x,Prior_PDF(x), linewidth=4,color='grey')
plt.plot(x,STA_PDF(x), linewidth=4,color='k')
plt.plot(x,Mode1_PDF(x), linewidth=4,color='b')
plt.plot(x,Mode2_PDF(x), linewidth=4,color='g')
plt.xlabel('projection',fontsize=20)
plt.ylabel('density',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(('Prior','STA','Mode1','Mode2'),loc='upper right')
plt.tight_layout()
plt.show()
