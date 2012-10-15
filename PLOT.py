import numpy as np
import matplotlib.pylab as plt
import matplotlib.font_manager as fm

STA = np.genfromtxt('STA_May7.csv', delimiter=',')
Eigen = np.genfromtxt('EigVal_May7.csv',delimiter=',')
Modes = np.genfromtxt('EigVect_May7.csv',delimiter=',', usecols=[0,1])
current = np.genfromtxt('CURRENT1e4-2e4.csv',delimiter=',')
voltage = np.genfromtxt('voltage1e4-2e4.csv',delimiter=',')
#CovMat = np.genfromtxt('CovMat.csv',delimiter=',')
oldSTA = np.genfromtxt('oldSTA.csv',delimiter=',') # for comparison w/ model STA
ModelSTA = np.genfromtxt('ModelSTA.csv',delimiter=',')

STA_TIME = 200
INTSTEP = 0.1


#plt.figure()
#plt.imshow(CovMat)
#plt.show()

fig = plt.figure()
X = np.arange(-STA_TIME/INTSTEP,STA_TIME/INTSTEP,dtype=float)*INTSTEP
ax = fig.add_subplot(111)
ax.plot(X[0:(STA_TIME/INTSTEP)], STA[0:(STA_TIME/INTSTEP)],
            linewidth=4, color='k')
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.ylabel('current(pA)', fontsize = 25)
plt.xlabel('time (ms)', fontsize =25)
plt.tight_layout()
plt.show()
#plt.savefig('STA.png')


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(np.arange(1,(STA_TIME/INTSTEP)+1), np.real(Eigen), linewidth=5)
plt.xlim([-50,2050])
plt.xticks(np.arange(-0,2010,500),('0','','','','2000'),fontsize = 25)
plt.yticks(fontsize = 25)
plt.xlabel('Eigenvalue number', fontsize = 25)
plt.tight_layout()
plt.show()



fig = plt.figure()
ax2 = fig.add_subplot(111)
ax2.plot(X[0:(STA_TIME/INTSTEP)],-Modes[:,0], X[0:(STA_TIME/INTSTEP)],-Modes[:,1], linewidth=3)
prop = fm.FontProperties(size=20)
plt.legend(('Mode1','Mode2'),loc = 'upper left', prop=prop)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.xlabel('time (ms)', fontsize = 25)
plt.tight_layout()
plt.show()
#plt.savefig('Eigen.png')
#plt.close()


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(X[0:(STA_TIME/INTSTEP)],-Modes[:,0], X[0:(STA_TIME/INTSTEP)],-Modes[:,1], linewidth=3)
prop = fm.FontProperties(size=20)
plt.legend(('Mode1','Mode2'),loc = 'upper left', prop=prop)
ax.plot(np.arange(-190,-170),np.ones(20)*0.03, linewidth=5,color='k')
ax.plot(np.ones(50)*-170,np.arange(0.03,0.08,0.001),linewidth=5,color='k')
plt.axis('off')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(X[0:(STA_TIME/INTSTEP)],STA[0:(STA_TIME/INTSTEP)], linewidth=3, color='k')
ax.plot(np.arange(-190,-170),np.ones(20)*0.35, linewidth=5,color='k')
ax.plot(np.ones(200)*-170,np.arange(0.35,0.549,0.001),linewidth=5,color='k')
ax.plot(np.arange(-200,0),np.zeros(200), 'k--', linewidth=2)
plt.axis('off')
plt.show()

fig = plt.figure()
ax2 = fig.add_subplot(211)
ax2.plot(np.arange(0,len(voltage)),voltage*1000, linewidth=2, color='k')
ax2.plot(np.ones(20)*30,np.arange(-40,-20), linewidth=5,color='k')
#ax2.plot(np.arange(0,len(voltage)),np.zeros(len(voltage)),'k--',linewidth=2)
plt.axis('off')
ax = fig.add_subplot(212)
ax.plot(np.arange(0,len(current)),current, linewidth=3, color='k')
ax.plot(np.arange(10,1010),np.ones(1000)*-7.5, linewidth=5,color='k')
ax.plot(np.ones(25)*1010,np.arange(-7.5,-5.,0.1),linewidth=5, color='k')
plt.ylim([-10,10])
plt.axis('off')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(X[0:(STA_TIME/INTSTEP)],oldSTA[0:(STA_TIME/INTSTEP)], 'k',X[0:(STA_TIME/INTSTEP)],ModelSTA[0:(STA_TIME/INTSTEP)], linewidth=3)
ax.plot(np.arange(-190,-170),np.ones(20)*0.35, linewidth=5,color='k')
ax.plot(np.ones(200)*-170,np.arange(0.35,0.549,0.001),linewidth=5,color='k')
ax.plot(np.arange(-200,0),np.zeros(200), 'k--', linewidth=2)
prop = fm.FontProperties(size=20)
plt.legend(('Experiment','Model'),loc = 'upper left', prop=prop)
plt.axis('off')
plt.show()


plt.figure()
plt.plot(X[0:(STA_TIME/INTSTEP)],Modes[:,0], 'k',linewidth=3)
plt.plot(np.arange(1701,1900),np.ones(200)*-0.3)
plt.axis('off')
plt.show()

plt.plot(X[0:(STA_TIME/INTSTEP)], -Modes[:,1], 'k',linewidth=3)
plt.axis('off')
plt.show()
