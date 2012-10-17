import numpy as np
import numpy.linalg as lin
import matplotlib.pylab as plt

INTSTEP = 0.05
STA_TIME = 180 # in msec

Prior = np.genfromtxt('PriorRecord.csv',delimiter=',')
Prior = Prior[:,0:STA_TIME/INTSTEP]

STA = np.genfromtxt('STA_May8.csv',delimiter=',')
STA = STA[0:STA_TIME/INTSTEP]
STA = STA/lin.norm(STA)

EigVect = np.genfromtxt('EigVect_May8.csv',delimiter=',')
Mode1 = EigVect[:,0]
Mode2 = EigVect[:,1]
del(EigVect)
Current = np.genfromtxt('DataRecord_May8.csv',delimiter=',')
Current = Current[:,0:STA_TIME/INTSTEP]

Priorproj = np.zeros(len(Current[:,0]))
STAproj = np.zeros(len(Current[:,0]))
Mode1proj = np.zeros(len(Current[:,0]))
Mode2proj = np.zeros(len(Current[:,0]))
Mode1Pr = np.zeros(len(Current[:,0]))
Mode2Pr = np.zeros(len(Current[:,0]))

for i in range(len(Current[:,0])):
    B = Current[i,:]
    Pr = Prior[i,:]
    #B_norm = lin.norm(B)
    Priorproj[i] = np.dot(STA,Pr)
    STAproj[i] = np.dot(STA,B)#/B_norm
    Mode1Pr[i] = np.dot(Mode1,Pr)
    Mode2Pr[i] = np.dot(Mode2,Pr)
    Mode1proj[i] = np.dot(Mode1,B)#/B_norm
    Mode2proj[i] = np.dot(Mode2,B)#/B_norm

np.savetxt('Mode1NoisePr_May8.csv',Mode1Pr, delimiter = ',')
np.savetxt('Mode2NoisePr_May8.csv',Mode2Pr, delimiter = ',')
np.savetxt('PriorProj_May8.csv', Priorproj, delimiter=',')
np.savetxt('STAproject_May8.csv', STAproj, delimiter=',')
np.savetxt('Mode1project_May8.csv', Mode1proj, delimiter=',')
np.savetxt('Mode2project_May8.csv', Mode2proj, delimiter=',')

