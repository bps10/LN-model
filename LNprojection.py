import numpy as np
import numpy.linalg as lin
import matplotlib.pylab as plt

INTSTEP = 0.1
STA_TIME = 200 # in msec

Prior = np.genfromtxt('PriorRecord.csv',delimiter=',')
Prior = STA[:,0:STA_TIME/INTSTEP]

STA = np.genfromtxt('STA.csv',delimiter=',')
STA = STA[0:STA_TIME/INTSTEP]
STA = STA/lin.norm(STA)

EigVect = np.genfromtxt('EigVect.csv',delimiter=',')
Mode1 = EigVect[:,0]
Mode2 = EigVect[:,1]
del(EigVect)
Current = np.genfromtxt('DataRecord.csv',delimiter=',')
Current = Current[:,0:STA_TIME/INTSTEP]

Priorproj = np.zeros(len(Current[:,0]))
STAproj = np.zeros(len(Current[:,0]))
Mode1proj = np.zeros(len(Current[:,0]))
Mode2proj = np.zeros(len(Current[:,0]))

for i in range(len(Current[:,0])):
    B = Current[i,:]
    #B_norm = lin.norm(B)
    Priorproj[i] = np.dot(STA,Prior)
    STAproj[i] = np.dot(STA,B)#/B_norm
    Mode1proj[i] = np.dot(Mode1,B)#/B_norm
    Mode2proj[i] = np.dot(Mode2,B)#/B_norm

np.savetxt('PriorProj.csv', Priorproj, delimiter=',')
np.savetxt('STAproject.csv', STAproj, delimiter=',')
np.savetxt('Mode1project.csv', Mode1proj, delimiter=',')
np.savetxt('Mode2project.csv', Mode2proj, delimiter=',')

