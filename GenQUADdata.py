import numpy as np

CURRENT = np.loadtxt('input.dat',unpack=True)*100
param = np.genfromtxt('EvolvedParam1_8.csv',delimiter=',')

INTSTEP = 0.1 
### HH Model ###
def QUADmodel(param, CURRENT):
    
    u1DOT = 0
    u2DOT = 0
    vDOT = 0
    CAP = 50.0 ## in pF
    vDOT = -70 ## V naught in mV
    vRESTING = -70
    VOLTAGE = np.zeros(len(CURRENT))
    b_1 = param[7] # if vDOT < param[5] else 0
    b_2 = -param[5]    
    a_1 = param[0]
    a_2 = param[1]
    d_1 = param[2]
    d_2 = param[3]
    vRESET = param[8]
    vTHR = param[4]

    for i in range(0,len(CURRENT)):
    
        k_ = param[6] if vDOT < param[4] else 2
        
        vDOT += (k_*(vDOT - vRESTING)*(vDOT - vTHR) - u1DOT - u2DOT +
                 CURRENT[i])*INTSTEP/CAP
        u1DOT += (a_1*(b_1*(vDOT - vRESTING) - u1DOT))*INTSTEP
        u2DOT += (a_2*(b_2*(vDOT - vRESTING) - u2DOT))*INTSTEP
    
        if vDOT >= 35 + (0.1*(u1DOT+u2DOT)):
            vDOT = vRESET - (0.1*(u1DOT+u2DOT))
            u1DOT = u1DOT + d_1
            u2DOT = u2DOT + d_2

        VOLTAGE[i] = vDOT
    return VOLTAGE

QuadModelVoltage = np.zeros((len(CURRENT),len(CURRENT[0,:])))

for i in range(0,len(CURRENT)):
    QuadModelVoltage[i,:] = QUADmodel(param,CURRENT[i])

np.savetxt('ModelVolt.csv',QuadModelVoltage,delimiter=',')
