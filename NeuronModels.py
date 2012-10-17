import numpy as np

## Models ##
def EXPmodel(param, CURRENT):
    
    VOLTAGE = np.zeros(len(CURRENT))
    vDOT = -70 ## V naught in mV
    u1DOT = 0 ## u1 naught in a.u.
    El = -70 ## mV
    gL = 20 ##nS
    
    a_1 = param[0]
    d_1 = param[1]
    vRESET = param[2]
    vTHR = param[3]
    DELTA_T = param[4]
    CAP = param[5]
    TauU = param[6]
    
    for i in range(0,len(CURRENT)):

        vDOT += (-gL*(vDOT - El) + gL*DELTA_T*np.exp((vDOT - vTHR)/DELTA_T) -
                 (u1DOT) + CURRENT[i])/CAP*INTSTEP

        if vDOT >= 35:
            VOLTAGE[i] = 35 ## vPEAK
            tPEAK = INTSTEP*((VOLTAGE[i] - VOLTAGE[i-1])/(vDOT - VOLTAGE[i-1]))
            u1RES =  (a_1*(VOLTAGE[i] - El) - u1DOT)/TauU*tPEAK
            u1DOT = u1RES + d_1
            vDOT = vRESET
            if i < len(CURRENT)-1:
                i += 1
        
        if vDOT < -150:
            VOLTAGE = np.array([np.nan])
            return VOLTAGE

        u1DOT += (a_1*(vDOT - El) - u1DOT)/TauU*INTSTEP
        
        VOLTAGE[i] = vDOT
    return VOLTAGE



def QUADmodel(param, CURRENT, INTSTEP):
    
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