import numpy as np
import matplotlib.pylab as plt


## Models ##
def EXPmodel(param):
    
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




## Analysis Functions ##


def coinc_detection(Data_Spikes, Model_Spikes, DELTA):
    
    spike_reward = 0
    max_spike = len(Model_Spikes)
    max_spike_data = len(Data_Spikes)
    if max_spike >= 1:

        interval = DELTA/INTSTEP
        counter = 0
        spike_test = 0
    
        while spike_test < max_spike and counter < max_spike_data:
            if ( Model_Spikes[spike_test] >= (Data_Spikes[counter] - interval) and
                 Model_Spikes[spike_test] <= (Data_Spikes[counter] + interval) ):
                spike_reward += 1
                counter += 1
                spike_test += 1
            elif Model_Spikes[spike_test] < Data_Spikes[counter] - interval:
                spike_test += 1
            elif Model_Spikes[spike_test] > Data_Spikes[counter] + interval:
                counter += 1
                
    return spike_reward

def runs(bits):
    # make sure all runs of ones are well-bounded
    bounded = np.hstack(([0], bits, [0]))
    # get 1 at run starts and -1 at run ends
    difs = np.diff(bounded)
    run_starts, = np.where(difs > 0)
    run_ends, = np.where(difs < 0)
    # eliminate noise or short duration chirps
    length = run_ends-run_starts
    return run_starts, run_ends


def Pspike_STA(x):
    out = STA_PDF(x)*Pspike/Prior_PDF(x)
    return out

def Pspike_M1(x):
    out = Mode1_PDF(x)*Pspike/Mode1Noise_PDF(x)
    return out

def Pspike_M2(x):
    out = Mode2_PDF(x)*Pspike/Mode2Noise_PDF(x)
    return out

#def Pspike_2d(x,y):
#    out = ( Mode1_PDF(x) + Mode2_PDF(y) )*Pspike/( Mode1Noise_PDF(x) + (Mode2Noise_PDF(y)) )
#    return out

def histOutline(histIn,binsIn):

    stepSize = binsIn[1] - binsIn[0]
 
    bins = np.zeros(len(binsIn)*2 + 2, dtype=np.float)
    data = np.zeros(len(binsIn)*2 + 2, dtype=np.float)
    for bb in range(len(binsIn)):
        bins[2*bb + 1] = binsIn[bb]
        bins[2*bb + 2] = binsIn[bb] + stepSize
        if bb < len(histIn):
            data[2*bb + 1] = histIn[bb]
            data[2*bb + 2] = histIn[bb]
 
    bins[0] = bins[1]
    bins[-1] = bins[-2]
    data[0] = 0
    data[-1] = 0 
    return (bins, data)



def Hist(BIN_SIZE):
    BINS = np.linspace(0,len(Spikes),len(Spikes)/BIN_SIZE*INTSTEP)
    Data_Hist = np.zeros(len(BINS))
    STA_Hist = np.zeros(len(BINS))
    Model_Hist = np.zeros(len(BINS))
    for i in range(0,len(BINS)-1):
        Start = BINS[i]
        End = BINS[i+1]
        Data_Total = sum(Spikes[Start:End])
        STA_Total = np.mean(Ps_STA[Start:End])
        Model_Total = np.mean(Ps_2d[Start:End])
        
        Data_Hist[i] = Data_Total
        STA_Hist[i] = STA_Total
        Model_Hist[i] = Model_Total
    Data_Hist = Data_Hist/BIN_SIZE*1000.0
    return Data_Hist, STA_Hist, Model_Hist,BINS
