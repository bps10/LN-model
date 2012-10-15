import numpy as np
import scipy as sp
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec

CURRENT = np.loadtxt('input.dat',unpack=True, usecols=[1])*100
voltage_Data = np.loadtxt('voltage.dat',unpack=True, usecols=[1])*1000
param = np.genfromtxt('EvolEXP_SSH1.csv',delimiter=',')

INTSTEP = 0.1 ## in msec
STA_TIME = 200
TIME = len(CURRENT)*INTSTEP ## in msec

PLOT1 = 1
PLOT3 = 1
SAVERECORD = 0

PLOT1NAME = 'Trace_Par7_May15.png'
PLOT2NAME = 'STA_Par7_May15.png'

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


voltage_Model = EXPmodel(param)


if PLOT1 is 1:
    t_beg = 5e3
    t_end = 2e5
    x_ = np.arange(0,t_end-t_beg)*INTSTEP/1000.

    s_beg = 9e3
    s_end = 12e3
    xx = np.arange(0,s_end-s_beg)*INTSTEP
    
    fig = plt.figure(figsize=(11,8.75))
    gs1 = gridspec.GridSpec(11, 1)
    gs1.update(left=0.1, right=0.95, top=0.98, bottom=0.075, hspace=0.35)
    ax1 = fig.add_subplot(gs1[0:2,:])
    ax1.plot(xx, voltage_Model[s_beg:s_end], 'b', xx,
             voltage_Data[s_beg:s_end],'k',linewidth=2)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.legend(('model','data'),loc='upper left')
    plt.yticks(fontsize = 16)
    plt.ylabel('mV', fontsize = 16)
    
    ax2 = fig.add_subplot(gs1[2:4,:], sharex = ax1)
    ax2.plot(xx, CURRENT[s_beg:s_end]/1000., 'k', linewidth=2)
    plt.yticks(fontsize = 16)
    plt.ylabel('nA', fontsize = 16)
    plt.xlabel('time (msec)', fontsize = 16)
    plt.xticks(fontsize=16)

    ax3 = fig.add_subplot(gs1[5:7,:])
    ax3.plot(x_, voltage_Data[t_beg:t_end], 'k',linewidth=2)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.yticks(fontsize = 16)
    plt.xticks(fontsize = 16)
    plt.ylabel('mV', fontsize = 16)
    
    ax4 = fig.add_subplot(gs1[7:9,:], sharex=ax3)
    ax4.plot(x_, voltage_Model[t_beg:t_end], 'b', linewidth=2)
    plt.yticks(fontsize = 16)
    plt.setp(ax4.get_xticklabels(), visible=False)
    plt.ylabel('mV', fontsize = 16)
    
    ax5 = fig.add_subplot(gs1[9:11,:], sharex=ax3)
    ax5.plot(x_, CURRENT[t_beg:t_end]/1000., 'k', linewidth=2)
    plt.yticks(fontsize = 16)
    plt.xticks(fontsize = 16)
    plt.ylabel('nA', fontsize = 16)
    plt.xlabel('time (sec)', fontsize = 16)

    # plt.savefig(PLOT1NAME)
    plt.show()
    
    
######################
#### STA ANALYSIS ####
######################
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

if PLOT3 is 1:
    
    
    Spikes_Data = voltage_Data > 0
    Spikes_Model = voltage_Model > 0
    Spikes_Data = runs(Spikes_Data)
    Spikes_Model = runs(Spikes_Model)

    #### Coincidence Detection
    DELTA = 4.0 ## msec
    Hits = float(coinc_detection(Spikes_Data[0],Spikes_Model[0], DELTA))
    N_data = len(Spikes_Data[0])
    N_model = len(Spikes_Model[0])
    Extra = (N_model - Hits)/N_model
    Missing = (N_data - Hits)/N_data
    Npoisson = (2*DELTA*N_model)*N_data/TIME


    print 'Hit (%): ', Hits/N_data*100
    print 'Missing Spikes (%): ', Missing*100
    print 'Extra Spikes (%): ', Extra*100
    print 'Simple Coincidence Factor: ', 1-((Extra + Missing)/2.)
    print 'Better Coincidence Factor: ', (Hits - Npoisson) / (0.5*(1-(Npoisson/N_data))*(N_data + N_model))

    ## Find Spikes that do not fall too close to the beginning or end
    S_beg_Data = np.where(Spikes_Data[0] > int(STA_TIME/INTSTEP*2))
    S_end_Data = np.where(Spikes_Data[0] < (len(voltage_Data)-(int(STA_TIME/INTSTEP*2)+20)))
    Length_Data = np.arange(min(S_beg_Data[0]),max(S_end_Data[0])+1)
    Spikes_Data = Spikes_Data[0][Length_Data],Spikes_Data[1][Length_Data]

    S_beg_Model = np.where(Spikes_Model[0] > int(STA_TIME/INTSTEP*2))
    S_end_Model = np.where(Spikes_Model[0] < (len(voltage_Model)-(int(STA_TIME/INTSTEP*2)+20)))
    Length_Model = np.arange(min(S_beg_Model[0]),max(S_end_Model[0])+1)
    Spikes_Model = Spikes_Model[0][Length_Model],Spikes_Model[1][Length_Model]



    ## Create record files
    Data_mV_Record = np.zeros((len(Spikes_Data[0]),int((STA_TIME/INTSTEP*2))))
    Data_C_Record= np.zeros((len(Spikes_Data[0]),int((STA_TIME/INTSTEP*2))))
    Model_mV_Record = np.zeros((len(Spikes_Model[0]),int((STA_TIME/INTSTEP*2))))
    Model_C_Record = np.zeros((len(Spikes_Model[0]),int((STA_TIME/INTSTEP*2))))
   
                   
    for i in range(0,len(Spikes_Data[0])):
               
        Peak = np.arange(Spikes_Data[0][i],Spikes_Data[1][i])
        Peak = voltage_Data[Peak]
        Height = np.argmax(Peak)
        Loc = Height + Spikes_Data[0][i]
        Range = np.arange(Loc-(STA_TIME/INTSTEP),Loc+(STA_TIME/INTSTEP), dtype=int)

        Data_mV_Record[i,:] = voltage_Data[Range]
        Data_C_Record[i,:] = CURRENT[Range]

    Data_Num_Spikes = len(Data_mV_Record[:,0])
    print '# Data Spikes:', Data_Num_Spikes

    for i in range(0,len(Spikes_Model[0])):
               
        Peak = np.arange(Spikes_Model[0][i],Spikes_Model[1][i])
        Peak = voltage_Model[Peak]
        Height = np.argmax(Peak)
        Loc = Height + Spikes_Model[0][i]
        Range = np.arange(Loc-(STA_TIME/INTSTEP),Loc+(STA_TIME/INTSTEP), dtype=int)

        Model_mV_Record[i,:] = voltage_Model[Range]
        Model_C_Record[i,:] = CURRENT[Range]

    Model_Num_Spikes = len(Model_mV_Record[:,0])
    print '# Model Spikes:', Model_Num_Spikes

    if SAVERECORD is 1:
        np.savetxt('Data_mV_Record', Data_mV_Record, delimiter=',')
        np.savetxt('Data_C_Record', Data_C_Record, delimiter = ',')
        np.savetxt('Model_mV_Record', Model_mV_Record, delimiter = ',')
        np.savetxt('Model_C_record', Model_C_Record, delimiter = ',')


    Data_STA_Current = Data_C_Record.mean(0)
    Data_StD_Current = Data_C_Record.std(0)

    Data_STA_Voltage = Data_mV_Record.mean(0)
    Data_StD_Voltage = Data_mV_Record.std(0)

    Model_STA_Current = Model_C_Record.mean(0)
    Model_StD_Current = Model_C_Record.std(0)

    Model_STA_Voltage = Model_mV_Record.mean(0)
    Model_StD_Voltage = Model_mV_Record.std(0)

    X = np.arange(-STA_TIME/INTSTEP,STA_TIME/INTSTEP,dtype=float)*INTSTEP


    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(211)
    ax.plot(X[0:(STA_TIME/INTSTEP)+50], Data_STA_Current[0:(STA_TIME/INTSTEP)+50],
            linewidth=3, color='k')
    # plt.setp(ax.get_xticklabels(), visible=False)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.ylabel('current(pA)', fontsize = 20)
    plt.legend(('data'), loc='upper right')
    #plt.grid(True,axis='x', linewidth=1, color='k', linestyle='-')
    #    plt.fill_between(X,STA_Current+StD_Current,STA_Current-StD_Current,
    #                facecolor='gray', alpha=1, edgecolor='gray')

    ax2 = fig.add_subplot(212)
    ax2.plot(X[0:(STA_TIME/INTSTEP)+50], Model_STA_Current[0:(STA_TIME/INTSTEP)+50],
             'b', linewidth=3)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.ylabel('current(pA)', fontsize = 20)
    plt.legend('model', loc='upper right')
    #plt.grid(True,axis='x', linewidth=1, color='k', linestyle='-')

    # plt.tight_layout()
    # plt.savefig(PLOT2NAME)
    plt.show()


    


