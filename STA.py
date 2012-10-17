import numpy as np
import numpy.linalg as lin
import scipy as sp
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec


current = np.loadtxt('input.dat',unpack=True)
voltage = np.loadtxt('voltage.dat',unpack=True)
    
#CURRENT = CURRENT.flatten()
#voltage_Data = voltage_Data.flatten()

INTSTEP = 0.1 ## in msec
STA_TIME = 200
TIME = len(current)*INTSTEP ## in msec

PLOT1 = 0
PLOT3 = 1
SAVERECORD = 0

PLOT1NAME = 'Trace_Par7_May15.png'
PLOT2NAME = 'STA_Par7_May15.png'

    
######################
#### STA ANALYSIS ####
######################

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

## Create Files ##
STA_CURRENT = np.zeros((len(current),int((STA_TIME/INTSTEP*2))))
STA_VOLTAGE = np.zeros((len(current),int((STA_TIME/INTSTEP*2))))

for i in range(0,len(current)):
    CURRENT = current[i]
    voltage_Data = voltage[i]
    
    Spikes_Data = voltage_Data > 0
    Spikes_Data = runs(Spikes_Data)

    ## Find Spikes that do not fall too close to the beginning or end
    S_beg_Data = np.where(Spikes_Data[0] > int(STA_TIME/INTSTEP*2))
    S_end_Data = np.where(Spikes_Data[0] < (len(voltage_Data)-(int(STA_TIME/INTSTEP*2)+20)))
    Length_Data = np.arange(min(S_beg_Data[0]),max(S_end_Data[0])+1)
    Spikes_Data = Spikes_Data[0][Length_Data],Spikes_Data[1][Length_Data]

    ## Create record files
    Data_mV_Record = np.zeros((len(Spikes_Data[0]),int((STA_TIME/INTSTEP*2))))
    Data_C_Record= np.zeros((len(Spikes_Data[0]),int((STA_TIME/INTSTEP*2))))
    Prior_Rec = np.zeros((len(Spikes_Data[0]),int((STA_TIME/INTSTEP*2))))

    for j in range(0,len(Spikes_Data[0])):
               
        Peak = np.arange(Spikes_Data[0][j],Spikes_Data[1][j])
        Peak = voltage_Data[Peak]
        Height = np.argmax(Peak)
        Loc = Height + Spikes_Data[0][j]
        RandLoc = np.random.random_integers(STA_TIME/INTSTEP,(len(CURRENT)/STA_TIME/INTSTEP))
        Range = np.arange(Loc-(STA_TIME/INTSTEP),Loc+(STA_TIME/INTSTEP), dtype=int)
        RandRANGE = np.arange(RandLoc-(STA_TIME/INTSTEP),RandLoc+(STA_TIME/INTSTEP), dtype=int)
        
        Data_mV_Record[j,:] = voltage_Data[Range]
        Data_C_Record[j,:] = CURRENT[Range]
        Prior_Rec[j,:] = CURRENT[RandRANGE]
        
    Data_Num_Spikes = len(Data_mV_Record[:,0])
    print '# Data Spikes:', Data_Num_Spikes


    if SAVERECORD is 1:
        np.savetxt('Data_mV_Record', Data_mV_Record, delimiter=',')
        np.savetxt('Data_C_Record', Data_C_Record, delimiter = ',')


    Data_STA_Current = Data_C_Record.mean(0)
    Data_STA_Voltage = Data_mV_Record.mean(0)

    STA_CURRENT[i,:] = Data_STA_Current
    STA_VOLTAGE[i,:] = Data_STA_Voltage

Data_STA_Current = STA_CURRENT.mean(0)
Data_STA_Voltage = STA_VOLTAGE.mean(0)

fig = plt.figure(figsize=(12,8))
X = np.arange(-STA_TIME/INTSTEP,STA_TIME/INTSTEP,dtype=float)*INTSTEP
ax = fig.add_subplot(111)
ax.plot(X[0:(STA_TIME/INTSTEP)+50], Data_STA_Current[0:(STA_TIME/INTSTEP)+50],
            linewidth=3, color='k')
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.ylabel('current(pA)', fontsize = 20)
plt.legend(('data'), loc='upper right')
    #plt.grid(True,axis='x', linewidth=1, color='k', linestyle='-')
    #    plt.fill_between(X,STA_Current+StD_Current,STA_Current-StD_Current,
    #                facecolor='gray', alpha=1, edgecolor='gray')

    # plt.tight_layout()
    # plt.savefig(PLOT2NAME)
plt.show()

