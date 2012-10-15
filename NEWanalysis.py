import numpy as np
import numpy.linalg as lin
import scipy as sp
import matplotlib.pylab as plt


current = np.loadtxt('input.dat',unpack=True)
voltage = np.loadtxt('voltage.dat',unpack=True)

INTSTEP = 0.1 ## in msec
STA_TIME = 200
TIME = len(current)*INTSTEP ## in msec

PLOT1 = 0
PLOT3 = 0

STAFILENAME = 'STA_May7.csv'
RECORDFILENAME = 'DataRecord_May7.csv'
COVMATRIX = 'CovMat_May7.csv'
EIGVAL = 'EigVal_May7.csv'
EIGVECT = 'EigVect_May7.csv'

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

Data_mV_Record = np.zeros((500*len(current),int((STA_TIME/INTSTEP*2))))
Data_C_Record= np.zeros((500*len(current),int((STA_TIME/INTSTEP*2))))
Prior_Rec = np.zeros((500*len(current),int((STA_TIME/INTSTEP*2))))
    
q = 0
LOC_IND = 0
ON_LOC = np.array([0,409900,815000])
OFF_LOC = np.array([355000,759900,1165000])


for i in range(0,len(current)):
    a = np.array([current[i][ON_LOC[0]:OFF_LOC[0]],current[i][ON_LOC[1]:OFF_LOC[1]],current[i][ON_LOC[2]:OFF_LOC[2]]])
    CURRENT = np.append(a[0],a[1])
    CURRENT = np.append(CURRENT,a[2])

    a = np.array([voltage[i][ON_LOC[0]:OFF_LOC[0]],voltage[i][ON_LOC[1]:OFF_LOC[1]],voltage[i][ON_LOC[2]:OFF_LOC[2]]])
    voltage_Data = np.append(a[0],a[1])
    voltage_Data = np.append(CURRENT,a[2])
    del(a)
    
    Spikes_Data = voltage_Data > 0
    Spikes_Data = runs(Spikes_Data)

    ## Find Spikes that do not fall too close to the beginning or end
    S_beg_Data = np.where(Spikes_Data[0] > int(STA_TIME/INTSTEP*2))
    S_end_Data = np.where(Spikes_Data[0] < (len(voltage_Data)-(int(STA_TIME/INTSTEP*2)+20)))
    Length_Data = np.arange(min(S_beg_Data[0]),max(S_end_Data[0])+1)
    Spikes_Data = Spikes_Data[0][Length_Data],Spikes_Data[1][Length_Data]

    for j in range(q,q+len(Spikes_Data[0])):
               
        Peak = np.arange(Spikes_Data[0][j-q],Spikes_Data[1][j-q])
        Peak = voltage_Data[Peak]
        Height = np.argmax(Peak)
        Loc = Height + Spikes_Data[0][j-q]
        RandLoc = np.random.random_integers(STA_TIME/INTSTEP,(len(CURRENT)-(STA_TIME/INTSTEP)))
        Range = np.arange(Loc-(STA_TIME/INTSTEP),Loc+(STA_TIME/INTSTEP), dtype=int)
        RandRANGE = np.arange(RandLoc-(STA_TIME/INTSTEP),RandLoc+(STA_TIME/INTSTEP), dtype=int)
        
        Data_mV_Record[j,:] = voltage_Data[Range]
        Data_C_Record[j,:] = CURRENT[Range]
        Prior_Rec[j,:] = CURRENT[RandRANGE]
    q += len(Spikes_Data[0])
    
Data_C_Record = Data_C_Record[np.any(Data_C_Record,1),:]
Data_mV_Record = Data_mV_Record[np.any(Data_mV_Record,1),:]
Prior_Rec = Prior_Rec[np.any(Prior_Rec,1),:]

Data_Num_Spikes = len(Data_mV_Record)
print '# Data Spikes:', Data_Num_Spikes

Data_STA_Current =Data_C_Record.mean(0)
Data_STA_Voltage = Data_mV_Record.mean(0)

np.savetxt(STAFILENAME,Data_STA_Current,delimiter=',')
np.savetxt(RECORDFILENAME,Data_C_Record,delimiter=',')
np.savetxt('PriorRecord.csv',Prior_Rec,delimiter=',')

fig = plt.figure(figsize=(12,8))
X = np.arange(-STA_TIME/INTSTEP,STA_TIME/INTSTEP,dtype=float)*INTSTEP
ax = fig.add_subplot(111)
ax.plot(X[0:(STA_TIME/INTSTEP)+50], Data_STA_Current[0:(STA_TIME/INTSTEP)+50],
            linewidth=3, color='k')
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.ylabel('current(pA)', fontsize = 20)
#plt.legend(('data'), loc='upper right')

plt.show()
plt.savefig('STA.png')
plt.close()


################
###COVARIANCE### 
################

Begin = 0
End = STA_TIME/INTSTEP

Data_C_spike = np.zeros((End,End))
C_prior = np.zeros((End,End))


STA = Data_STA_Current[Begin:End]
mat_STA = STA*STA[:,np.newaxis]*Data_Num_Spikes/(Data_Num_Spikes-1)

for i in range(0,Data_Num_Spikes):
    a = Data_C_Record[i,Begin:End]
    b = Prior_Rec[i,Begin:End]
    
    mat_spike = a*a[:,np.newaxis]
    mat_prior = b*b[:,np.newaxis]
    
    
    Data_C_spike += (mat_spike - mat_STA)
    C_prior += mat_prior
    
### FIND MEANS ###
Data_C_spike = Data_C_spike/(Data_Num_Spikes-1)
C_prior = C_prior/(Data_Num_Spikes-1)

### FIND DELTA COV ###
Data_C_delta = Data_C_spike - C_prior

### EigenValues, EigenVectors
Data_E, V = lin.eig(Data_C_delta)
I = np.argsort(Data_E)
Data_E = Data_E[I][:(STA_TIME/INTSTEP)]
Data_Vect = V[:,I][:,:(STA_TIME/INTSTEP)]

np.savetxt(COVMATRIX,Data_C_delta, delimiter=',')
np.savetxt(EIGVAL,Data_E, delimiter=',')
np.savetxt(EIGVECT,Data_Vect,delimiter=',')

#fig = plt.figure(figsize=(12,9))
#ax = fig.add_subplot(111)
#ax.imshow(Data_C_delta, origin = 'lower')
#plt.title('Delta_C')

#plt.show()
#plt.savefig('DeltaC.png')
#plt.close()

##fig = plt.figure()
#ax = fig.add_subplot(211)
#ax.scatter(np.arange(1,len(Data_E)+1), np.real(Data_E))

#ax2 = fig.add_subplot(212)
#ax2.plot(np.arange(0,(STA_TIME/INTSTEP)),Data_Vect[0],
#           np.arange(0,(STA_TIME/INTSTEP)),Data_Vect[1])
#plt.show()
#plt.savefig('Eigen.png')
#plt.close()
