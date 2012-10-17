import numpy as np
import matplotlib.pylab as plt



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
	
	
def ComputeProbSpikeGiven(Priorproject, STAproject, Mode1project, Mode1Noise, Mode2project,
							Mode2Noise):
	BINS = np.arange(-20,18,0.5)

	Prior_Hist = np.zeros(len(BINS)-1)
	STA_Hist = np.zeros(len(BINS)-1)
	M12_Spike = np.zeros((len(BINS)-1,len(BINS)-1))
	M12_Prior = np.zeros((len(BINS)-1,len(BINS)-1))
	for i in range(0,len(BINS)-1):
		Start = BINS[i]
		End = BINS[i+1]
			
		Prior_Hist[i] = sum(np.logical_and(Priorproject>Start, Priorproject<=End))+1
		STA_Hist[i] = sum(np.logical_and(STAproject>Start, STAproject<=End))+1
		Mode1_Hist = np.logical_and(Mode1project>Start, Mode1project<=End)
		Mode1Prior_Hist = np.logical_and(Mode1Noise>Start, Mode1Noise<=End)
		for j in range(0,len(BINS)-1):
			S1 = BINS[j]
			E1 = BINS[j+1]
			
			Mode2_Hist = np.logical_and(Mode2project>S1, Mode2project<=E1)
			Mode2Prior_Hist = np.logical_and(Mode2Noise>S1, Mode2Noise<=E1)
			Total = sum(np.logical_and(Mode1_Hist==1,Mode2_Hist==1))
			Total_Hist = sum(np.logical_and(Mode1Prior_Hist==1,Mode2Prior_Hist==1))

			M12_Spike[i,j] = Total+1
			M12_Prior[i,j] = Total_Hist+1


	Pspike_STA = STA_Hist*Pspike/Prior_Hist
	Pspike_2d = M12_Spike*Pspike/M12_Prior
	return Pspike_STA, Pspike_2d


### GEN FILES ###

Priorproject = np.genfromtxt('PriorProj_May7.csv',delimiter=',')
STAproject = np.genfromtxt('STAproject_May7.csv',delimiter=',')
Mode1project = np.genfromtxt('Mode1project_May7.csv',delimiter=',')
Mode2project = np.genfromtxt('Mode2project_May7.csv',delimiter=',')
Mode1Noise = np.genfromtxt('Mode1NoisePr_May7.csv',delimiter=',')
Mode2Noise = np.genfromtxt('Mode2NoisePr_May7.csv',delimiter=',')

Pspike = 7477/((1055000*0.1)*20/1000) ## in Hz. Spikes/len(one row)*dt*20(rows)/1000(msec/sec)


X,Y = np.meshgrid(BINS[0:len(BINS)-1],BINS[0:len(BINS)-1])#(x[0:len(x)-1],y[0:len(y)-1])#
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Pspike_2d, rstride=8, cstride=8, alpha=0.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.grid(True)
ax.set_xlabel('s2', fontsize=20)
ax.set_ylabel('s1', fontsize=20)
ax.set_zlabel('Hz', fontsize=20)
plt.show()



plt.figure()
plt.plot(x,Pspike_M1(x), linewidth=4,color='b')
plt.plot(x,Pspike_M2(x), linewidth=4,color='g')
plt.xlabel('projection',fontsize=20)
plt.ylabel('firing rate (Hz)',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(('P(Spike|M1)','P(Spike|M2)'),loc='upper right')
plt.tight_layout()
plt.show()

## LOAD EigenModes ##
Current = np.loadtxt('input.dat', unpack=True,usecols=[0])
Voltage = np.loadtxt('voltage.dat',unpack=True,usecols=[0])
STA = np.genfromtxt('STA_May7.csv',delimiter=',')
STA = STA[0:STA_TIME/INTSTEP]
STA = STA/lin.norm(STA)
EigVect = np.genfromtxt('EigVect_May7.csv',delimiter=',')
Mode1 = EigVect[:,0]
Mode2 = EigVect[:,1]
del(EigVect)

### FIND P(spike|s0,s1,s2) ####
STA_LEN = STA_TIME/INTSTEP
Ps_STA = np.zeros(END-START+1)
Ps_2d = np.zeros(END-START+1)
S1 = np.zeros(END-START+1)
S2 = np.zeros(END-START+1)

for i in range(START,END):
    
    S0 = round((float(np.dot(STA,Current[i-STA_LEN:i]))/0.25)*0.25)
    loc = np.where(BINS==[S0])
    Ps_STA[i-START] = Pspike_STA[loc]
    
    S1[i-START] =round((float(np.dot(Mode1,Current[i-STA_LEN:i]))/0.25)*0.25)
    S2[i-START] =round((float(np.dot(Mode2,Current[i-STA_LEN:i]))/0.25)*0.25)
    loc1 = np.where(BINS==[S1[i-START]])
    loc2 = np.where(BINS==[S2[i-START]])
    Ps_2d[i-START] = Pspike_2d[loc1,loc2]




### FIND SPIKES IN DATA ###
Spikes = Voltage[START:END] > 0
Spikes = runs(Spikes)


## Call Plot functions here ##

