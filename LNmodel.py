import numpy as np
import scipy.stats as stats
import numpy.linalg as lin
from mpl_toolkits.mplot3d import axes3d
from matplotlib.ticker import MaxNLocator
import matplotlib.pylab as plt

##### NOTES #####
## 1. FIND NEW SPIKE COUNT
## 2. FIND NEW LENGTH
## 3. Adjust current script


STA_TIME = 200
INTSTEP = 0.1

## PLOTTING OPTIONS ##
ONEd_THRESH_FUNC = 1
TWOd_THRESH_FUNC = 1
HIST_OUTLINE = 1
HIST_OUTLINE_NOLINES = 1
PSTH_PROJ = 1

LENGTH = 2 # in seconds of stimulus presentation
HIST_BIN_SIZE = 10 # in msec, for HIST_OUTLINE

### START/END of STIMULUS ###
START = 10001
END = int(START + (LENGTH*1000/INTSTEP) - 1)

#################
### FUNCTIONS ###
#################

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

def runs(bits):
    # make sure all runs of ones are well-bounded
    bounded = np.hstack(([0], bits, [0]))
    # get 1 at run starts and -1 at run ends
    difs = np.diff(bounded)
    difs[difs==-1] = 0
    return difs

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



### GEN FILES ###

Priorproject = np.genfromtxt('PriorProj_May7.csv',delimiter=',')
STAproject = np.genfromtxt('STAproject_May7.csv',delimiter=',')
Mode1project = np.genfromtxt('Mode1project_May7.csv',delimiter=',')
Mode2project = np.genfromtxt('Mode2project_May7.csv',delimiter=',')
Mode1Noise = np.genfromtxt('Mode1NoisePr_May7.csv',delimiter=',')
Mode2Noise = np.genfromtxt('Mode2NoisePr_May7.csv',delimiter=',')



### GEN PROB DENSITY FUNCTIONS ###

Prior_PDF = stats.kde.gaussian_kde(Priorproject)
STA_PDF = stats.kde.gaussian_kde(STAproject)
Mode1_PDF = stats.kde.gaussian_kde(Mode1project)
Mode2_PDF = stats.kde.gaussian_kde(Mode2project)
Mode1Noise_PDF = stats.kde.gaussian_kde(Mode1Noise)
Mode2Noise_PDF = stats.kde.gaussian_kde(Mode2Noise)

Pspike = 7477/((1055000*0.1)*20/1000) ## in Hz. Spikes/len(one row)*dt*20(rows)/1000(msec/sec)


if ONEd_THRESH_FUNC is 1:
    x = np.linspace(-20,18,1000)
    plt.figure()
    plt.plot(x,Pspike_STA(x), linewidth=4,color='k')
    plt.xlabel('projection',fontsize=20)
    plt.ylabel('firing rate (Hz)',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #plt.legend(('P(Spike|STA)','P(Spike|M1)','P(Spike|M2)',),loc='upper left')
    plt.tight_layout()
    plt.show()


    
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

if TWOd_THRESH_FUNC is 1:

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


if HIST_OUTLINE is 1:
    ## PLOT HISTOGRAM OUTLINES ##
    Data_Hist,STA_Hist,Model_Hist,BINS = Hist(HIST_BIN_SIZE)
    
    bins,Outline_Data_Hist = histOutline(Data_Hist,BINS)
    bins,Outline_STA_Hist = histOutline(STA_Hist,BINS)
    bins,Outline_Model_Hist = histOutline(Model_Hist,BINS)

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(411)
    ax.axes.get_xaxis().set_ticks([])
    plt.yticks(fontsize=20)
    plt.ylabel('Hz',fontsize=20)
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.plot(bins, Outline_STA_Hist, linewidth=2, color='k')
    plt.ylim([0,125])
    #plt.legend(('1D model'),loc='upper left')

    ax4 = fig.add_subplot(412)
    plt.yticks(fontsize=20)
    plt.ylabel('Hz',fontsize=20)
    ax4.axes.get_xaxis().set_ticks([])
    ax4.yaxis.set_major_locator(MaxNLocator(3))
    ax4.plot(bins, Outline_Model_Hist, linewidth=2, color='k')
    plt.ylim([0,125])
    #plt.legend(('2D model'),loc='upper left')

    ax5 = fig.add_subplot(413)
    plt.yticks(fontsize=20)
    plt.ylabel('Hz',fontsize=20)
    ax5.axes.get_xaxis().set_ticks([])
    ax5.yaxis.set_major_locator(MaxNLocator(3))
    ax5.plot(bins, Outline_Data_Hist, linewidth=2, color='k')
    plt.ylim([0,225])
    #plt.legend(('Data'),loc='upper left')

    ax6 = fig.add_subplot(414)
    plt.yticks(fontsize=20)
    plt.ylabel('nA',fontsize=20)
    ax6.yaxis.set_major_locator(MaxNLocator(3))
    ax6.plot(np.arange(1,len(Ps_2d))*INTSTEP, Current[START:END], linewidth=2, color='k')
    #ax6.plot(np.arange(100,(1000*INTSTEP)+100,np.ones(1000*INTSTEP)+4.5, linewidth=4, label='1 ms')
    plt.xlabel('time (msec)', fontsize =20)
    plt.xticks(fontsize=20)
    #ax6.axis('off')
    plt.show()

if HIST_OUTLINE_NOLINES is 1:

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(411)
    ax.axes.get_xaxis().set_ticks([])
    ax.plot(bins, Outline_STA_Hist, linewidth=2, color='k')
    plt.ylim([-10,125])
    ax.axis('off')

    ax4 = fig.add_subplot(412)
    ax4.axes.get_xaxis().set_ticks([])
    ax4.plot(np.ones(50)*(max(bins)-10),np.arange(30,80), linewidth=6 , color='k')
    ax4.plot(bins, Outline_Model_Hist, linewidth=2, color='k')
    plt.ylim([-10,125])
    ax4.axis('off')

    ax5 = fig.add_subplot(413)
    ax5.axes.get_xaxis().set_ticks([])
    ax5.plot(np.ones(50)*(max(bins)-10),np.arange(30,80), linewidth=6 , color='k')
    ax5.plot( np.arange(0, (100/INTSTEP)) ,np.ones(100/INTSTEP)*-24, linewidth=6, color='k')
    ax5.plot(bins, Outline_Data_Hist, linewidth=2, color='k')
    plt.ylim([-25,225])
    ax5.axis('off')

    ax6 = fig.add_subplot(414)
    ax6.plot(np.arange(1,len(Ps_2d))*INTSTEP, Current[START:END], linewidth=2, color='k')
    #ax6.plot(np.arange(100,(1000/INTSTEP)+100),np.ones(1000/INTSTEP)+4.5, linewidth=4, color='k')
             #plt.xlabel('time (msec)', fontsize =20)
    #plt.xticks(fontsize=20)
    ax6.axis('off')
    plt.tight_layout()
    plt.show()

    
if PSTH_PROJ is 1:
    TimeRes = np.array([0.1,0.25,0.5,1,2.5,5.0,10.0,25.0,50.0,100.0])

    Projection_PSTH = np.zeros((2,len(TimeRes)))
    for i in range(0,len(TimeRes)):
        Data_Hist,STA_Hist,Model_Hist,B = Hist(TimeRes[i])
        data = Data_Hist/np.linalg.norm(Data_Hist)
        sta = STA_Hist/np.linalg.norm(STA_Hist)
        model = Model_Hist/np.linalg.norm(Model_Hist)
        Projection_PSTH[0,i] = np.dot(data,sta)
        Projection_PSTH[1,i] = np.dot(data,model)
        
    import matplotlib.font_manager as fm
    
    plt.figure()
    plt.semilogx(TimeRes,Projection_PSTH[0,:],'gray',TimeRes,Projection_PSTH[1,:],'k',linewidth=3, marker='o', markersize = 12)
    plt.xlabel('Time Resolution, ms',fontsize=25)
    plt.xticks(fontsize=25)
    #plt.axis["right"].set_visible(False)
    plt.ylabel('Projection onto PSTH',fontsize=25)
    plt.yticks(fontsize=25)
    prop = fm.FontProperties(size=20)
    plt.legend(('1D model','2D model'),loc='upper left',prop=prop)
    plt.tight_layout()
    plt.show()





### FIND P(spike|s0,s1,s2) ####
OLD = 0
if OLD is 1:
    STA_LEN = STA_TIME/INTSTEP
    Ps_STA = np.zeros(END-START+1)
    Ps_2d = np.zeros(END-START+1)
    for i in range(START,END):
    
        S0 = np.dot(STA,Current[i-STA_LEN:i])
        Ps_STA[i-START] = Pspike_STA(S0)
    
        S1 = np.dot(Mode1,Current[i-STA_LEN:i])
        S2 = np.dot(Mode2,Current[i-STA_LEN:i])
        Ps_2d[i-START] = Pspike_2d(S1,S2)







    #BINS = np.arange(-20,18,0.5)
    #Pm12_spike = np.zeros((len(BINS),len(BINS)))
    #Pm12 = np.zeros((len(BINS),len(BINS)))
    #for i in range(0,len(BINS)):
    #    for j in range(0,len(BINS)):
    #        Pm12_spike[i,j] = Mode1_PDF(BINS[i]) + Mode1_PDF(BINS[j])
    #        Pm12[i,j] = Mode1Noise_PDF(BINS[i]) + Mode1Noise_PDF(BINS[j])
            
    #Joint = np.genfromtxt('JointProb.csv', delimiter=',')
    #plt.figure()
    #plt.imshow(Joint, origin='lower',extent=(-20,18,-20,18))
    #plt.colorbar()
    #plt.xticks(fontsize=25)
    #plt.yticks(fontsize=25)
    #plt.xlabel('projection, SD',fontsize=22)
    #plt.ylabel('projection, SD',fontsize=22)
    #plt.tight_layout()
    #plt.show()

#BINS = np.arange(0,1.,0.05)

#normM1 = (Mode1project - min(Mode1project))/(max(Mode1project)-min(Mode1project))
#normM2 = (Mode2project - min(Mode2project))/(max(Mode2project)-min(Mode2project))
#normM1noise = (Mode1Noise - min(Mode1Noise))/(max(Mode1Noise)-min(Mode1Noise))
#normM2noise = (Mode2Noise - min(Mode2Noise))/(max(Mode2Noise)-min(Mode2Noise))

            #Pm1m2_spike,x,y = np.histogram2d(Mode1project,Mode2project, bins = (BINS,BINS))
            #Pm1m2,x,y = np.histogram2d(Mode1Noise,Mode1Noise, bins = (BINS,BINS))

            #Pspike_2d = Pm1m2_spike*Pspike/Pm1m2
            #Pspike_2d[np.isnan(Pspike_2d)] = 0
            #plt.imshow(Pspike_2d)

            #Pm0_spike,x = np.histogram(STAproject, bins=BINS)
            #Pm0, x = np.histogram(Priorproject, bins=BINS)

            #Pspike_STA = Pm0_spike*Pspike/Pm0

### HISTOGRAMS --- OLD!! ###
OFF = 1
if OFF is 0:
    Prior_Hist = np.zeros(len(BINS))
    STA_Hist = np.zeros(len(BINS))
    Mode1_Hist = np.zeros(len(BINS))
    Mode2_Hist = np.zeros(len(BINS))
    Mode1Prior_Hist = np.zeros(len(BINS))
    Mode2Prior_Hist = np.zeros(len(BINS))
    for i in range(0,len(BINS)-1):
        Start = BINS[i]
        End = BINS[i+1]
    
        Prior_Hist[i] = sum(np.logical_and(Priorproject>Start, Priorproject<=End))
        STA_Hist[i] = sum(np.logical_and(STAproject>Start, STAproject<=End))
        Mode1_Hist[i] = sum(np.logical_and(Mode1project>Start, Mode1project<=End))
        Mode2_Hist[i] = sum(np.logical_and(Mode2project>Start, Mode2project<=End))
        Mode1Prior_Hist[i] = sum(np.logical_and(Mode1Noise>Start, Mode1Noise<=End))
        Mode2Prior_Hist[i] = sum(np.logical_and(Mode2Noise>Start, Mode2Noise<=End))
