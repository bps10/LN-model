import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pylab as plt

sys.path.append('C:\\Users\\Brian\\Documents\\Neitz-Lab\\Python')
import PlottingFun as pf

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


def PlotHistOutline():

    ## PLOT HISTOGRAM OUTLINES ##
    Data_Hist,STA_Hist,Model_Hist,BINS = Neuron.Hist(HIST_BIN_SIZE)
    
    bins,Outline_Data_Hist = histOutline(Data_Hist,BINS)
    bins,Outline_STA_Hist = histOutline(STA_Hist,BINS)
    bins,Outline_Model_Hist = histOutline(Model_Hist,BINS)

	
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

	
def PSTH():
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

	
def STAplot(STA):

	fig = plt.figure(figsize=(12,8))
	X = np.arange(-STA_TIME/INTSTEP,STA_TIME/INTSTEP,dtype=float)*INTSTEP
	ax = fig.add_subplot(111)
	ax.plot(X[0:(STA_TIME/INTSTEP)+50], Data_STA_Current[0:(STA_TIME/INTSTEP)+50],
				linewidth=3, color='k')
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.ylabel('current(pA)', fontsize = 20)
	plt.legend(('data'), loc='upper right')
	plt.show()
