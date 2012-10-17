import numpy as np
import matplotlib.pylab as plt
import NeuronModels as models
import scipy.io as sio

## Data Preprocessing ##
########################
def GetData(DIRECTORY = '/120511c3.mat', SaveName = 'Data\RebeccaProcessedData.npz'):
	
	try:
		Data = np.load('Data\RebeccaProcessedData.npz')
		
	except IOError:
		print 'No preprocessed data. Now trying raw data.'


	RawData = sio.loadmat(DIRECTORY)
	SamplingRate = 10000 # in Hz
	Volt = RawData['V'][0][0]
	Stim = RawData['stim']
	RepLoc = RawData['rep1']

	RepVolt = np.zeros((RepLoc.shape[1],Volt.shape[1]))
	RepStim = np.zeros((RepLoc.shape[1],Volt.shape[1]))

	for i in range ( 0 , Stim.shape[1]):
		RepVolt[:,i] = Volt[RepLoc , i]
		RepStim[:,i] = Stim[RepLoc , i]

	Data = 	{
			'SamplingRate': SamplingRate,
			'RawVolt': Volt,
			'RawStim': Stim,
			'RepLoc': RepLoc,
			'RepVolt': RepVolt,
			'RepStim': RepStim,
			'name': 'Rebecca'
			}
	np.savez( SaveName, SamplingRate=SamplingRate, RawVolt=Volt, RawStim=Stim,
			RepLoc=RepLoc, RepVolt=RepVolt, RepStim=RepStim)
		
	return Data


def GenModelData(Data, model = models.QUADmodel, params = 'EvolvedParam1_8.csv',
				SaveName = 'ModelVoltage.npz'):
	"""
	"""
	INTSTEP = FindIntStep(Data)
	
	params = np.genfromtxt(params,delimiter=',')
	
	current = Data['RepStim']
	ModelVolt = np.zeros((current.shape[0],current.shape[1]))
	for i in range(0, current.shape[1]):
		
		ModelVolt[:,i] = model(params, current[:,i], INTSTEP)
	
	Model =	{
			'SamplingRate': Data['SamplingRate'],
			'RepStim': current,
			'RepVolt': ModelVolt
			}
	
	np.savez('/Data/' + SaveName, stim=current,volt=ModelVolt, 
								SamplingRate=Data['SamplingRate'])
	
	return Model

	
	
######################
#### STA ANALYSIS ####
######################

def FindIntStep(Data):
	"""
	convert kHZ into Intstep in msec
	"""
	return (Data['SamplingRate']**-1)*1000
	
def STA_Analysis(Data, STA_TIME = 200):
	
	INTSTEP = FindIntStep(Data)## in msec
	current = Data['RepStim']
	voltage = Data['RepVolt']
	
	TIME = len(current)*INTSTEP

	## Create Files ##
	STA_CURRENT = np.zeros((current.shape[1]*1500,int((STA_TIME/INTSTEP*2))))
	STA_VOLTAGE = np.zeros((current.shape[1]*1500,int((STA_TIME/INTSTEP*2))))
	
	Data_mV_Record = np.zeros((current.shape[1]*1500,int((STA_TIME/INTSTEP*2))))
	Data_C_Record= np.zeros((current.shape[1]*1500,int((STA_TIME/INTSTEP*2))))
	Prior_Rec = np.zeros((current.shape[1]*1500,int((STA_TIME/INTSTEP*2))))

	CUR_IND = 0
	q = 0
	for i in range(0,current.shape[1]):
	
		CURRENT = current[:,i]
		voltage_Data = voltage[:,i]
		
		CUR_IND+=1
	 
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
			RandLoc = np.random.random_integers(7000,(len(CURRENT)-(STA_TIME/INTSTEP)))
			Range = np.arange(Loc-(STA_TIME/INTSTEP),Loc+(STA_TIME/INTSTEP), dtype=int)
			RandRANGE = np.arange(RandLoc-(STA_TIME/INTSTEP),RandLoc+(STA_TIME/INTSTEP), dtype=int)
			
			Data_mV_Record[j,:] = voltage_Data[Range]
			Data_C_Record[j,:] = CURRENT[Range]
			Prior_Rec[j,:] = CURRENT[RandRANGE]
		q += len(Spikes_Data[0])
	
	
	Data_C_Record = Data_C_Record[np.any(Data_C_Record,1),:]
	Data_mV_Record = Data_mV_Record[np.any(Data_mV_Record,1),:]
	Prior_Rec = Prior_Rec[0:len(Data_mV_Record),:]
		
	Data_Num_Spikes = len(Data_mV_Record)
	print '# Data Spikes:', Data_Num_Spikes

	Data_STA_Current = Data_C_Record.mean(0)
	Data_STA_Voltage = Data_mV_Record.mean(0)
	
	STA = 	{
			'STA_TIME': STA_TIME,
			'INTSTEP': INTSTEP,
			'spikenum': Data_Num_Spikes,
			'stimRecord': Data_C_Record,
			'voltRecord': Data_mV_Record,
			'STAstim': Data_STA_Current,
			'STAvolt': Data_STA_Voltage
			}
	np.savez('/Data/' + Data['name'] + 'STA.npz', stimRecord=Data_C_Record, 
			voltRecord=Data_mV_Record,STAstim=Data_STA_Current,STAvolt=Data_STA_Voltage)
	
	return STA







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

## LNmodel functions ##
	

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
