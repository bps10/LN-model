from __future__ import division
import numpy as np
import matplotlib.pylab as plt
import scipy.io as sio
import numpy.linalg as lin
import NeuronModels as models


## Data Preprocessing ##
########################
def GetData(DIRECTORY = '/120511c3.mat', SaveName = 'Rebecca'):
	
	try:
		Data = np.load('Data/' + SaveName + 'ProcessedData.npz')
		print 'Successfully loaded {0} preprocessed data from file.'.format(SaveName)
	except IOError:
		Data = []
		print 'No preprocessed data. Now trying raw data.'

	if Data == []:
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
				'name': SaveName
				}
		np.savez('Data/' + SaveName + 'ProcessedData.npz', 
				SamplingRate=SamplingRate, 
				RawVolt=Volt, 
				RawStim=Stim,
				RepLoc=RepLoc, 
				RepVolt=RepVolt, 
				RepStim=RepStim, 
				name=SaveName)
		
	return Data


def GenModelData(Data, model = models.QUADmodel, params = 'EvolvedParam1_8.csv',
				SaveName = 'Quad'):
	"""
	"""
	try:
		Model = np.load('Data/' + SaveName + 'ModelProcessedData.npz')
		print 'Successfully loaded {0} preprocessed model data from file.'.format(SaveName)
	except IOError:
		Model = []
		print 'No preprocessed model data. Now trying raw data.'

	if Model == []:
		INTSTEP = FindIntStep(Data)
		
		params = np.genfromtxt(params,delimiter=',')
		
		current = Data['RawStim'] * 100.0
		ModelVolt = np.zeros((current.shape[0],current.shape[1]))
		for i in range(0, current.shape[1]):
			
			ModelVolt[:,i] = model(params, current[:,i], INTSTEP)
		
		RepModelVolt = np.zeros((Data['RepLoc'].shape[1],Data['RawVolt'].shape[1]))
		RepStim = Data['RawStim']
		for i in range ( 0 , Data['RawStim'].shape[1]):
			RepModelVolt[:,i] = ModelVolt[Data['RepLoc'] , i]

			
		Model =	{
				'SamplingRate': Data['SamplingRate'],
				'RawStim': current,
				'RawVolt': ModelVolt,
				'RepStim': RepStim,
				'RepVolt': RepModelVolt,
				'name': SaveName
				}
		
		np.savez('Data/' + SaveName + 'ModelProcessedData.npz',
				RawStim=current,
				RawVolt=ModelVolt,
				RepStim = RepStim,
				RepVolt = RepModelVolt,
				SamplingRate=Data['SamplingRate'], 
				name = SaveName)
	
	return Model

	
	
######################
#### STA ANALYSIS ####
######################

def FindIntStep(Data):
	"""
	convert kHZ into Intstep in msec
	"""
	return (Data['SamplingRate']**-1.0)*1000.0
	
	
def STA_Analysis(Data, STA_TIME = 200):
	
	INTSTEP = FindIntStep(Data)## in msec
	current = Data['RawStim']
	voltage = Data['RawVolt']
	
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
	
	name = str(Data['name'])
	STA = 	{
			'STA_TIME': STA_TIME,
			'INTSTEP': INTSTEP,
			'spikenum': Data_Num_Spikes,
			'stimRecord': Data_C_Record,
			'voltRecord': Data_mV_Record,
			'priorRecord': Prior_Rec,
			'STAstim': Data_STA_Current,
			'STAvolt': Data_STA_Voltage,
			'name': name
			}
	
	
	np.savez('Data/' + name + 'STA.npz', 
			STA_TIME=STA_TIME,
			INTSTEP=INTSTEP,
			spikenum=Data_Num_Spikes,
			stimRecord=Data_C_Record, 
			voltRecord=Data_mV_Record,
			priorRecord=Prior_Rec,
			STAstim=Data_STA_Current,
			STAvolt=Data_STA_Voltage,
			name=name)
	
	return STA


################
###COVARIANCE### 
################
def FindCovariance(Data):
	"""
	"""
	
	Begin = 0
	End = Data['STA_TIME']/Data['INTSTEP']

	Data_C_spike = np.zeros((End,End))
	C_prior = np.zeros((End,End))


	STA = Data['STAstim'][Begin:End]
	mat_STA = STA*STA[:,np.newaxis]*Data['spikenum']/(Data['spikenum']-1)

	for i in range(0,Data['spikenum']):
		a = Data['stimRecord'][i,Begin:End]
		b = Data['priorRecord'][i,Begin:End]
		
		mat_spike = a*a[:,np.newaxis]
		mat_prior = b*b[:,np.newaxis]
		
		
		Data_C_spike += (mat_spike - mat_STA)
		C_prior += mat_prior
		
	### FIND MEANS ###
	Data_C_spike = Data_C_spike/(Data['spikenum']-1)
	C_prior = C_prior/(Data['spikenum']-1)

	### FIND DELTA COV ###
	Data_C_delta = Data_C_spike - C_prior

	### EigenValues, EigenVectors
	Data_E, V = lin.eig(Data_C_delta)
	I = np.argsort(Data_E)
	Data_E = Data_E[I][:( Data['STA_TIME']/Data['INTSTEP'])]
	Data_Vect = V[:,I][:,:( Data['STA_TIME']/Data['INTSTEP'])]

	## needs to be reformatted this way for savez for some reason.
	name = str(Data['name'])
	STA_TIME = Data['STA_TIME']
	INTSTEP = Data['INTSTEP']
	
	Covariance =	{
					'name': name,
					'STA_TIME': STA_TIME,
					'INTSTEP': INTSTEP,
					'cov': Data_C_delta,
					'eigval': Data_E,
					'eigvect': Data_Vect
					}

	
	np.savez('Data/' + name + 'Covariance.npz', 
			name=name,
			STA_TIME=STA_TIME,
			INTSTEP=INTSTEP,
			cov=Data_C_delta,
			eigval=Data_E,
			eigvect=Data_Vect)
			
	return Covariance


## LN model analysis   ##
#########################
	
def FindProjection(Data,STA, Cov, STA_TIME = 180):

	INTSTEP = FindIntStep(Data)
	##STA_TIME = STA['STA_TIME']
	
	Prior = STA['priorRecord']
	Prior = Prior[:,0:int(STA_TIME/INTSTEP)]
	
	Current = STA['stimRecord']
	Current = Current[:,0:int(STA_TIME/INTSTEP)]
	sta = STA['STAstim']
	sta = sta[0:int(STA_TIME/INTSTEP)]
	sta = sta/lin.norm(sta)

	EigVect = Cov['eigvect']
	Mode1 = EigVect[:int(STA_TIME/INTSTEP),0]
	Mode2 = EigVect[:int(STA_TIME/INTSTEP),1]
	del(EigVect)
	

	Priorproj = np.zeros(len(Current[:,0]))
	STAproj = np.zeros(len(Current[:,0]))
	Mode1proj = np.zeros(len(Current[:,0]))
	Mode2proj = np.zeros(len(Current[:,0]))
	Mode1Pr = np.zeros(len(Current[:,0]))
	Mode2Pr = np.zeros(len(Current[:,0]))

	for i in range(len(Current[:,0])):
		B =  Current[i,:]
		Pr = Prior[i,:]

		Priorproj[i] = np.dot(sta,Pr)
		STAproj[i] = np.dot(sta,B)
		Mode1Pr[i] = np.dot(Mode1,Pr)
		Mode2Pr[i] = np.dot(Mode2,Pr)
		Mode1proj[i] = np.dot(Mode1,B)
		Mode2proj[i] = np.dot(Mode2,B)
	
	name = str(Data['name'])
	Projection = 	{
					'priorproject': Priorproj,
					'STAproject': STAproj,
					'Mode1Pr': Mode1Pr,
					'Mode2Pr': Mode2Pr,
					'Mode1proj': Mode1proj,
					'Mode2proj': Mode2proj,
					'name': name
					}
					
	np.savez('Data/' + name + 'Project.npz',
			priorproject=Priorproj,
			STAproject=STAproj,
			Mode1Pr=Mode1Pr,
			Mode2Pr=Mode2Pr,
			Mode1proj=Mode1proj,
			Mode2proj=Mode2proj,
			name=name)
	return Projection
	
	

	
	
def PspikeHist(Projection, Pspike, BIN_SIZE = 0.5):

	Priorproject = Projection['priorproject']
	STAproject = Projection['STAproject']
	Mode1project = Projection['Mode1proj']
	Mode2project = Projection['Mode2proj']
	Mode1Noise = Projection['Mode1Pr']
	Mode2Noise = Projection['Mode2Pr']
	
	BINS = np.arange(-20,18,BIN_SIZE)
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


	Pspike_STA = STA_Hist * Pspike / Prior_Hist
	Pspike_2d = M12_Spike * Pspike / M12_Prior
	
	PspikeData = 	{
					'Pspike_STA': Pspike_STA,
					'Pspike_2d': Pspike_2d,
					'name': Projection['name']
					}
	
	np.savez('Data/' + str(Projection['name']) + 'PspikeData.npz',
			Pspike_STA=Pspike_STA,
			Pspike_2d=Pspike_2d)
	
	return PspikeData

	
	
def LNmodel(Data, Cov):
	## LOAD EigenModes ##
	Current = Data['RepStim']
	Voltage = Data['RepVolt']
	sta = STA['STAvolt']
	sta = sta[0:STA_TIME/INTSTEP]
	sta = sta/lin.norm(sta)
	
	EigVect = Cov['eigvect']
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
		
		S0 = round((float(np.dot(sta,Current[i-STA_LEN:i]))/0.25)*0.25)
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
	
	LNmodel =	{
				'spikes': Spikes,
				'Ps_2d': Ps_2d,
				'Ps_STA': Ps_STA
				}
	np.savez('Data/' + Data['name'] + 'LNmodelData.npz',
			spikes=Spikes,
			Ps_2d=Ps_2d,
			Ps_STA=Ps_STA)

	
	return LNmodel

	
## General functions ##
#######################
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
	

## Analysis Functions ##
########################

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


	

