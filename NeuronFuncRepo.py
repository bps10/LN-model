from __future__ import division
import numpy as np
import matplotlib.pylab as plt
import scipy.io as sio
import numpy.linalg as lin
import NeuronModels as models
import Database as Database

# TODO:
# 0. Running list of what has been done to the data.
# 1. Find Spike locations in GenData() - integrate w Runs into single function.
# 2. Scale size of STA memory allocation based on GenData().
# 3. Git objects.
# 4. MetaData. Or Info Group.
# 5. Plotting
# 6. Move neuron models into NeuronModels

class LNmodel:

	def __init__(self):
		self.NAME = None
		self.data = None
		self.STA = None
		self.Cov = None
		self.Projection = None
		self.Bayes = None
		self.ProbOfSpike = None
		self.INTSTEP = None
		self.Files = None
		
	def FindIntStep(self):
		"""
		convert kHZ into Intstep in msec
		"""
		self.INTSTEP = (self.data['SamplingRate']**-1.0) * 1000.0
		
		
	def STA_Analysis(self, STA_TIME = 180):
		
		try: 

			self.Files.OpenDatabase(self.NAME + '.h5')

			spikenum = self.Files.QueryDatabase('STA_Analysis', 'spikenum')
			print 'Successfully loaded {0} data STA file'.format(self.NAME)
			print '# Spikes: {0}'.format(spikenum)
			self.STA = 'Full'
			self.INTSTEP = self.Files.QueryDatabase('DataProcessing', 'INTSTEP')[0][0]
			self.Files.CloseDatabase()
			
		except :
			self.Files.CreateGroup('STA_Analysis')
			print 'No {0} STA file, now generating new STA files'.format(self.NAME)
			
		if self.STA == None:
		
			self.INTSTEP = self.Files.QueryDatabase('DataProcessing', 'INTSTEP')[0][0]
			current = self.Files.QueryDatabase('DataProcessing', 'RawStim')
			voltage = self.Files.QueryDatabase('DataProcessing', 'RawVolt')
			
			TIME = len(current)*self.INTSTEP

			## Create Files ##
			STA_CURRENT = np.zeros((current.shape[1]*1000,int((STA_TIME/self.INTSTEP*2))))
			STA_VOLTAGE = np.zeros((current.shape[1]*1000,int((STA_TIME/self.INTSTEP*2))))
			
			Data_mV_Record = np.zeros((current.shape[1]*1000,int((STA_TIME/self.INTSTEP*2))))
			Data_C_Record= np.zeros((current.shape[1]*1000,int((STA_TIME/self.INTSTEP*2))))
			Prior_Rec = np.zeros((current.shape[1]*1000,int((STA_TIME/self.INTSTEP*2))))

			CUR_IND = 0
			q = 0
			for i in range(0,current.shape[1]):
			
				CURRENT = current[:,i]
				voltage_Data = voltage[:,i]
				
				CUR_IND+=1
			 
				Spikes_Data = voltage_Data > 0
				Spikes_Data = runs(Spikes_Data)

				## Find Spikes that do not fall too close to the beginning or end
				S_beg_Data = np.where(Spikes_Data[0] > int(STA_TIME/self.INTSTEP*2))
				S_end_Data = np.where(Spikes_Data[0] < (len(voltage_Data)-(int(STA_TIME/self.INTSTEP*2)+20)))
				Length_Data = np.arange(min(S_beg_Data[0]),max(S_end_Data[0])+1)
				Spikes_Data = Spikes_Data[0][Length_Data],Spikes_Data[1][Length_Data]

				for j in range(q,q+len(Spikes_Data[0])):
						   
					Peak = np.arange(Spikes_Data[0][j-q],Spikes_Data[1][j-q])
					Peak = voltage_Data[Peak]
					Height = np.argmax(Peak)
					Loc = Height + Spikes_Data[0][j-q]
					RandLoc = np.random.random_integers(7000,(len(CURRENT)-(STA_TIME/self.INTSTEP)))
					Range = np.arange(Loc-(STA_TIME/self.INTSTEP),Loc+
											(STA_TIME/self.INTSTEP), dtype=int)
					RandRANGE = np.arange(RandLoc-(STA_TIME/self.INTSTEP),RandLoc+
										(STA_TIME/self.INTSTEP), dtype=int)
					
					Data_mV_Record[j,:] = voltage_Data[Range]
					Data_C_Record[j,:] = CURRENT[Range]
					Prior_Rec[j,:] = CURRENT[RandRANGE]
				q += len(Spikes_Data[0])
			
			
			Data_C_Record = Data_C_Record[np.any(Data_C_Record,1),:]
			Data_mV_Record = Data_mV_Record[np.any(Data_mV_Record,1),:]
			Prior_Rec = Prior_Rec[0:len(Data_mV_Record),:]
				
			Data_Num_Spikes = np.array([len(Data_mV_Record)])
			print '# Spikes:', Data_Num_Spikes

			Data_STA_Current = Data_C_Record.mean(0)
			Data_STA_Voltage = Data_mV_Record.mean(0)
			
			self.Files.AddData2Database('STA_TIME', np.array([STA_TIME]), 'STA_Analysis')
			self.Files.AddData2Database('spikenum', Data_Num_Spikes, 'STA_Analysis')
			self.Files.AddData2Database('stimRecord', Data_C_Record, 'STA_Analysis')
			self.Files.AddData2Database('voltRecord', Data_mV_Record, 'STA_Analysis')
			self.Files.AddData2Database('priorRecord', Prior_Rec, 'STA_Analysis')
			self.Files.AddData2Database('STAstim', Data_STA_Current, 'STA_Analysis')
			self.Files.AddData2Database('STAvolt', Data_STA_Voltage, 'STA_Analysis')

			self.Files.CloseDatabase()		



	################
	###COVARIANCE### 
	################
	def FindCovariance(self):
		"""
		"""
			## Covariance Analysis:
		try:

			self.Files.OpenDatabase(self.NAME + '.h5')
			self.Files.QueryDatabase('Cov_Analysis','eigval')
			
			print 'Successfully loaded {0} data Cov file'.format(self.NAME)
			self.Cov = 'Full'
			self.Files.CloseDatabase()
			
		except :
			self.Files.CreateGroup('Cov_Analysis')
			print 'No {0} Cov files, now generating. This could take several minutes'.format(self.NAME)
		if self.Cov == None:

			STA_TIME = self.Files.QueryDatabase('STA_Analysis','STA_TIME')[0]
			spikenum = self.Files.QueryDatabase('STA_Analysis', 'spikenum')
			stimRecord = self.Files.QueryDatabase('STA_Analysis', 'stimRecord')
			priorRecord = self.Files.QueryDatabase('STA_Analysis', 'priorRecord')
			STAstim = self.Files.QueryDatabase('STA_Analysis', 'STAstim')
		
			Begin = 0
			End = STA_TIME / self.INTSTEP

			Data_C_spike = np.zeros((End,End))
			C_prior = np.zeros((End,End))


			STA = STAstim[Begin:End]
			mat_STA = STA*STA[:,np.newaxis]*spikenum/(spikenum - 1.0)

			for i in range(0,spikenum):
				a = stimRecord[i,Begin:End]
				b = priorRecord[i,Begin:End]
				
				mat_spike = a*a[:,np.newaxis]
				mat_prior = b*b[:,np.newaxis]
				
				
				Data_C_spike += (mat_spike - mat_STA)
				C_prior += mat_prior
				
			### FIND MEANS ###
			Data_C_spike = Data_C_spike/(spikenum - 1.0)
			C_prior = C_prior/(spikenum - 1.0)

			### FIND DELTA COV ###
			Data_C_delta = Data_C_spike - C_prior

			### EigenValues, EigenVectors
			Data_E, V = lin.eig(Data_C_delta)
			I = np.argsort(Data_E)
			Data_E = Data_E[I][:( STA_TIME / self.INTSTEP)]
			Data_Vect = V[:,I][:,:( STA_TIME / self.INTSTEP)]


			self.Files.AddData2Database('cov', Data_C_delta, 'Cov_Analysis')
			self.Files.AddData2Database('eigval', Data_E, 'Cov_Analysis')
			self.Files.AddData2Database('eigvect', Data_Vect, 'Cov_Analysis')



	## LN model analysis   ##
	#########################
		
	def FindProjection(self):
		
		## Projection:
		try:

			self.Files.OpenDatabase(self.NAME + '.h5')
			self.Files.QueryDatabase('Projection','STAproject')
			print 'Successfully loaded {0} Projection data'.format(self.NAME)
			
			self.Projection = 'Full'
			self.Files.CloseDatabase()
			
		except :
			self.Files.CreateGroup('Projection')
			print 'No {0} Projection files, now generating. This could take several minutes'.format(self.NAME)
			
		if self.Projection == None:
			
			STA_TIME = self.Files.QueryDatabase('STA_Analysis', 'STA_TIME')
			Prior = self.Files.QueryDatabase('STA_Analysis','priorRecord')
			stimRecord = self.Files.QueryDatabase('STA_Analysis', 'stimRecord')
			STAstim = self.Files.QueryDatabase('STA_Analysis', 'STAstim')
			EigVect = self.Files.QueryDatabase('Cov_Analysis', 'eigvect')
			
			Prior = Prior[:,0:int(STA_TIME / self.INTSTEP)]
			
			Current = stimRecord
			Current = Current[:,0:int(STA_TIME / self.INTSTEP)]
			STAstim = STAstim[0:int(STA_TIME / self.INTSTEP)]
			STAstim = STAstim/lin.norm(STAstim)

			Mode1 = EigVect[:int(STA_TIME / self.INTSTEP),0]
			Mode2 = EigVect[:int(STA_TIME / self.INTSTEP),1]
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

				Priorproj[i] = np.dot(STAstim,Pr)
				STAproj[i] = np.dot(STAstim,B)
				Mode1Pr[i] = np.dot(Mode1,Pr)
				Mode2Pr[i] = np.dot(Mode2,Pr)
				Mode1proj[i] = np.dot(Mode1,B)
				Mode2proj[i] = np.dot(Mode2,B)


			self.Files.AddData2Database('priorproject',Priorproj, 'Projection')
			self.Files.AddData2Database('STAproject', STAproj, 'Projection')
			self.Files.AddData2Database('Mode1Pr', Mode1Pr, 'Projection')
			self.Files.AddData2Database('Mode2Pr', Mode2Pr, 'Projection')
			self.Files.AddData2Database('Mode1proj', Mode1proj, 'Projection')
			self.Files.AddData2Database('Mode2proj', Mode2proj, 'Projection')

			self.Files.CloseDatabase()
			

		
		
	def BayesPspike(self, BIN_SIZE = 0.5):
			## Pspike histograms to set up LN model:
		try:

			self.Files.OpenDatabase(self.NAME + '.h5')
			
			self.Files.QueryDatabase('Bayes','BINS')
			print 'Successfully loaded {0} data Bayes file'.format(self.NAME)
			
			self.Bayes = 'Full'

			spikenum = self.Files.QueryDatabase('STA_Analysis', 'spikenum')
			RawVolt = self.Files.QueryDatabase('DataProcessing', 'RawVolt')
			
			LNmodel.FindP_spike(self, spikenum, RawVolt)
			print '{0} spikes/second'.format(self.Pspike)

			self.Files.CloseDatabase()
			
		except :
			
			self.Files.CreateGroup('Bayes')
			print 'No {0} Bayes files, now generating. This could take several minutes'.format(self.NAME)
			
		if self.Bayes == None:

			Priorproject = self.Files.QueryDatabase('Projection', 'priorproject')
			STAproject = self.Files.QueryDatabase('Projection', 'STAproject')
			Mode1project = self.Files.QueryDatabase('Projection', 'Mode1proj')
			Mode2project = self.Files.QueryDatabase('Projection', 'Mode2proj')
			Mode1Noise = self.Files.QueryDatabase('Projection', 'Mode1Pr')
			Mode2Noise = self.Files.QueryDatabase('Projection', 'Mode2Pr')

			spikenum = self.Files.QueryDatabase('STA_Analysis', 'spikenum')
			RawVolt = self.Files.QueryDatabase('DataProcessing', 'RawVolt')
			
			BINS = np.arange(-20,20,BIN_SIZE)
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

			
			LNmodel.FindP_spike(self, spikenum, RawVolt)[0]
			print '{0} spikes/second'.format(self.Pspike)
			
			## Bayes Theorem:
			Pspike_STA = STA_Hist * self.Pspike / Prior_Hist
			Pspike_2d = M12_Spike * self.Pspike / M12_Prior


			self.Files.AddData2Database('Pspike_STA', Pspike_STA, 'Bayes')
			self.Files.AddData2Database('Pspike_2d', Pspike_2d, 'Bayes')
			self.Files.AddData2Database('BINS', BINS, 'Bayes')

			self.Files.CloseDatabase()
		

		
	def FindProbOfSpike(self):
			## LNmodel final set up
		try:
			
			self.Files.OpenDatabase(self.NAME + '.h5')
			self.Files.QueryDatabase('ProbOfSpike','Ps_2d')
			print 'Successfully loaded {0} Prob of Spike files'.format(self.NAME)
			self.Files.CloseDatabase()
			
			self.ProbOfSpike = 'Full'
		except :
			
			self.Files.CreateGroup('ProbOfSpike')
			print 'No {0} Bayes files, now generating. This could take several minutes'.format(self.NAME)
			
		if self.ProbOfSpike == None:
		
			## LOAD EigenModes ##
			Current = self.Files.QueryDatabase('DataProcessing', 'RepStim')
			Voltage = self.Files.QueryDatabase('DataProcessing', 'RepVolt')
			STAstim = self.Files.QueryDatabase('STA_Analysis', 'STAstim')
			STA_TIME = self.Files.QueryDatabase('STA_Analysis', 'STA_TIME')[0]
			EigVect = self.Files.QueryDatabase('Cov_Analysis', 'eigvect')
			BINS = self.Files.QueryDatabase('Bayes', 'BINS')
			Pspike_STA = self.Files.QueryDatabase('Bayes', 'Pspike_STA')
			Pspike_2d = self.Files.QueryDatabase('Bayes', 'Pspike_2d')
			
			STAstim = STAstim[0:STA_TIME / self.INTSTEP]
			STAstim = STAstim / lin.norm(STAstim)
			
			Mode1 = EigVect[:,0]
			Mode2 = EigVect[:,1]
			del(EigVect)
			
			END = 1e5
			START = 1e4

			### FIND P(spike|s0,s1,s2) ####
			STA_LEN = STA_TIME / self.INTSTEP
			Ps_STA = np.zeros((END-START+1, Current.shape[1]))
			Ps_2d = np.zeros((END-START+1, Current.shape[1]))
			S1 = np.zeros((END-START+1))
			S2 = np.zeros((END-START+1))
			
			for j in range(0, Current.shape[1]):
				for i in range(int(START),int(END)):
					
					S0 = round((float(np.dot(STAstim,Current[i-STA_LEN:i,j]))/0.25)*0.25)
					loc = np.where(BINS==[S0])
					Ps_STA[i-START,j] = Pspike_STA[loc]
					
					S1[i-START] = round((float(np.dot(Mode1,Current[i-STA_LEN:i,j]))/0.25)*0.25)
					S2[i-START] = round((float(np.dot(Mode2,Current[i-STA_LEN:i,j]))/0.25)*0.25)
					loc1 = np.where(BINS==[S1[i-START]])
					loc2 = np.where(BINS==[S2[i-START]])
					Ps_2d[i-START,j] = Pspike_2d[loc1,loc2]

			self.Files.AddData2Database('Ps_2d', Ps_2d, 'ProbOfSpike')
			self.Files.AddData2Database('Ps_STA', Ps_STA, 'ProbOfSpike')

			self.Files.CloseDatabase()
			
			
	def FindP_spike(self, spikenum, RawVolt):
		"""
		## in Hz. Spikes/len(one row)*dt*20(rows)/1000(msec/sec)
		"""
		
		self.Pspike = spikenum / (RawVolt.shape[0] * RawVolt.shape[1] * self.INTSTEP / 1000.0) 





## Data Preprocessing ##
########################
class DataProcessing(LNmodel,Database.Database):

	
	def GetData(self, DIRECTORY, SaveName):
		
		try:
			self.Files = Database.Database()
			self.Files.OpenDatabase(SaveName + '.h5')
			#= np.load('Data/' + SaveName + 'ProcessedData.npz')
			self.NAME = np.array([SaveName])[0]
			
			SamplingRate = self.Files.QueryDatabase('DataProcessing','SamplingRate')
			print 'Successfully found {0} database.'.format(SaveName)
			self.data = 'Full'
			self.Files.CloseDatabase()
			
		except :
			
			self.Files.CreateGroup('DataProcessing')
			print 'No preprocessed data. Now trying raw data.'

		if self.data == None:
			self.NAME = np.array([SaveName])[0]
		
			RawData = sio.loadmat(DIRECTORY)
			SamplingRate = np.array([10000]) # in Hz
			Volt = RawData['V'][0][0]
			Stim = RawData['stim']
			RepLoc = RawData['rep1']

			RepVolt = np.zeros((RepLoc.shape[1],Volt.shape[1]))
			RepStim = np.zeros((RepLoc.shape[1],Volt.shape[1]))

			for i in range ( 0 , Stim.shape[1]):
				RepVolt[:,i] = Volt[RepLoc , i]
				RepStim[:,i] = Stim[RepLoc , i]
			
			self.INTSTEP = np.array([(SamplingRate**-1.0) * 1000.0])
			
			self.Files.AddData2Database('SamplingRate', SamplingRate, 'DataProcessing')
			self.Files.AddData2Database('INTSTEP', self.INTSTEP, 'DataProcessing')
			self.Files.AddData2Database('RawVolt', Volt, 'DataProcessing')
			self.Files.AddData2Database('RawStim', Stim, 'DataProcessing')
			self.Files.AddData2Database('RepLoc', RepLoc, 'DataProcessing')
			self.Files.AddData2Database('RepVolt', RepVolt, 'DataProcessing')
			self.Files.AddData2Database('RepStim', RepStim, 'DataProcessing')
			self.Files.AddData2Database('name', np.array([SaveName]), 'DataProcessing')

			self.Files.CloseDatabase()		
					
					

class NeuronModel(LNmodel, Database.Database):

	
	def GenModelData(self, SaveName, DataName,  params = 'EvolvedParam1_8.csv', model = models.QUADmodel):
		"""
		"""
		try:
			self.Files = Database.Database()
			self.Files.OpenDatabase(SaveName + '.h5')
			self.NAME = np.array([SaveName])[0]
			self.Files.QueryDatabase('DataProcessing', 'SamplingRate')
			
			DATA = Database.Database()
			DATA.OpenDatabase(DataName + '.h5')
			SamplingRate = DATA.QueryDatabase('DataProcessing', 'SamplingRate')
			
			print 'Successfully opened {0} database.'.format(self.NAME)
			self.Files.CloseDatabase()
			self.data = 'Full'
		except :
			
			self.Files.CreateGroup('DataProcessing')
			print 'No preprocessed data. Now trying raw data.'

		if self.data == None:

			DATA = Database.Database()
			DATA.OpenDatabase(DataName + '.h5')
			self.INTSTEP = DATA.QueryDatabase('DataProcessing', 'INTSTEP')
			RepLoc = DATA.QueryDatabase('DataProcessing', 'RepLoc')
			RawStim = DATA.QueryDatabase('DataProcessing', 'RawStim')
			RawVolt = DATA.QueryDatabase('DataProcessing', 'RawVolt')
			RepStim = DATA.QueryDatabase('DataProcessing', 'RepStim')
			DATA.CloseDatabase()
			
			params = np.genfromtxt(params,delimiter=',')
			
			current = RawStim * 100.0
			ModelVolt = np.zeros((current.shape[0],current.shape[1]))
			for i in range(0, current.shape[1]):
				
				ModelVolt[:,i] = model(params, current[:,i], self.INTSTEP)
			
			RepModelVolt = np.zeros((RepLoc.shape[1], RawVolt.shape[1]))

			for i in range ( 0 , RawStim.shape[1]):
				RepModelVolt[:,i] = ModelVolt[ RepLoc , i]

			
			self.Files.AddData2Database('SamplingRate', SamplingRate, 'DataProcessing')
			self.Files.AddData2Database('INTSTEP', self.INTSTEP, 'DataProcessing')
			self.Files.AddData2Database('RawVolt', ModelVolt, 'DataProcessing')
			self.Files.AddData2Database('RawStim', current, 'DataProcessing')
			self.Files.AddData2Database('RepLoc',RepLoc, 'DataProcessing')
			self.Files.AddData2Database('RepVolt', RepModelVolt, 'DataProcessing')
			self.Files.AddData2Database('RepStim', RepStim, 'DataProcessing')
			self.Files.AddData2Database('ModelParams', params, 'DataProcessing')
			self.Files.AddData2Database('name', np.array([SaveName]), 'DataProcessing')

					
			self.Files.CloseDatabase()		

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
		



	
"""
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
"""
