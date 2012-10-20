from __future__ import division
import numpy as np
import matplotlib.pylab as plt
import scipy.io as sio
import numpy.linalg as lin
import NeuronModels as models
import Database as Database

# ToDO:
# 1. Add a __getitem__ function and clean up.
# 2. Generalize: particularly, GenData()

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
		
		
	def STA_Analysis(self, STA_TIME = 180, SaveFiles = 'no'):
		
		try: 
			self.Files = Database.Database()
			self.Files.OpenDatabase(self.NAME[0] + '.h5')

			spikenum = self.Files.QueryDatabase('STA_Analysis', 'spikenum')
			print 'Successfully loaded {0} data STA file'.format(self.NAME)
			print '# Spikes: {0}'.format(spikenum)
			self.STA = 'Full'
			
		except :
			self.Files.CreateGroup('STA_Analysis')
			print 'No {0} STA file, now generating new STA files'.format(self.NAME)
		if self.STA == None:
		
			self.INTSTEP = self.Files.QueryDatabase('DataProcessing', 'INTSTEP')
			current = self.Files.QueryDatabase('DataProcessing', 'RawStim')
			voltage = self.Files.QueryDatabase('DataProcessing', 'RawVolt')
			
			TIME = len(current)*self.INTSTEP

			## Create Files ##
			STA_CURRENT = np.zeros((current.shape[1]*1500,int((STA_TIME/self.INTSTEP*2))))
			STA_VOLTAGE = np.zeros((current.shape[1]*1500,int((STA_TIME/self.INTSTEP*2))))
			
			Data_mV_Record = np.zeros((current.shape[1]*1500,int((STA_TIME/self.INTSTEP*2))))
			Data_C_Record= np.zeros((current.shape[1]*1500,int((STA_TIME/self.INTSTEP*2))))
			Prior_Rec = np.zeros((current.shape[1]*1500,int((STA_TIME/self.INTSTEP*2))))

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
				S_end_Data = np.where(Spikes_Data[0] < 
										(len(voltage_Data)-(int(STA_TIME/self.INTSTEP*2)+20)))
				Length_Data = np.arange(min(S_beg_Data[0]),max(S_end_Data[0])+1)
				Spikes_Data = Spikes_Data[0][Length_Data],Spikes_Data[1][Length_Data]

				for j in range(q,q+len(Spikes_Data[0])):
						   
					Peak = np.arange(Spikes_Data[0][j-q],Spikes_Data[1][j-q])
					Peak = voltage_Data[Peak]
					Height = np.argmax(Peak)
					Loc = Height + Spikes_Data[0][j-q]
					RandLoc = np.random.random_integers(7000,(len(CURRENT)-
														(STA_TIME/self.INTSTEP)))
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
			
			self.Files.AddData2Database('STA_TIME', np.array([STA_TIME]), 'DataProcessing')
			self.Files.AddData2Database('spikenum', Data_Num_Spikes, 'DataProcessing')
			self.Files.AddData2Database('stimRecord', Data_C_Record, 'DataProcessing')
			self.Files.AddData2Database('voltRecord', Data_mV_Record, 'DataProcessing')
			self.Files.AddData2Database('priorRecord', Prior_Rec, 'DataProcessing')
			self.Files.AddData2Database('STAstim', Data_STA_Current, 'DataProcessing')
			self.Files.AddData2Database('STAvolt', Data_STA_Voltage, 'DataProcessing')

			self.Files.CloseDatabase()		



	################
	###COVARIANCE### 
	################
	def FindCovariance(self, SaveFiles = 'no'):
		"""
		"""
			## Covariance Analysis:
		try:
			self.Cov = np.load('Data/' + str(self.data['name']) + 'Covariance.npz')
			print 'Successfully loaded {0} data Cov file'.format(self.data['name'])
		except IOError:
			print 'No {0} Cov file, now generating. This could take several minutes'.format(
					self.data['name'])
		if self.Cov == None:

		
			Begin = 0
			End = self.STA['STA_TIME'] / self.INTSTEP

			Data_C_spike = np.zeros((End,End))
			C_prior = np.zeros((End,End))


			STA = self.STA['STAstim'][Begin:End]
			mat_STA = STA*STA[:,np.newaxis]*self.STA['spikenum']/(self.STA['spikenum']-1)

			for i in range(0,self.STA['spikenum']):
				a = self.STA['stimRecord'][i,Begin:End]
				b = self.STA['priorRecord'][i,Begin:End]
				
				mat_spike = a*a[:,np.newaxis]
				mat_prior = b*b[:,np.newaxis]
				
				
				Data_C_spike += (mat_spike - mat_STA)
				C_prior += mat_prior
				
			### FIND MEANS ###
			Data_C_spike = Data_C_spike/(self.STA['spikenum']-1)
			C_prior = C_prior/(self.STA['spikenum']-1)

			### FIND DELTA COV ###
			Data_C_delta = Data_C_spike - C_prior

			### EigenValues, EigenVectors
			Data_E, V = lin.eig(Data_C_delta)
			I = np.argsort(Data_E)
			Data_E = Data_E[I][:( self.STA['STA_TIME']/self.INTSTEP)]
			Data_Vect = V[:,I][:,:( self.STA['STA_TIME']/self.INTSTEP)]

			## needs to be reformatted this way for savez for some reason.
			name = str(self.data['name'])
			STA_TIME = self.STA['STA_TIME']
			INTSTEP = self.INTSTEP
			
			self.Cov =	{
								'cov': Data_C_delta,
								'eigval': Data_E,
								'eigvect': Data_Vect
								}

			if SaveFiles.lower() == 'yes':
				np.savez('Data/' + name + 'Covariance.npz', 
						cov=Data_C_delta,
						eigval=Data_E,
						eigvect=Data_Vect)



	## LN model analysis   ##
	#########################
		
	def FindProjection(self, SaveFiles = 'no'):
		
		## Projection:
		try: 
			self.Projection = np.load('Data/' + str(self.data['name']) + 'Project.npz')
			print 'Successfully loaded {0}  Projection file'.format(self.data['name'])
		except IOError:
			print 'No {0} projection file, now generating new'.format(self.data['name'])
		if self.Projection == None:

			
			Prior = self.STA['priorRecord']
			Prior = Prior[:,0:int(self.STA['STA_TIME']/self.INTSTEP)]
			
			Current = self.STA['stimRecord']
			Current = Current[:,0:int(self.STA['STA_TIME']/self.INTSTEP)]
			sta = self.STA['STAstim']
			sta = sta[0:int(self.STA['STA_TIME']/self.INTSTEP)]
			sta = sta/lin.norm(sta)

			EigVect = self.Cov['eigvect']
			Mode1 = EigVect[:int(self.STA['STA_TIME']/self.INTSTEP),0]
			Mode2 = EigVect[:int(self.STA['STA_TIME']/self.INTSTEP),1]
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
			
			name = str(self.data['name'])
			self.Projection = 	{
								'priorproject': Priorproj,
								'STAproject': STAproj,
								'Mode1Pr': Mode1Pr,
								'Mode2Pr': Mode2Pr,
								'Mode1proj': Mode1proj,
								'Mode2proj': Mode2proj,
								'name': name
								}
			if SaveFiles.lower() == 'yes':			
				np.savez('Data/' + name + 'Project.npz',
						priorproject=Priorproj,
						STAproject=STAproj,
						Mode1Pr=Mode1Pr,
						Mode2Pr=Mode2Pr,
						Mode1proj=Mode1proj,
						Mode2proj=Mode2proj,
						name=name)
						

		
		
	def BayesPspike(self, BIN_SIZE = 0.5, SaveFiles = 'no'):
			## Pspike histograms to set up LN model:
		try:
			self.Bayes = np.load('Data/' + str(self.data['name']) + 'BayesPspike.npz')
			print 'Successfully loaded {0} BayesPspike data file'.format(self.data['name'])
			LNmodel.FindP_spike(self)
			print '{0} spikes/sec'.format(self.Pspike)
		except IOError:
			print 'No {0} Pspike file, now generating new BayesPspike files'.format(self.data['name'])
		if self.Bayes == None:

			Priorproject = self.Projection['priorproject']
			STAproject = self.Projection['STAproject']
			Mode1project = self.Projection['Mode1proj']
			Mode2project = self.Projection['Mode2proj']
			Mode1Noise = self.Projection['Mode1Pr']
			Mode2Noise = self.Projection['Mode2Pr']
			
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

			
			LNmodel.FindP_spike(self)
			print '{0} spikes/second'.format(self.Pspike)
			
			## Bayes Theorem:
			Pspike_STA = STA_Hist * self.Pspike / Prior_Hist
			Pspike_2d = M12_Spike * self.Pspike / M12_Prior
			
			self.Bayes = 	{
								'Pspike_STA': Pspike_STA,
								'Pspike_2d': Pspike_2d,
								'BINS': BINS
								}
			if SaveFiles.lower() == 'yes':
				np.savez('Data/' + str(self.data['name']) + 'BayesPspike.npz',
						Pspike_STA=Pspike_STA,
						Pspike_2d=Pspike_2d,
						BINS = BINS)
		

		
	def FindProbOfSpike(self, SaveFiles = 'no'):
			## LNmodel final set up
		try:
			self.ProbOfSpike = np.load('Data/' + str(self.data['name']) + 'ProbOfSpike.npz')
			print 'Successfully loaded {0} Pspike data file'.format(self.data['name'])
		except IOError:
			print 'No {0} LN file, now generating new Pspike files'.format(self.data['name'])
		if self.ProbOfSpike == None:
		
			## LOAD EigenModes ##
			Current = self.data['RepStim']
			Voltage = self.data['RepVolt']
			sta = self.STA['STAstim']
			sta = sta[0:self.STA['STA_TIME']/self.INTSTEP]
			sta = sta/lin.norm(sta)
			
			EigVect = self.Cov['eigvect']
			Mode1 = EigVect[:,0]
			Mode2 = EigVect[:,1]
			del(EigVect)
			
			END = 1e5
			START = 1e4
			BINS = self.Bayes['BINS']
			Pspike_STA = self.Bayes['Pspike_STA']
			Pspike_2d = self.Bayes['Pspike_2d']
			### FIND P(spike|s0,s1,s2) ####
			STA_LEN = self.STA['STA_TIME']/self.INTSTEP
			Ps_STA = np.zeros((END-START+1, Current.shape[1]))
			Ps_2d = np.zeros((END-START+1, Current.shape[1]))
			S1 = np.zeros((END-START+1))
			S2 = np.zeros((END-START+1))
			
			for j in range(0, Current.shape[1]):
				for i in range(int(START),int(END)):
					
					S0 = round((float(np.dot(sta,Current[i-STA_LEN:i,j]))/0.25)*0.25)
					loc = np.where(BINS==[S0])
					Ps_STA[i-START,j] = Pspike_STA[loc]
					
					S1[i-START] = round((float(np.dot(Mode1,Current[i-STA_LEN:i,j]))/0.25)*0.25)
					S2[i-START] = round((float(np.dot(Mode2,Current[i-STA_LEN:i,j]))/0.25)*0.25)
					loc1 = np.where(BINS==[S1[i-START]])
					loc2 = np.where(BINS==[S2[i-START]])
					Ps_2d[i-START,j] = Pspike_2d[loc1,loc2]

			"""
			### FIND SPIKES IN DATA ###
			Spikes = np.zeros((Voltage.shape[0],Voltage.shape[1]))
			for i in range(0,Voltage.shape[1]):
				foo = Voltage[:,i] > 0
				Spikes[:,i],a = runs(foo)
			"""
			self.ProbOfSpike =	{
									'Ps_2d': Ps_2d,
									'Ps_STA': Ps_STA
									}
			if SaveFiles.lower() == 'yes':
				np.savez('Data/' + str(self.data['name']) + 'ProbOfSpike.npz',
						Ps_2d=Ps_2d,
						Ps_STA=Ps_STA)


	def FindP_spike(self):
		"""
		## in Hz. Spikes/len(one row)*dt*20(rows)/1000(msec/sec)
		"""
		
		self.Pspike = self.STA['spikenum'] / (self.data['RawVolt'].shape[0]*
						self.data['RawVolt'].shape[1] * self.INTSTEP / 1000.0) 





## Data Preprocessing ##
########################
class DataProcessing(LNmodel,Database.Database):

	
	def GetData(self, DIRECTORY = '/120511c3.mat', SaveName = 'Rebecca'):
		
		try:
			self.Files = Database.Database()
			self.Files.OpenDatabase(SaveName + '.h5')
			#= np.load('Data/' + SaveName + 'ProcessedData.npz')
			self.NAME = np.array([SaveName])
			
			SamplingRate = self.Files.QueryDatabase('DataProcessing','SamplingRate')
			print 'Successfully opened {0} database.'.format(SaveName)
			self.data = 'Full'
			
		except :
			
			self.Files.CreateGroup('DataProcessing')
			print 'No preprocessed data. Now trying raw data.'

		if self.data == None:
			self.NAME = np.array([SaveName])
		
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

	
	def GenModelData(self, Data, model = models.QUADmodel, params = 'EvolvedParam1_8.csv',
					SaveName = 'Quad'):
		"""
		"""
		try:
			self.Files = Database.Database()
			self.Files.OpenDatabase(SaveName + '.h5')
			self.NAME = np.array([SaveName])
			
			SamplingRate = self.Files.QueryDatabase('DataProcessing','SamplingRate')
			print 'Successfully opened {0} database.'.format(SaveName)
		except :
			
			self.Files.CreateGroup('DataProcessing')
			print 'No preprocessed data. Now trying raw data.'

		if self.data == None:
			self.INTSTEP = Data.data['SamplingRate']**-1.0 * 1000.0
			
			params = np.genfromtxt(params,delimiter=',')
			
			current = Data.data['RawStim'] * 100.0
			ModelVolt = np.zeros((current.shape[0],current.shape[1]))
			for i in range(0, current.shape[1]):
				
				ModelVolt[:,i] = model(params, current[:,i], self.INTSTEP)
			
			RepModelVolt = np.zeros((Data.data['RepLoc'].shape[1],
									Data.data['RawVolt'].shape[1]))
			RepStim = Data.data['RawStim']
			for i in range ( 0 , Data.data['RawStim'].shape[1]):
				RepModelVolt[:,i] = ModelVolt[Data.data['RepLoc'] , i]

			INT = self.INTSTEP
			
			
			self.Files.AddData2Database('SamplingRate', SamplingRate, 'DataProcessing')
			self.Files.AddData2Database('INTSTEP', INT, 'DataProcessing')
			self.Files.AddData2Database('RawVolt', ModelVolt, 'DataProcessing')
			self.Files.AddData2Database('RawStim', current, 'DataProcessing')
			self.Files.AddData2Database('RepLoc', Data.data['RepLoc'], 'DataProcessing')
			self.Files.AddData2Database('RepVolt', RepModelVolt, 'DataProcessing')
			self.Files.AddData2Database('RepStim', RepStim, 'DataProcessing')
			self.Files.AddData2Database('ModelParams', params, 'DataProcessing')
			self.Files.AddData2Database('name', np.array([SaveName]), 'DataProcessing')

					
			self.Files.CloseDatabase()		

		
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


	

