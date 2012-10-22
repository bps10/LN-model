import numpy as np

## Models ##
"""
class NeuronModel(LNmodel, Database.Database):

	
	def GenModelData(self, SaveName, DataName,  params = 'EvolvedParam1_8.csv', model = models.QUADmodel):
		
		
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
"""


def EXPmodel(param, CURRENT):
    
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



def QUADmodel(param, CURRENT, INTSTEP):
    
    u1DOT = 0
    u2DOT = 0
    vDOT = 0
    CAP = 50.0 ## in pF
    vDOT = -70 ## V naught in mV
    vRESTING = -70
    VOLTAGE = np.zeros(len(CURRENT))
    b_1 = param[7] # if vDOT < param[5] else 0
    b_2 = -param[5]    
    a_1 = param[0]
    a_2 = param[1]
    d_1 = param[2]
    d_2 = param[3]
    vRESET = param[8]
    vTHR = param[4]

    for i in range(0,len(CURRENT)):
    
        k_ = param[6] if vDOT < param[4] else 2
        
        vDOT += (k_*(vDOT - vRESTING)*(vDOT - vTHR) - u1DOT - u2DOT +
                 CURRENT[i])*INTSTEP/CAP
        u1DOT += (a_1*(b_1*(vDOT - vRESTING) - u1DOT))*INTSTEP
        u2DOT += (a_2*(b_2*(vDOT - vRESTING) - u2DOT))*INTSTEP
    
        if vDOT >= 35 + (0.1*(u1DOT+u2DOT)):
            vDOT = vRESET - (0.1*(u1DOT+u2DOT))
            u1DOT = u1DOT + d_1
            u2DOT = u2DOT + d_2

        VOLTAGE[i] = vDOT
    return VOLTAGE
