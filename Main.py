from __future__ import division
import numpy as np
import matplotlib.pylab as plt

import NeuronModels as models
import NeuronFuncRepo as Neuron
import LNplotting as LNplot

## To Do:
## 1. Putting it all together, plotting.
## 2. User options: where to find data, type of neuron model.
## 3. Classes, subclasses.
## 4. Help files, documentation.


def main(DataName,ModelName):
	Pspike = 8302/((1055000*0.1)*20/1000) ## in Hz. Spikes/len(one row)*dt*20(rows)/1000(msec/sec)

	## Preprocess data:
	Data = Neuron.GetData(SaveName=DataName)
	
	## STA:
	try: 
		STA = np.load('Data/' + DataName + 'STA.npz')
		print 'Successfully loaded {0} data STA file'.format(Data['name'])
	except IOError:
		STA = []
		print 'No {0} STA file, now generating new STA files'.format(Data['name'])
	
	if STA == []:
		STA = Neuron.STA_Analysis(Data)
	
	## Covariance Analysis:
	try:
		Cov = np.load('Data/RebeccaCovariance.npz')
		print 'Successfully loaded {0} data Cov file'.format(Data['name'])
	except IOError:
		Cov = []
		print 'No {0} Cov file, now generating new cov files'.format(Data['name'])
	
	if Cov == []:
		Cov = Neuron.FindCovariance(STA)
		
	## Projection:
	try: 
		Project = np.load('Data/' + DataName + 'Project.npz')
		print 'Successfully loaded {0} data Projection file'.format(Data['name'])
	except IOError:
		Project = []
		print 'No {0} data Projection file, now generating new projection files'.format(Data['name'])
	if Project == []:
		Project = Neuron.FindProjection(Data,STA,Cov,180)

	## Pspike histograms to set up LN model:
	try:
		Hist = np.load('Data/' + DataName + 'PspikeData.npz')
		print 'Successfully loaded {0} Pspike data file'.format(Data['name'])
	except IOError:
		Hist = []
		print 'No {0} Pspike file, now generating new Pspike files'.format(Data['name'])
	if Hist == []:
		Hist = Neuron.PspikeHist(Project, Pspike)
	
	## LNmodel final set up
	try:
		LN = np.load('Data/' + DataName + 'LNmodel.npz')
		print 'Successfully loaded {0} Pspike data file'.format(Data['name'])
	except IOError:
		LN = []
		print 'No {0} LN file, now generating new Pspike files'.format(Data['name'])
	if LN == []
		LN = Neuron.LNmodel(Data,Cov)
	
	
	#####################################
	
	print ' '
	print 'Now the models turn'
	print ' '
	
	#####################################

	
	## Preprocess data:
	ModelData = Neuron.GenModelData(Data, models.QUADmodel, params = 'EvolvedParam1_8.csv',
									SaveName=ModelName)
	
	## STA:
	try :
		ModelSTA = np.load('Data/' + ModelName + 'STA.npz')
		print 'Successfully loaded {0} STA file'.format(ModelData['name'])
	except IOError:
		ModelSTA = []
		print 'No {0} model STA data file, now generating new STA files'.format(ModelData['name'])
	if ModelSTA == []:
		ModelSTA = Neuron.STA_Analysis(ModelData)
		
	## Covariance Analysis:
	try:
		ModelCov = np.load('Data/' + ModelName + 'Covariance.npz')
		print 'Successfully loaded {0} data STA file'.format(ModelData['name'])
	except IOError:
		ModelCov = []
		print 'No {0} data Cov file, now generating new cov files'.format(ModelData['name'])
	
	if ModelCov == []:
		ModelCov = Neuron.FindCovariance(ModelSTA)	

	
	## Projection:
	try: 
		ModelProject = np.load('Data/' + ModelName + 'Project.npz')
		print 'Successfully loaded {0} data Projection file'.format(ModelData['name'])
	except IOError:
		ModelProject = []
		print 'No {0} data Project file, now generating new project files'.format(ModelData['name'])
	if ModelProject == []:
		ModelProject = Neuron.FindProjection(ModelData,ModelSTA,ModelCov,180)
	
	
	## Pspike histograms to set up LN model:
	try:
		ModelHist = np.load('Data/' + ModelName + 'PspikeData.npz')
		print 'Successfully loaded {0} Pspike data file'.format(Data['name'])
	except IOError:
		ModelHist = []
		print 'No {0} Pspike file, now generating new Pspike files'.format(ModelData['name'])
	if ModelHist == []:
		ModelHist = Neuron.PspikeHist(ModelProject, Pspike)
	
	## LNmodel final set up
	try:
		ModelLN = np.load('Data/' + ModelName + 'LNmodel.npz')
		print 'Successfully loaded {0} LN data file'.format(ModelData['name'])
	except IOError:
		ModelLN = []
		print 'No {0} LN file, now generating new LN files'.format(ModelData['name'])
	if ModelLN == []
		ModelLN = Neuron.LNmodel(ModelData,ModelCov)
		
	
	############################
	## LN model plotting
	############################
	
	# Plot eigen vectors, etc.
	# LNplot.PlotHistOutline()
	# plot psth

	
if __name__ == '__main__':
	
	## Get some user options first to pass into main.
	main(DataName='Rebecca', ModelName='Quad')

	
	
	
		