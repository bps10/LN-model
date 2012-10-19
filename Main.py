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

	## Preprocess data:
	Data = Neuron.DataProcessing()
	Data.GetData('/120511c3.mat', DataName)
	
	Data.STA_Analysis(180, SaveFiles = 'yes')
	# Add STA plot call here.
	
	Data.FindCovariance(SaveFiles = 'yes')
	# Add Cov plot call here.
	
	Data.FindProjection(SaveFiles = 'yes')
	# Add Projection plot call here.
	
	Data.BayesPspike(SaveFiles = 'yes')
	
	Data.FindProbOfSpike(SaveFiles = 'yes')
	# Add LNmodel plot call here.
	
	#####################################
	
	print ' '
	print 'Now the models turn'
	print ' '
	
	#####################################

	## Preprocess Model:
	Model = Neuron.NeuronModel()
	Model.GenModelData(Data, model = models.QUADmodel, params = 'EvolvedParam1_8.csv',
						SaveName = ModelName)
	
	Model.STA_Analysis(180, SaveFiles = 'yes')
	# Add STA plot call here.
	
	Model.FindCovariance(SaveFiles = 'yes')
	# Add Cov plot call here.
	
	Model.FindProjection(SaveFiles = 'yes')
	# Add Projection plot call here.
	
	Model.BayesPspike(SaveFiles = 'yes')
	
	Model.FindProbOfSpike(SaveFiles = 'yes')
	# Add LNmodel plot call here.


	
if __name__ == '__main__':
	
	## Get some user options first to pass into main.
	main(DataName='Rebecca', ModelName='Quad')

	
	
	
		