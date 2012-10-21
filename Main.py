from __future__ import division
import numpy as np
import matplotlib.pylab as plt

import NeuronModels as models
import NeuronFuncRepo as Neuron

#import LNplotting as LNplot

## To Do:

## 1. Putting it all together, plotting.
## 2. User options: where to find data, type of neuron model.
## 3. Help files, documentation.


#def main(DataName,ModelName):

## Preprocess data:
Data = Neuron.DataProcessing()
Data.GetData('120511c3.mat', 'Rebecca')

Data.STA_Analysis(180)
# Add STA plot call here.

Data.FindCovariance()
# Add Cov plot call here.

Data.FindProjection()
# Add Projection plot call here.

Data.BayesPspike()

Data.FindProbOfSpike()
Data.Files.CloseDatabase()
	# Add LNmodel plot call here.
	
	#####################################
	
print ' '
print 'Now the models turn'
print ' '
	
	#####################################

	## Preprocess Model:
Model = Neuron.NeuronModel()
Model.GenModelData(SaveName = 'Quad', DataName= 'Rebecca', model = models.QUADmodel, params = 'EvolvedParam1_8.csv')
	
Model.STA_Analysis(180)
	# Add STA plot call here.

Model.FindCovariance()
	# Add Cov plot call here.
	
Model.FindProjection()
	# Add Projection plot call here.
	
Model.BayesPspike()

Model.FindProbOfSpike()
	# Add LNmodel plot call here.


	
if __name__ == '__main__':
	
	## Get some user options first to pass into main.
	main(DataName='Rebecca', ModelName='Quad')

	
	
	
		
