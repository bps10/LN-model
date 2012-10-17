from __future__ import division

import numpy as np
import numpy.linalg as lin
import scipy as sp
import matplotlib.pylab as plt

import NeuronModels as models
import NeuronFuncRepo as Neuron

#import LNplotting as LNplot
    
## 1. Covariance from NEW analysis.
## 2. Projections.
## 3. Putting it all together, plotting.
		
if __name__ == '__main__':

	RebData = Neuron.GetData()
	
	try: 
		RebSTA = np.load('/Data/RebSTA.npz')
	except IOError:
		RebSTA = []
		print 'No data STA file, now generating new'
	
	if RebSTA == []:
		RebSTA = Neuron.STA_Analysis(RebData)
	
	ModelData = Neuron.GenModelData(RebData, models.QUADmodel, params = 'EvolvedParam1_8.csv')
	ModelSTA = Neuron.STA_Analysis(ModelData)