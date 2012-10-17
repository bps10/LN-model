from __future__ import division

import numpy as np
import numpy.linalg as lin
import scipy as sp
import matplotlib.pylab as plt

import NeuronModels as models
import NeuronFuncRepo as Neuron

#import LNplotting as LNplot
    
## 0. Figure out why model not spiking.
## 1. Covariance from NEW analysis.
## 2. Projections.
## 3. Putting it all together, plotting.
		
if __name__ == '__main__':

	RebData = Neuron.GetData()
	RebSTA = Neuron.STA_Analysis(RebData)
	
	ModelData = Neuron.GenModelData(RebData, models.QUADmodel, params = 'EvolvedParam1_8.csv')
	ModelSTA = Neuron.STA_Analysis(ModelData)