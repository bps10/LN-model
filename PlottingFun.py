import numpy as np
import matplotlib.pylab as plt
import matplotlib.ticker as tic

## To DO:
## 1. Help files.
## 2. General plot cosmetic function.
##


### Plotting cosmetics ###
##########################

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

# http://stackoverflow.com/questions/925024/how-can-i-remove-the-top-and-right-axis-in-matplotlib
def simpleaxis(ax):
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	
	
## http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html
def TufteAxis(ax,spines):
	for loc, spine in ax.spines.iteritems():
		if loc in spines:
			spine.set_position(('outward',10)) # outward by 10 points
			spine.set_smart_bounds(True)
		else:
			spine.set_color('none') # don't draw spine

	# turn off ticks where there is no spine
	if 'left' in spines:
		ax.yaxis.set_ticks_position('left')
	else:
		# no yaxis ticks
		ax.yaxis.set_ticks([])

	if 'bottom' in spines:
		ax.xaxis.set_ticks_position('bottom')
	else:
		# no xaxis ticks
		ax.xaxis.set_ticks([])

		
def SciNoteAxis(gca_handle,spines):
	"""
	Force scientific notation
	"""
	if 'y' in spines:
	
		gca_handle.yaxis.major.formatter.set_powerlimits((-1, 0))
		#t.ticklabel_format(style='sci', axis='y') 

	if 'x' in spines:
	
		gca_handle.xaxis.major.formatter.set_powerlimits((-1, 0))
		#t.ticklabel_format(style='sci', axis='x') 
		
	

def AxisFormat(FONTSIZE):

	font = {'weight' : 'norm', 'size'  : FONTSIZE}
	legend = {'frameon' : False}
	ticks = {'direction' : 'out', 'major.size' : 10 }
	#axes = {'limits'}
	plt.rc('font', **font)
	plt.rc('legend', **legend)
	plt.rc('xtick', **ticks)
	plt.rc('ytick', **ticks)