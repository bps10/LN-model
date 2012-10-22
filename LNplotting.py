from __future__ import division
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pylab as plt
import PlottingFun as pf

class LNplotting():

	def Hist(self, BIN_SIZE, option = 1):
		try:
			self.Files.OpenDatabase(self.NAME + '.h5')
			Ps_STA = self.Files.QueryDatabase('ProbOfSpike', 'Ps_STA')
			Ps_2d = self.Files.QueryDatabase('ProbOfSpike', 'Ps_2d')
			INTSTEP = self.Files.QueryDatabase('DataProcessing', 'INTSTEP')[0]
			
			
			if option == 1:
				Spikes = self.Files.QueryDatabase('Spikes', 'RepSpikes')
			elif option == 0:
				Spikes = self.Files.QueryDatabase('Spikes', 'Spikes')
				
			self.Files.CloseDatabase()	
		except:
			print 'Sorry error. Either no Bayes or Spikes files found.'
			
		BINS = np.linspace(0, Spikes.shape[0], Spikes.shape[0] / BIN_SIZE * INTSTEP)
		Data_Hist = np.zeros((len( BINS ), Spikes.shape[1]) )
		STA_Hist = np.zeros((len( BINS ), Spikes.shape[1]) )
		Model2d_Hist = np.zeros((len( BINS ), Spikes.shape[1]) )

		for j in range(0, Spikes.shape[1]):
			for i in range(0,len( BINS ) - 1):
				Start = BINS[i]
				End = BINS[i+1]
				Data_Total = np.sum( Spikes[Start:End, j] )
				STA_Total = np.mean( Ps_STA[Start:End, j] )
				Model2d_Total = np.mean( Ps_2d[Start:End, j] )
			
				Data_Hist[i,j] = Data_Total
				STA_Hist[i,j] = STA_Total
				Model2d_Hist[i,j] = Model2d_Total
		
		Data_Hist =Data_Hist.mean(1)
		STA_Hist = STA_Hist.mean(1)
		Model2d_Hist = Model2d_Hist.mean(1)
		
		Data_Hist = Data_Hist / BIN_SIZE * 1000.0
		
		return Data_Hist, STA_Hist, Model2d_Hist, BINS


	def histOutline(self, histIn, binsIn):
		"""
		"""

		
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
		
		return bins, data



	def PlotHistOutline(self, HIST_BIN_SIZE = 10):
		try:
			self.Files.OpenDatabase(self.NAME + '.h5')
			INTSTEP = self.Files.QueryDatabase('DataProcessing', 'INTSTEP')[0]
			Ps_2d = self.Files.QueryDatabase('ProbOfSpike', 'Ps_2d')
			Current = self.Files.QueryDatabase('DataProcessing', 'RepStim')
			Current = Current[:,0] # each rep is identical.
		except :
			print 'Sorry no data'
		## PLOT HISTOGRAM OUTLINES ##
		START = 1000
		END = 10000
		Data_Hist,STA_Hist,Model_Hist,BINS = self.Hist(HIST_BIN_SIZE)
		
		bins,Outline_Data_Hist = self.histOutline(Data_Hist,BINS)
		bins,Outline_STA_Hist = self.histOutline(STA_Hist,BINS)
		bins,Outline_Model_Hist = self.histOutline(Model_Hist,BINS)

		
		fig = plt.figure(figsize=(12,8))
		ax1 = fig.add_subplot(411)
		ax1.axes.get_xaxis().set_ticks([])
		ax1.plot(bins, Outline_STA_Hist, linewidth=2, color='k')
		plt.ylim([-10,125])
		ax1.axis('off')

		ax2 = fig.add_subplot(412)
		ax2.axes.get_xaxis().set_ticks([])
		ax2.plot(np.ones(50)*(max(bins)-10),np.arange(30,80), linewidth=6 , color='k')
		ax2.plot(bins, Outline_Model_Hist, linewidth=2, color='k')
		plt.ylim([-10,125])
		ax2.axis('off')

		ax3 = fig.add_subplot(413)
		ax3.axes.get_xaxis().set_ticks([])
		ax3.plot(np.ones(50)*(max(bins)-10),np.arange(30,80), linewidth=6 , color='k')
		ax3.plot( np.arange(0, (100/INTSTEP)) ,np.ones(100/INTSTEP)*-24, linewidth=6, color='k')
		ax3.plot(bins, Outline_Data_Hist, linewidth=2, color='k')
		plt.ylim([-25,225])
		ax3.axis('off')

		ax4 = fig.add_subplot(414)
		ax4.plot(np.arange(0,len(Current))*INTSTEP, Current, linewidth=2, color='k')
		ax4.axis('off')
		plt.tight_layout()
		plt.show()

		
	def PSTH(self, Data, STA, Model2d):
	
		try:
			self.Files.OpenDatabase(self.NAME + '.h5')
			Ps_STA = self.Files.QueryDatabase('ProbOfSpike', 'Ps_STA')
			Ps_2d = self.Files.QueryDatabase('ProbOfSpike', 'Ps_2d')
			INTSTEP = self.Files.QueryDatabase('DataProcessing', 'INTSTEP')[0]
			self.Files.CloseDatabase()
			print 'All data found. Computing.'
				
		except:
			print 'Sorry error. Either no Bayes or Spikes files found.'
			
		TimeRes = np.array([0.1,0.25,0.5,1,2.5,5.0,10.0,25.0,50.0,100.0])

		Projection_PSTH = np.zeros((2,len(TimeRes)))
		for i in range(0,len(TimeRes)):
			Data_Hist,STA_Hist,Model_Hist,B = Hist(TimeRes[i], Data, STA, Model2d)
			data = Data_Hist/np.linalg.norm(Data_Hist)
			sta = STA_Hist/np.linalg.norm(STA_Hist)
			model = Model_Hist/np.linalg.norm(Model_Hist)
			Projection_PSTH[0,i] = np.dot(data,sta)
			Projection_PSTH[1,i] = np.dot(data,model)
			
		import matplotlib.font_manager as fm
		
		plt.figure()
		plt.semilogx(TimeRes,Projection_PSTH[0,:],'gray',TimeRes,Projection_PSTH[1,:],'k',
			     linewidth=3, marker='o', markersize = 12)
		plt.xlabel('Time Resolution, ms',fontsize=25)
		plt.xticks(fontsize=25)
		#plt.axis["right"].set_visible(False)
		plt.ylabel('Projection onto PSTH',fontsize=25)
		plt.yticks(fontsize=25)
		prop = fm.FontProperties(size=20)
		plt.legend(('1D model','2D model'),loc='upper left',prop=prop)
		plt.tight_layout()
		plt.show()

		
	def STAplot(self, option = 0):
		try:
			self.Files.OpenDatabase(self.NAME + '.h5')
			STA_TIME = self.Files.QueryDatabase('STA_Analysis', 'STA_TIME')[0]
			STA_Current = self.Files.QueryDatabase('STA_Analysis', 'STAstim')
			INTSTEP = self.Files.QueryDatabase('DataProcessing', 'INTSTEP')[0][0]
		except:
			print 'Sorry no data found'
		
		X = np.arange(-STA_TIME / INTSTEP, STA_TIME / INTSTEP, dtype=float) * INTSTEP
		
		if option == 1:
			fig = plt.figure()
			ax = fig.add_subplot(111)
			ax.plot(X[0:(STA_TIME/INTSTEP)],STA_Current[0:(STA_TIME/INTSTEP)], 
						linewidth=3, color='k')
			ax.plot(np.arange(-190,-170),np.ones(20)*0.35, linewidth=5,color='k')
			ax.plot(np.ones(200)*-170,np.arange(0.35,0.549,0.001),linewidth=5,color='k')
			ax.plot(np.arange(-200,0),np.zeros(200), 'k--', linewidth=2)
			plt.axis('off')
			plt.show()
		
		if option == 0:
			fig = plt.figure(figsize=(12,8))
			ax = fig.add_subplot(111)
			ax.plot(X[0:(STA_TIME / INTSTEP) + 50], STA_Current[0:(STA_TIME / INTSTEP) + 50],
						linewidth=3, color='k')
			plt.xticks(fontsize = 20)
			plt.yticks(fontsize = 20)
			plt.ylabel('current(pA)', fontsize = 20)
			plt.legend(('data'), loc='upper right')
			plt.show()

