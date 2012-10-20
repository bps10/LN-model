import tables as tables
import numpy as np
	

class Database:

	def __init__(self):
		self = None

	def CreateGroup(self, GroupName):
		
		self.group = self.file.createGroup("/", GroupName,'Neuron')
		
	def CreateDatabase(self, FileName):
		self.file = tables.openFile(FileName, mode = "w", title = "NeuronData")


	def AddData2Database(self, DataName, Data):
		
		atom = tables.Atom.from_dtype(Data.dtype)
		ds = self.file.createCArray(self.file.root, DataName, atom, Data.shape)
		ds[:] = Data

	
	# Functions below completely untested and unlikely to work.
		
	def OpenDatabase(self, DatabaseName):
	
		self.file = tables.openFile(DatabaseName, mode = "w")
		
	def QueryDatabase(self, GroupName, DataName):
	
		out = self.file.root.GroupName.DataName
		
		return out
		
	def CloseDatabase(self):
		
		self.file.close()