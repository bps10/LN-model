import tables as tables
import numpy as np
	

class Database:

	def __init__(self):
		self = None

	def CreateGroup(self, GroupName):
		
		self.group = self.file.createGroup("/", GroupName,'Neuron')
		
	def CreateDatabase(self, FileName):
	
		self.file = tables.openFile(FileName, mode = "w", title = "NeuronData")

	def AddData2Database(self, DataName, Data, GroupName):
		
		loc = 'self.file.root.' + GroupName
		
		atom = tables.Atom.from_dtype(Data.dtype)
		ds = self.file.createCArray(eval(loc), DataName, atom, Data.shape)
		ds[:] = Data
		
	def OpenDatabase(self, DatabaseName):
	
		self.file = tables.openFile(DatabaseName, mode = "a")
		
	def QueryDatabase(self, GroupName, DataName):
	
		loc = 'self.file.root.' + GroupName + '.' + DataName + '.read()'
		return eval(loc)
	
	def AddMetaData(self):

		fnode.attrs.content_type = 'text/plain; charset=us-ascii'
		
	def CloseDatabase(self):
		
		self.file.close()
		

		

	