import tables as tables
import numpy as np
	
## To Do:

## 1. Add a class that records the analyses.
## 2. AddMetaData()

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
		print '{0} database opened'.format(DatabaseName)
		
	def QueryDatabase(self, GroupName, DataName):
	
		loc = 'self.file.root.' + GroupName + '.' + DataName + '.read()'
		return eval(loc)

	def RemoveGroup(self, GroupName):
		
		decision = raw_input('Are you absolutely sure you want to delete this group permenently? (y/n): ')

		if decision.lower() == 'y' or decision.lower() == 'yes':
			eval('self.file.root.' + GroupName + '._f_remove()')
			print '{0} successfully deleted'.format(GroupName)
		else:
			print 'Ok, nothing changed.'
				
	def AddMetaData(self):

		fnode.attrs.content_type = 'text/plain; charset=us-ascii'
		
	def CloseDatabase(self):
		
		self.file.close()
		print 'database closed'

		

