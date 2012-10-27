# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 13:30:48 2012

@author: Brian
"""
from guidata.dataset.qtwidgets import DataSetShowGroupBox
import Database as Db
from guidata.qt.QtGui import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QMainWindow, QLineEdit, QListView, QListWidget)
from guidata.qt.QtCore import (SIGNAL, QAbstractListModel, QModelIndex, QVariant, 
                               Qt)
from guidata.dataset.dataitems import StringItem, DirectoryItem
from guidata.dataset.datatypes import DataSet
from guidata.qthelpers import create_action, add_actions, get_std_icon
import numpy as np
import sys

#---Import plot widget base class
from guiqwt.curve import CurvePlot
from guiqwt.plot import PlotManager
from guiqwt.builder import make
from guidata.configtools import get_icon
#---


    
class FilterTestWidget(QWidget):
    """
    Filter testing widget
    parent: parent widget (QWidget)
    x, y: NumPy arrays
    func: function object (the signal filter to be tested)
    """
    def __init__(self, parent, func):
        QWidget.__init__(self, parent)
        self.data = Dbase()
        self.y = self.data.Query()
        self.x = np.arange(0, len(self.y))
        self.setMinimumSize(600, 650)
        #self.x = x
        #self.y = y
        self.func = func
        #---guiqwt related attributes:
        self.plot = None
        self.curve_item = None
        #---
        
    def setup_widget(self, title):
        #---Create the plot widget:
        self.plot = CurvePlot(self)
        self.curve_item = make.curve([], [], color='b')
        self.plot.add_item(self.curve_item)
        self.plot.set_antialiasing(True)
        #---
        
        self.Neuron     = QLineEdit("Neuron name")
        self.Epoch      = QLineEdit("Epoch name")
        self.QueryName  = QLineEdit("Data selection")
        
        self.tree = self.data.GetTree('Oct0212Bc8')        

        #DatabaseList = QListView().setModel(self.tree)
        #DatabaseList.QListView.setViewMode(QListView.ListMode) 
        # create table
        
        list_data = self.tree
        listmodel = MyListModel(list_data, self)
        self.listview = QListView()
        self.listview.setModel(listmodel)

        
         
        listButton = QPushButton(u"Change List")
        button = QPushButton(u"New Query: %s" % title)
        #itemDoubleClicked
        self.connect(listButton, SIGNAL('clicked()'), self.dud)
        self.connect(button, SIGNAL('clicked()'), self.query_database)
        vlayout = QVBoxLayout()
        hlayout = QHBoxLayout()
        vlayout.addWidget(self.listview)
        vlayout.addWidget(listButton)
        hlayout.addWidget(self.Neuron)
        hlayout.addWidget(self.Epoch)
        hlayout.addWidget(self.QueryName)
        vlayout.addLayout(hlayout)
        
        vlayout.addWidget(button)
        vlayout.addWidget(self.plot)
        

        self.setLayout(vlayout)
        
        self.update_curve()
        
    
    def dud(self):
        print self.listview.rootIndex().row()
        
    def query_database(self):
        neuronname = str(self.Neuron.displayText())
        epochname = str(self.Epoch.displayText())
        dataname = str(self.QueryName.displayText())
        self.y = self.data.Query(NeuronName = neuronname, Epoch = epochname, DataName = dataname)
        self.update_curve()

        
    def update_curve(self):
        #---Update curve
        self.curve_item.set_data(self.x, self.y)
        self.plot.replot()
        self.plot.do_autoscale()
        
        
class MyListModel(QAbstractListModel): 
    def __init__(self, datain, parent=None, *args): 
        """ datain: a list where each item is a row
        """
        QAbstractListModel.__init__(self, parent, *args) 
        self.listdata = datain
 
    def rowCount(self, parent=QModelIndex()): 
        return len(self.listdata) 
 
    def data(self, index, role): 
        if index.isValid() and role == Qt.DisplayRole:
            return QVariant(self.listdata[index.row()])
        else: 
            return QVariant()
            
class Dbase():
    def __init__(self):
        
        self.Data = Db.Database()
        self.Data.OpenDatabase('NeuronData')
        
    def Query(self, NeuronName = 'Oct0212Bc8', Epoch = 'epoch040', DataName = 'rawData'):
        return self.Data.QueryDatabase( NeuronName, Epoch, DataName)
        
    def AddData(self, NeuronName, Directory):
        
        self.Data.ImportAllData(NeuronName, Directory)        
        
    def GetTree(self, NeuronName):
        return self.Data.GetChildList(NeuronName)
        

class FindFile(DataSet):

    Directory = DirectoryItem("Directory")
    NeuronName = StringItem("NeuronName")
    
    
class TestWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle("Neuron Database")
        self.setWindowIcon(get_icon('guiqwt.png'))
        
        file_menu = self.menuBar().addMenu("File")
        quit_action = create_action(self, "Quit",
                                    shortcut="Ctrl+Q",
                                    icon=get_std_icon("DialogCloseButton"),
                                    tip="Quit application",
                                    triggered=self.close)
        add_actions(file_menu, (quit_action, ))
        
        # Edit menu
        self.importData = DataSetShowGroupBox("Neuron Data",
                                             FindFile, comment='')
        
        edit_menu = self.menuBar().addMenu("Edit")
        editparam1_action = create_action(self, "Add dataset",
                                          triggered=self.add_newData)
        add_actions(edit_menu, (editparam1_action, ))
        
        hlayout = QHBoxLayout()
        central_widget = QWidget(self)
        central_widget.setLayout(hlayout)
        self.setCentralWidget(central_widget)
        #---guiqwt plot manager
        self.manager = PlotManager(self)
        #---

    def add_newData(self):
        if self.importData.dataset.edit():
            self.importData.get()
            print str(self.importData.dataset.NeuronName)
            print str(self.importData.dataset.Directory)
            addData = Dbase()
            addData.AddData(str(self.importData.dataset.NeuronName), 
                            str(self.importData.dataset.Directory))
                            
        
    def add_plot(self, func, title):
        widget = FilterTestWidget(self, func)
        widget.setup_widget(title)
        self.centralWidget().layout().addWidget(widget)
        #---Register plot to manager
        self.manager.add_plot(widget.plot)
        #---
                       
        
    def setup_window(self):
        #---Add toolbar and register manager tools
        toolbar = self.addToolBar("tools")
        self.manager.add_toolbar(toolbar, id(toolbar))
        self.manager.register_all_curve_tools()
        #---
        

def main():
    """Testing this simple Qt/guiqwt example"""
    from guidata.qt.QtGui import QApplication
    import scipy.signal as sps, scipy.ndimage as spi
    
    app = QApplication([])
    win = TestWindow()
    

    win.add_plot(lambda x: spi.gaussian_filter1d(x, 1.), "1")
    win.add_plot(sps.wiener, "2")
    #---Setup window
    win.setup_window()
    #---
    
    win.show()
    sys.exit(app.exec_())
        
        
if __name__ == '__main__':
    main()
    