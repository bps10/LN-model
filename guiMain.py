# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 13:30:48 2012

@author: Brian
"""
from guidata.dataset.qtwidgets import DataSetShowGroupBox, DataSetEditGroupBox
import Database as Db
from guidata.qt.QtGui import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QMainWindow, QLineEdit, QListView )
from guidata.qt.QtCore import SIGNAL
from guidata.dataset.dataitems import StringItem
from guidata.dataset.datatypes import DataSet
import numpy as np
import sys
#---Import plot widget base class
from guiqwt.curve import CurvePlot
from guiqwt.plot import PlotManager
from guiqwt.builder import make
from guidata.configtools import get_icon
#---

class ArrayDataset(DataSet):
    u"""
    Example 3
    """
    Name = StringItem("Data", 'monitorData')


    
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
        self.setMinimumSize(800, 700)
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
        
        self.Neuron     = QLineEdit()
        self.Epoch      = QLineEdit()
        self.QueryName  = QLineEdit()
        
        DatabaseList = QListView()                            
        button = QPushButton(u"New Query: %s" % title)
        self.connect(button, SIGNAL('clicked()'), self.process_data)
        #self.connect(self.querybox, SIGNAL('clicked()'), self.query_database())
        vlayout = QVBoxLayout()
        vlayout.addWidget(DatabaseList)
        vlayout.addWidget(self.Neuron)
        vlayout.addWidget(self.Epoch)
        vlayout.addWidget(self.QueryName)
        vlayout.addWidget(button)
        vlayout.addWidget(self.plot)
        

        self.setLayout(vlayout)
        
        self.update_curve()
        
    
    def query_database(self):
        
        print self.querybox.displayText()      
    
    
    def process_data(self):
        neuronname = str(self.Neuron.displayText())
        epochname = str(self.Epoch.displayText())
        dataname = str(self.QueryName.displayText())
        print dataname
        self.y = self.data.Query(NeuronName = 'OctO212Bc8', Epoch = epochname, DataName = dataname)
        self.update_curve()

        
    def update_curve(self):
        #---Update curve
        self.curve_item.set_data(self.x, self.y)
        self.plot.replot()
        self.plot.do_autoscale()
        

class Dbase():
    def __init__(self):
        
        self.Data = Db.Database()
        self.Data.OpenDatabase('NeuronData')
        
    def Query(self, NeuronName = 'Oct0212Bc8', Epoch = 'epoch040', DataName = 'rawData'):
        return self.Data.QueryDatabase('Oct0212Bc8', Epoch, DataName)
        
    def GetTree(self):
        pass

    
class TestWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle("Neuron Database")
        self.setWindowIcon(get_icon('guiqwt.png'))
        
        hlayout = QHBoxLayout()
        central_widget = QWidget(self)
        central_widget.setLayout(hlayout)
        self.setCentralWidget(central_widget)
        #---guiqwt plot manager
        self.manager = PlotManager(self)
        #---
        
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
    