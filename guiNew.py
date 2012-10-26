# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 13:37:56 2012

@author: Brian
"""

from guidata.qt.QtGui import QMainWindow, QSplitter
from guidata.qt.QtCore import SIGNAL

from guidata.dataset.datatypes import (DataSet, BeginGroup, EndGroup,
                                       BeginTabGroup, EndTabGroup)
from guidata.dataset.dataitems import (ChoiceItem, FloatItem, StringItem,
                                       DirectoryItem, FileOpenItem)
from guidata.dataset.qtwidgets import DataSetShowGroupBox, DataSetEditGroupBox
from guidata.configtools import get_icon
from guidata.qthelpers import create_action, add_actions, get_std_icon
import guidata.dataset.dataitems as di
import guidata.dataset.datatypes as dt
import numpy as np
import Database as Db

"""
import guiqwt.plot as plt
from guiqwt.builder import make

def plot( *items ):
    win = plt.CurveDialog(edit=False, toolbar=True)
    plot = win.get_plot()
    for item in items:
        plot.add_item(item)
    win.show()
    win.exec_()
"""
        
class AnotherDataSet(DataSet):
    u"""
    Example 2
    <b>Simple dataset example</b>
    """
    param0 = di.ChoiceItem(u"Choice", ['deazdazk', 'aeazee', '87575757'])
    param1 = di.FloatItem(u"Foobar 1", default=0, min=0)
    a_group = dt.BeginGroup("A group")
    param2 = di.FloatItem(u"Foobar 2", default=.93)
    param3 = di.FloatItem(u"Foobar 3", default=123)
    _a_group = dt.EndGroup("A group")


class ArrayDataset(DataSet):
    u"""
    Example 3
    """

    Dat = Db.Database()
    Dat.OpenDatabase('NeuronData')
    Name = None
    if Name == None:
        Name = di.StringItem("Data", 'monitorData')
    
        Query = Dat.QueryDatabase('Oct0212Bc8', 'epoch000', 'monitorData')
        floatarray = di.FloatArrayItem("Float array", default = Query,
                                format=" %.2e ")
    else:

        Query = Dat.QueryDatabase('Oct0212Bc8', 'epoch000', 'monitorData')
        floatarray.accept(Query)
        print 'here'
                                

class ExampleMultiGroupDataSet(DataSet):
    ## Choose neuron to look at:  Call dataset
    # call database
    param0 = di.ChoiceItem(u"Choose Neuron", ['deazdazk', 'aeazee', '87575757'])
    param1 = di.FloatItem(u"Foobar 1", default=0, min=0)
    t_group = BeginTabGroup("T group")
    a_group = BeginGroup("A group")
    param2 = di.FloatItem(u"Foobar 2", default=.93)
    dir1 = DirectoryItem(u"Directory 1")
    file1 = FileOpenItem(u"File 1")
    _a_group = EndGroup("A group")
    b_group = BeginGroup("B group")
    param3 = di.FloatItem(u"Foobar 3", default=123)
    _b_group = EndGroup("B group")
    c_group = BeginGroup("C group")
    param4 = di.FloatItem(u"Foobar 4", default=250)
    _c_group = EndGroup("C group")
    _t_group = EndTabGroup("T group")
    
class OtherDataSet(DataSet):
    Name = di.StringItem("Title", default="Title")
    icon = di.ChoiceItem("Icon", (("python.png", "Python"),
                               ("guidata.svg", "guidata"),
                               ("settings.png", "Settings")))
    opacity = di.FloatItem("Opacity", default=1., min=.1, max=1)

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowIcon(get_icon('python.png'))
        self.setWindowTitle("Application example")
        
        # Instantiate dataset-related widgets:
        self.groupbox1 = DataSetEditGroupBox("Dataset", ArrayDataset,
                                             comment='')
        self.groupbox2 = DataSetShowGroupBox("Standard dataset",
                                             AnotherDataSet, comment='')
        self.groupbox3 = DataSetEditGroupBox("Standard dataset",
                                             OtherDataSet, comment='')
        self.groupbox4 = DataSetShowGroupBox("Standard dataset",
                                             ExampleMultiGroupDataSet, comment='')
        self.connect(self.groupbox1, SIGNAL("apply_button_clicked()"),
                     self.update_window)
        self.update_groupboxes()
        
        splitter = QSplitter(self)
        splitter.addWidget(self.groupbox1)
        splitter.addWidget(self.groupbox2)
        splitter.addWidget(self.groupbox3)
        splitter.addWidget(self.groupbox4)
        self.setCentralWidget(splitter)
        self.setContentsMargins(10, 5, 10, 5)
        
        # File menu
        file_menu = self.menuBar().addMenu("File")
        quit_action = create_action(self, "Quit",
                                    shortcut="Ctrl+Q",
                                    icon=get_std_icon("DialogCloseButton"),
                                    tip="Quit application",
                                    triggered=self.close)
        add_actions(file_menu, (quit_action, ))
        
        # Edit menu
        edit_menu = self.menuBar().addMenu("Edit")
        
        #editparam1_action = create_action(self, "Edit dataset1", 
                                          #triggered=self.edit_dataset1)
        editparam2_action = create_action(self, "Edit dataset 2",
                                          triggered=self.edit_dataset2)
        editparam4_action = create_action(self, "Edit dataset 4",
                                          triggered=self.edit_dataset4)
        add_actions(edit_menu, (editparam2_action,
                                editparam4_action))

    def update_window(self):
        self.groupbox1.dataset.edit()
        
        #self.groupbox1.dataset.set_writeable()

        self.groupbox1.get()
        
    def update_groupboxes(self):
 # This is an activable dataset
        self.groupbox1.get()
        self.groupbox2.get()
        self.groupbox4.get()
        
    def edit_dataset1(self):
        #self.groupbox1.dataset.set_writeable() # This is an activable dataset
        if self.groupbox1.dataset.edit():
            self.update_groupboxes()

    def edit_dataset2(self):
        if self.groupbox2.dataset.edit():
            self.update_groupboxes()

    def edit_dataset4(self):
        if self.groupbox4.dataset.edit():
            self.update_groupboxes()
        
if __name__ == '__main__':
    from guidata.qt.QtGui import QApplication
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    