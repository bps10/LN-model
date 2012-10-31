# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 15:34:23 2012

@author: Brian
"""

from distutils.core import setup
import py2exe # Patching distutils setup


# Including/excluding DLLs and Python modules
#DATA_FILES = [('NeuronData', ['c:/Users/Brian/Documents/LN-Model/NeuronData.h5'])]


setup(
      options={
               "py2exe": {"compressed": 2, "optimize": 2, 'bundle_files': 1,
                          "includes": ["sip", "PyQt4", "guidata", "guiqwt", "numpy",
                                       "Database", "sys"],
                          "dist_dir": "dist",},
               },
      #data_files=DATA_FILES,
      windows=[{
                "script": "guiMain.py",
                "dest_base": "simpledialog",
                "version": "1.0.0",
                "company_name": u"CEA",
                "copyright": u"Copyright © 2012",
                "name": "Database",
                "description": "Simple GUI",
                },],
      zipfile = None,
      )