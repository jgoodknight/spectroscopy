# -*- coding: utf-8 -*-
"""
Created on Fri Jan 04 13:23:27 2013

@author: Joey
"""
import os
import os.path
import hickle
import datetime
import cPickle as pickle
import datetime

      
class experiment(object):
    "object which defines I/O operations for an experiment"
        
    def __directoryName__(self):
        date = str(datetime.date.today())
        directoryName = './data/' + date + "/"
        return directoryName
        
    def __save_util__(self, dirName, titleString):
        directories = dirName.split("/")
        last_dir = ""
        for subdirName in directories:
            if subdirName == ".":
                last_dir = subdirName
                continue
            last_dir = last_dir + "/" + subdirName
            try:
                os.mkdir(last_dir)
            except OSError:
                pass
        self.fileName = last_dir + self.experiment_type_string + titleString + ".pkl"
        i = 0
        if os.path.exists(self.fileName):
            while os.path.exists(self.fileName):
                i = i + 1
                self.fileName =  last_dir +  self.experiment_type_string + titleString + "---" + str(i) + ".pkl"
        fileToBeSaved = open(self.fileName, 'wb')
        pickle.dump(self, fileToBeSaved, protocol=-1)
        print "file saved: ", self.fileName
        fileToBeSaved.close()
        
    def save(self, titleString):
        self.__save_util__(self.__directoryName__(), titleString)
        
    def save_in_cwd(self, titleString):
        self.__save_util__(os.getcwd(), titleString)
        
        
        
    @staticmethod 
    def openSavedExperiment(filename):
        myFile = open(filename, 'rb')
        return pickle.load(myFile)
       
if __name__ == "__main__":
    print "Hi!"
