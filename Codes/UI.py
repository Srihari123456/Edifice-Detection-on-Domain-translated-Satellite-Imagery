############################################### MAIN OPENING WINDOW BEGINS

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication,QWidget,QFileDialog 
from GAN import GAN
import sys
global g_model,d_model,gan_model,gan,map_path,gmodel_path,dmodel_path

from os import listdir
import os
from pathlib import Path
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
import PIL
from numpy import asarray
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from matplotlib import pyplot

from keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint

from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from numpy.random import shuffle
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets import cifar10
from skimage.transform import resize
from numpy import asarray
import cv2



class Ui_window1(object):
    def switch_to_window2(self,window2):
        '''

        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_window2()
        self.ui.setupUi(self.window)
        self.window.show()
        window1.hide()

        '''
        #window1.hide()
        window1.hide()
        window2.show()




    def setupUi(self, window1):
        window1.setObjectName("window1")
        window1.resize(812, 585)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(70, 14, 14))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(70, 14, 14))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(70, 14, 14))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(70, 14, 14))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(70, 14, 14))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(70, 14, 14))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(70, 14, 14))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(70, 14, 14))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(70, 14, 14))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        window1.setPalette(palette)
        window1.setStyleSheet("background-color:#460e0e;")
        self.centralwidget = QtWidgets.QWidget(window1)
        self.centralwidget.setObjectName("centralwidget")
        self.title1 = QtWidgets.QTextEdit(self.centralwidget)
        self.title1.setGeometry(QtCore.QRect(60, 50, 641, 111))
        self.title1.setStyleSheet("background-color:#750606;\n"
"color:white;\n"
"border-style:outset;\n"
"border-width:0.5px;\n"
"border-color:white;\n"
"border-radius:5px;\n"
"\n"
"")
        self.title1.setObjectName("title1")
        self.gsbutton = QtWidgets.QPushButton(self.centralwidget)
        self.gsbutton.setGeometry(QtCore.QRect(280, 310, 201, 41))
        self.gsbutton.setStyleSheet("background-color:#750606;\n"
"color:white;\n"
"border-style:outset;\n"
"border-width:0.5px;\n"
"border-color:white;\n"
"border-radius:5px;\n"
"padding:6px;\n"
"min-width:10px;")
        self.gsbutton.setObjectName("gsbutton")
        self.gsbutton.clicked.connect(lambda: self.switch_to_window2(window2))
        self.cip = QtWidgets.QTextEdit(self.centralwidget)
        self.cip.setGeometry(QtCore.QRect(150, 180, 491, 91))
        self.cip.setStyleSheet("background-color:#750606;\n"
"color:white;\n"
"border-style:outset;\n"
"border-width:0.5px;\n"
"border-color:white;\n"
"border-radius:5px;\n"
"\n"
"")
        self.cip.setObjectName("cip")
        window1.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(window1)
        self.statusbar.setObjectName("statusbar")
        window1.setStatusBar(self.statusbar)

        self.retranslateUi(window1)
        QtCore.QMetaObject.connectSlotsByName(window1)

    def retranslateUi(self, window1):
        _translate = QtCore.QCoreApplication.translate
        window1.setWindowTitle(_translate("window1", "MainWindow"))
        self.title1.setHtml(_translate("window1", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'.AppleSystemUIFont\'; font-size:18pt;\">SATELLITE DOMAIN TRANSLATION USING GENERATIVE ADVERSARIAL NETWORKS</span></p></body></html>"))
        self.gsbutton.setText(_translate("window1", "GET STARTED"))
        self.cip.setHtml(_translate("window1", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'.AppleSystemUIFont\'; font-size:18pt;\">CREATIVE AND INNOVATIVE PROJECT</span></p></body></html>"))


#################################################### WINDOW 2 BEGINS




class Ui_window2(object):
    def switch_to_window3(self,window3):
        '''
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_window3()
        self.ui.setupUi(self.window)
        self.window.show()
        window2.hide()

        '''

        window2.hide()
        window3.show()


    def lg_clicked(self):
        gan.g_model = load_model(gmodel_path)
        print("GENERATOR MODEL LOADED SUCCESSFULLY..!!")
        print("Input = ",gan.g_model.input_shape)
        print("Output = ",gan.g_model.output_shape)

    def ld_clicked(self):

        gan.d_model = load_model(dmodel_path)
        print("DISCRIMINATOR MODEL LOADED SUCCESSFULLY..!!")
        print("Input = ",gan.d_model.input_shape)
        print("Output = ",gan.d_model.output_shape)
        gan.gan_model = gan.define_gan()
        print('GAN MODEL CREATED SUCCESSFULLY..!!')
        print("Input = ",gan.gan_model.input_shape)
        print("Output = ",gan.gan_model.output_shape)

    def sm_clicked(self):
        
        [X1, X2] = gan.load_real_samples(map_path)
        ix = randint(0, len(X1), 1)
        print(ix)
        src_image, tar_image = X1[ix], X2[ix]
        # generate image from source
        gen_image = gan.g_model.predict(src_image)
        # plot all three images
        ####plot_images(src_image, gen_image, tar_image)
        gan.plot_images(src_image,gen_image,tar_image)





    def setupUi(self, window2):
        window2.setObjectName("window2")
        window2.resize(800, 600)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(70, 14, 14))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(70, 14, 14))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(70, 14, 14))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(70, 14, 14))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(70, 14, 14))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(70, 14, 14))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(70, 14, 14))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(70, 14, 14))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(70, 14, 14))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        window2.setPalette(palette)
        window2.setStyleSheet("background-color:#460e0e;")
        self.centralwidget = QtWidgets.QWidget(window2)
        self.centralwidget.setObjectName("centralwidget")
        self.title2 = QtWidgets.QTextEdit(self.centralwidget)
        self.title2.setGeometry(QtCore.QRect(140, 50, 501, 41))
        self.title2.setStyleSheet("background-color:#750606;\n"
"color:white;\n"
"border-style:outset;\n"
"border-width:0.5px;\n"
"border-color:white;\n"
"border-radius:5px;\n"
"\n"
"")
        self.title2.setObjectName("title2")
        self.lgbutton = QtWidgets.QPushButton(self.centralwidget)
        self.lgbutton.setGeometry(QtCore.QRect(90, 200, 211, 81))
        self.lgbutton.setStyleSheet("background-color:#750606;\n"
"color:white;\n"
"border-style:outset;\n"
"border-width:0.5px;\n"
"border-color:white;\n"
"border-radius:5px;\n"
"padding:6px;\n"
"min-width:10px;")
        self.lgbutton.setObjectName("lgbutton")
        self.lgbutton.clicked.connect(self.lg_clicked)
        self.ldbutton = QtWidgets.QPushButton(self.centralwidget)
        self.ldbutton.setGeometry(QtCore.QRect(440, 200, 211, 81))
        self.ldbutton.setStyleSheet("background-color:#750606;\n"
"color:white;\n"
"border-style:outset;\n"
"border-width:0.5px;\n"
"border-color:white;\n"
"border-radius:5px;\n"
"padding:6px;\n"
"min-width:10px;")
        self.ldbutton.setObjectName("ldbutton")
        self.ldbutton.clicked.connect(self.ld_clicked)
        self.smbutton = QtWidgets.QPushButton(self.centralwidget)
        self.smbutton.setGeometry(QtCore.QRect(90, 350, 211, 71))
        self.smbutton.setStyleSheet("background-color:#750606;\n"
"color:white;\n"
"border-style:outset;\n"
"border-width:0.5px;\n"
"border-color:white;\n"
"border-radius:5px;\n"
"padding:6px;\n"
"min-width:10px;")
        self.smbutton.setObjectName("smbutton")
        self.smbutton.clicked.connect(self.sm_clicked)
        self.movebutton = QtWidgets.QPushButton(self.centralwidget)
        self.movebutton.setGeometry(QtCore.QRect(440, 350, 211, 71))
        self.movebutton.setStyleSheet("background-color:#750606;\n"
"color:white;\n"
"border-style:outset;\n"
"border-width:0.5px;\n"
"border-color:white;\n"
"border-radius:5px;\n"
"padding:6px;\n"
"min-width:10px;")
        self.movebutton.setObjectName("movebutton")
        self.movebutton.clicked.connect(lambda : self.switch_to_window3(window3))

        window2.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(window2)
        self.statusbar.setObjectName("statusbar")
        window2.setStatusBar(self.statusbar)

        self.retranslateUi(window2)
        QtCore.QMetaObject.connectSlotsByName(window2)

    def retranslateUi(self, window2):
        _translate = QtCore.QCoreApplication.translate
        window2.setWindowTitle(_translate("window2", "MainWindow"))
        self.title2.setHtml(_translate("window2", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'.AppleSystemUIFont\'; font-size:18pt; font-weight:600;\">MAIN DASHBOARD</span></p></body></html>"))
        self.lgbutton.setText(_translate("window2", "LOAD GENERATOR MODEL"))
        self.ldbutton.setText(_translate("window2", "LOAD DISCRIMINATOR MODEL"))
        self.smbutton.setText(_translate("window2", "SUMMARIZE MODEL"))
        self.movebutton.setText(_translate("window2", "MOVE TO TESTING"))


#################################################### WINDOW 3 BEGINS




class Ui_window3(object):

    def __init__(self):
    #    self.testimagepath = ''
        pass


    def gam_clicked(self):
        gan.testimage =  gan.load_image()
        gan.generatedimage = gan.g_model.predict(gan.testimage)
        print(gan.generatedimage.shape)
        print(type(gan.generatedimage))
        # scale from [-1,1] to [0,1]
        gan.generatedimage = (gan.generatedimage + 1) / 2.0
        # plot the image
        pyplot.imshow(gan.generatedimage[0])
        pyplot.axis('off')
        aerial_map = gan.testimagepath.split('/')[-2] + '/' +gan.testimagepath.split('/')[-1]
        pyplot.savefig("../Model/Dataset/aerial/"+aerial_map)
        pyplot.show()


    def de_clicked(self):
        print("\n\nDETECTING THE EDIFICES")
        gan.detect_edifices()

    def cis_clicked(self):
        print("\n\nCALCULATING INCEPTION SCORE")
        [X1, X2] = gan.load_real_samples("../Model/Dataset/maps_256.npz")
        print('Loaded', X1.shape, X2.shape)
        val_predicted = list()
        i = 0
        for image in X1:
          print(i)
          i = i + 1
          image = expand_dims(image,axis = 0)
          gen_image = gan.g_model.predict(image)
          val_predicted.append(gen_image)

        gan.out = asarray(val_predicted)
        gan.out = np.transpose(gan.out,(1,0,2,3,4))
        gan.out = gan.out[0]
        images = gan.out
        shuffle(images)
        print('loaded', images.shape)
        is_avg, is_std = gan.calculate_inception_score(images)
        print('score', is_avg, is_std)

          


    def input_clicked(self):
        file = QFileDialog.getOpenFileName()
        gan.testimagepath = file[0]
        print(gan.testimagepath)
        print(type(gan.testimagepath))
        print("\n\nTEST IMAGE INPUTTED SUCCESSFULLY..!!")


        

    def setupUi(self, window3):
        window3.setObjectName("window3")
        window3.resize(800, 600)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(70, 14, 14))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(70, 14, 14))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(70, 14, 14))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(70, 14, 14))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(70, 14, 14))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(70, 14, 14))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(70, 14, 14))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(70, 14, 14))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(70, 14, 14))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        window3.setPalette(palette)
        window3.setStyleSheet("background-color:#460e0e;")
        self.centralwidget = QtWidgets.QWidget(window3)
        self.centralwidget.setObjectName("centralwidget")
        self.inputbutton = QtWidgets.QPushButton(self.centralwidget)
        self.inputbutton.setGeometry(QtCore.QRect(100, 210, 231, 81))
        self.inputbutton.setStyleSheet("background-color:#750606;\n"
"color:white;\n"
"border-style:outset;\n"
"border-width:0.5px;\n"
"border-color:white;\n"
"border-radius:5px;\n"
"padding:6px;\n"
"min-width:10px;")
        self.inputbutton.setObjectName("inputbutton")
        self.inputbutton.clicked.connect(self.input_clicked)
        self.title3 = QtWidgets.QTextEdit(self.centralwidget)
        self.title3.setGeometry(QtCore.QRect(150, 60, 501, 51))
        self.title3.setStyleSheet("background-color:#750606;\n"
"color:white;\n"
"border-style:outset;\n"
"border-width:0.5px;\n"
"border-color:white;\n"
"border-radius:5px;\n"
"\n"
"")
        self.title3.setObjectName("title3")
        self.gambutton = QtWidgets.QPushButton(self.centralwidget)
        self.gambutton.setGeometry(QtCore.QRect(440, 210, 241, 81))
        self.gambutton.setStyleSheet("background-color:#750606;\n"
"color:white;\n"
"border-style:outset;\n"
"border-width:0.5px;\n"
"border-color:white;\n"
"border-radius:5px;\n"
"padding:6px;\n"
"min-width:10px;")
        self.gambutton.setObjectName("gambutton")
        self.gambutton.clicked.connect(self.gam_clicked)
        self.debutton = QtWidgets.QPushButton(self.centralwidget)
        self.debutton.setGeometry(QtCore.QRect(100, 360, 231, 81))
        self.debutton.setStyleSheet("background-color:#750606;\n"
"color:white;\n"
"border-style:outset;\n"
"border-width:0.5px;\n"
"border-color:white;\n"
"border-radius:5px;\n"
"padding:6px;\n"
"min-width:10px;\n"
"")
        self.debutton.setObjectName("debutton")
        self.debutton.clicked.connect(self.de_clicked)
        self.cisbutton = QtWidgets.QPushButton(self.centralwidget)
        self.cisbutton.setGeometry(QtCore.QRect(440, 360, 241, 81))
        self.cisbutton.setStyleSheet("background-color:#750606;\n"
"color:white;\n"
"border-style:outset;\n"
"border-width:0.5px;\n"
"border-color:white;\n"
"border-radius:5px;\n"
"padding:6px;\n"
"min-width:10px;")
        self.cisbutton.setObjectName("cisbutton")
        self.cisbutton.clicked.connect(self.cis_clicked)
        window3.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(window3)
        self.statusbar.setObjectName("statusbar")
        window3.setStatusBar(self.statusbar)

        self.retranslateUi(window3)
        QtCore.QMetaObject.connectSlotsByName(window3)

    def retranslateUi(self, window3):
        _translate = QtCore.QCoreApplication.translate
        window3.setWindowTitle(_translate("window3", "MainWindow"))
        self.inputbutton.setText(_translate("window3", "INPUT TEST IMAGE"))
        self.title3.setHtml(_translate("window3", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'.AppleSystemUIFont\'; font-size:18pt; font-weight:600;\">MODEL TESTING</span></p></body></html>"))
        self.gambutton.setText(_translate("window3", "GENERATE AERIAL MAP VERSION"))
        self.debutton.setText(_translate("window3", "DETECT EDIFICE"))
        self.cisbutton.setText(_translate("window3", "CALCULATE FID"))


###############################################  MAIN FUNCTION CALL BEGINS


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gan = GAN()
    
    gmodel_path = "../Model/Trained_Models/g20.2"
    dmodel_path = "../Model/Trained_Models/d20.2"
    
    window1= QtWidgets.QMainWindow()
    ui = Ui_window1()
    ui.setupUi(window1)
    window2 = QtWidgets.QMainWindow()
    ui = Ui_window2()
    ui.setupUi(window2)
    window3 = QtWidgets.QMainWindow()
    ui = Ui_window3()
    ui.setupUi(window3)
    window1.show()


    #Load dataset
    #gan.compress_images("../Model/Dataset/maps/val/","../Model/Dataset/val_256.npz")
    map_path = "../Model/Dataset/maps_256.npz"
    dataset = gan.load_real_samples(map_path)
    print('Loaded', dataset[0].shape, dataset[1].shape)
    gan.image_shape = dataset[0].shape[1:]
    

    sys.exit(app.exec_())

