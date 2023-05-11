# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:02:55 2022

@author: ilia
"""
## General libraries
import sys, os, time
import numpy as np
import logging
## GUI
from PyQt6 import QtGui, QtWidgets, QtCore, uic
## model, detection, video
import cv2 as cv

from enum import Enum, IntEnum


class VideoMode(Enum):
    FullRecognition = 1
    TextRecognition = 2
    TableDetection = 3
    ImageFiltered = 4
    def __eq__(self, other):    return self.value ==  other.value
    def __ne__(self, other):    return self.value !=  other.value


# logging.basicConfig(filename='./App-running.log', filemode='w', 
#                     encoding='utf-8', level=logging.ERROR)
# logger = logging.getLogger("AppSudoku")
# logger.setLevel(logging.DEBUG)



class Signals(QtCore.QObject):
    """ Signals (Qt signals) used in whole application
    """
    signal_game_started = QtCore.pyqtSignal()
    signal_game_stopped = QtCore.pyqtSignal()
    signal_game_status = QtCore.pyqtSignal(str)
    signal_game_p1_status = QtCore.pyqtSignal(str)
    signal_game_p2_status = QtCore.pyqtSignal(str)
    signal_game_board = QtCore.pyqtSignal(np.ndarray) ## Approved board
    
    signal_change_pixmap = QtCore.pyqtSignal(np.ndarray)
    signal_detection_matrix = QtCore.pyqtSignal(np.ndarray)
    signal_video_change_port = QtCore.pyqtSignal(int)
    signal_video_change_mode = QtCore.pyqtSignal(VideoMode)
    
    signal_app_start_game = QtCore.pyqtSignal(tuple)
    signal_app_stop_game = QtCore.pyqtSignal()
    
    def __init__(self):
        super().__init__()
        pass

AppSignals = Signals()
###########################################################
###########################################################
### Helper  classs
###########################################################
###########################################################

class DeltaQueue(object):
    def __init__(self, size, initVal=None):
        self.N = size
        self.Samples =  [initVal] * self.N
        self.currN = 0
        self._last = initVal
        self._val = initVal
        self._integral = 0.0
        
    # using property decorator
    # a getter function
    @property
    def val(self):
        return self._val
    # a setter function
    @val.setter
    def val(self, element):
        try:
            element = element
        except:
            return
        self.add(element)


    @property
    def last(self):
        return self._last

    
    def add(self, element):
        cutted = self.Samples
        self.Samples = cutted + [element]
        self._val = element
        if (self.currN < self.N):
            self.currN += 1
        self._last = self.Samples[-self.currN]
        self._integral = self._integral + (self._val + self._last) / 2.0 * self.currN

    def Equal(self):
        E = []
        for i in range(self.N-1):
            e = np.array_equiv( self.Samples[i],  self.Samples[i+1])
            E.append(e)
        res = np.asarray(E).all()
        return res
            


    

###########################################################
###########################################################
### Application classs
###########################################################
###########################################################

class AppSudoku(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        ## icons list
        self.icons = {
            "icon"      : "./icons/icon.png",
            }
        uic.loadUi('./app.ui', self)
        ## non-resizable main window
        #self.setFixedSize(self.size())
        self.setWindowIcon(QtGui.QIcon(self.icons["icon"]))
        self.GameSize = 9 ## sudoku size
        self.setWindowTitle("Qt sudoku recognizer and solver")
        self.display_width = 320
        self.display_height = 240
        self.OriginalImage = self.generateImage("Load image first")
        ## create the label that holds the image
        self.imageLabelOriginal.resize(self.display_width, self.display_height)

        self.imageLabelProcessed.resize(self.display_width, self.display_height)       
        self.imageLabelProcessed.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.imageLabelProcessed.customContextMenuRequested.connect(self.video_on_context_menu)

        ## Log editors for read only
        self.logApp.setReadOnly(True)

        ## Openbutton
        icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton)
        self.actionOpenImage.setIcon(icon)

        self.makeGameArea()
        self.signals = AppSignals

        ## connect its signal to the update_image slot
        self.signals.signal_change_pixmap.connect(self.update_original_image)
        self.signals.signal_detection_matrix.connect(self.update_game_buttons_detection)

        self.signals.signal_change_pixmap.emit(self.OriginalImage)
        

    def generateImage(self, text):
        blank_image = np.zeros((self.display_height, self.display_width,3), np.uint8)
        blank_image[:,:] = (0,128,0)
        font                   = cv.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,self.display_width // 3)
        fontScale              = 1
        fontColor              = (255,255,255)
        thickness              = 2
        lineType               = 2
        
        cv.putText(blank_image,text, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)
        return blank_image


    def closeEvent(self, event):
        logger.info("Close event received.")
        #self.threadGame.stop()
        #self.threadVideo.stop()
        event.accept()



    def act_open_image_clicked(self):
        """
        Open dialog for image file selection
        """
        logger.debug("OpenImage button clicked.")
        openFileResult = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open File",
            "",
            "All Files (*);; JPG files (*.jpg, *.jpeg);; PNG Files (*.png)",
        )
        logger.debug(f"OpenImage: result {openFileResult}.")
        fname = openFileResult[0]
        if fname:
            logger.debug(f'OpenImage: filename to open "{fname}".')
            self.LoadImageFromFile(fname)

    def act_stop_recognition(self):
        logger.info("StopRecognition button clicked.")
        self.signals.signal_game_stopped.emit()

    def LoadImageFromFile(self, fname):
        img = cv.imread(fname, cv.IMREAD_GRAYSCALE)
        if img is not None:
            self.OriginalImage = img
            self.signals.signal_change_pixmap.emit(self.OriginalImage)
        pass



    def btn_game_pressed(self):
        btn = self.sender()
        logger.info("GameButton pressed." + 'r=' + str(btn.row) + 'c=' + str(btn.col))
#        self.setGameButtonIcon(btn, "red")
        pass

        

    def setGameButtonIcon(self, btn, icon):
        """
        set button icon and enlarge it to full size
        """
        #ico = QtGui.QIcon(self.icons[icon])
        #btn.setIcon(ico)
        #btn.setIconSize(btn.size())
        pass


    def makeGameArea(self):
        """
        Form the button area using <Board Size>. 
        """
        ## delete all existed buttons
        for i in reversed(range(self.gameLayout.count())): 
            self.gameLayout.itemAt(i).widget().setParent(None)

        ## create buttons
        ## TODO: not buttons, but special widget
        for c in range(self.GameSize):
            for r in range(self.GameSize):
                btn = QtWidgets.QPushButton("")
                btn.setFixedSize(40,40)
                self.setGameButtonIcon(btn, "empty")
                btn.clicked.connect(self.btn_game_pressed)
                btn.row = r
                btn.col = c
                ## skip every 3rd row/col
                ra = r // 3
                ca = c // 3
                self.gameLayout.addWidget(btn, r+ra, c+ca)
        ## set skipped row/col minimal size to make space btwn
        for i in range(3,self.GameSize,2):
            self.gameLayout.setColumnMinimumWidth (i, 3)
            self.gameLayout.setRowMinimumHeight(i, 3)
        pass


    def video_on_context_menu(self, pos):
        """
        Create Context Menu on video frame
        """
        contextMenu = QtWidgets.QMenu(self)
        Acts = []
        for i, mode in enumerate(VideoMode):
            act = QtGui.QAction(mode.name, self)
            act.modeToSwitch = mode
            Acts.append(act)
        contextMenu.addActions(Acts)
        
        ## contextMenu for cameraImageLabel only, so use it for position search
        action = contextMenu.exec(self.imageLabelProcessed.mapToGlobal(pos))   
        ## pass action to video class
        if action != None:
            if hasattr(action, 'modeToSwitch'):
                self.signals.signal_video_change_mode.emit(action.modeToSwitch)
            if hasattr(action, 'newVideoPort'):
                self.signals.signal_video_change_port.emit(action.newVideoPort)
                

    @QtCore.pyqtSlot(np.ndarray)
    def update_game_buttons_detection(self, GB):
        """Updates the images of game buttons, placing detected cups"""
        ## TODO: BoardState
        color_map = {-1 : "red", 0 : "empty", 1 : "blue" }
        for i in reversed(range(self.gameLayout.count())): 
            btn = self.gameLayout.itemAt(i).widget()
            self.setGameButtonIcon(btn, color_map[ int(GB[btn.row, btn.col]) ])

    @QtCore.pyqtSlot(np.ndarray)
    def update_game_buttons_board(self, GB):
        """Updates the color of game buttons according to current approved game Board"""
        ## TODO: BoardState
        color_map = {-1 : "(255,128,128,128)", 0 : None, 1 : "(128,128,255,128)" }
        for i in reversed(range(self.gameLayout.count())): 
            btn = self.gameLayout.itemAt(i).widget()
            color = color_map[ int(GB[btn.row, btn.col]) ]
            if color != None:
                style = "background-color:rgba" + color + ";"
            else:
                style = ""
            btn.setStyleSheet(style)


    @QtCore.pyqtSlot(np.ndarray)
    def update_original_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.imageLabelOriginal.setPixmap(qt_img)

    @QtCore.pyqtSlot(np.ndarray)
    def update_processed_image(self, cv_img):
        """Updates the image_label2 with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.imageLabelProcessed.setPixmap(qt_img)

        
    @QtCore.pyqtSlot()
    def update_game_started(self):
        icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaStop)
        self.btnGameControl.setIcon(icon)
        self.btnGameControl.setText("Stop game")
        pass

    @QtCore.pyqtSlot()
    def update_game_stopped(self):
        icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay)
        self.btnGameControl.setIcon(icon)
        self.btnGameControl.setText("Start game")
        pass

    @QtCore.pyqtSlot(str)
    def update_game_status(self, stat):
        self.lblGameStatus.setText(stat)
        pass

    @QtCore.pyqtSlot(str)
    def update_game_p1_status(self, stat):
        self.lblPlayer1Status.setText(stat)
        pass

    @QtCore.pyqtSlot(str)
    def update_game_p2_status(self, stat):
        self.lblPlayer2Status.setText(stat)
        pass


    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        qtpix = QtGui.QPixmap.fromImage(p)
        return qtpix

    
###########################################################
###########################################################
###########################################################
###########################################################
### Running App
###########################################################
###########################################################
###########################################################
###########################################################
    
if __name__=="__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    logging.basicConfig(filename='./App-running.log', filemode='w', 
                    encoding='utf-8', level=logging.ERROR)
    logger = logging.getLogger("AppSudoku")
    logger.setLevel(logging.DEBUG)
    logger.debug('*' * 80)
    logger.debug('*' * 80)
    logger.debug('*' * 80)

    app = QtWidgets.QApplication(sys.argv)
    a = AppSudoku()
    a.show()
    sys.exit(app.exec())