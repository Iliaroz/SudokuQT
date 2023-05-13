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
from PyQt6.QtCore import pyqtSlot
## model, detection, video
import cv2 as cv

from enum import Enum, IntEnum

## Other parts
from SudokuRecognition import SudokuRecognition


class VideoMode(Enum):
    SudokuSolution = 1
    DigitRecognition = 2
    TableDetection = 3
    ImageFiltered = 4
    OriginalImage = 5
    def __eq__(self, other):    return self.value ==  other.value
    def __ne__(self, other):    return self.value !=  other.value


# logging.basicConfig(filename='./App-running.log', filemode='w', 
#                     encoding='utf-8', level=logging.ERROR)
# logger = logging.getLogger("AppSudoku")
# logger.setLevel(logging.DEBUG)



class Signals(QtCore.QObject):
    """ Signals (Qt signals) used in whole application
    """

    signal_detected_sudoku_board = QtCore.pyqtSignal(tuple) ## recognized board and solved board
    
    signal_change_original_pixmap = QtCore.pyqtSignal(tuple)   ## tuple:(bool: flag, np.ndarray)
    signal_change_processed_pixmap = QtCore.pyqtSignal(tuple)  ## tuple:(bool: flag, np.ndarray)
    signal_video_change_mode = QtCore.pyqtSignal(VideoMode)

    signal_open_image_file = QtCore.pyqtSignal(str)
    signal_recognition_process = QtCore.pyqtSignal(tuple)   ## status of recognition process
    signal_recognition_completed = QtCore.pyqtSignal(bool)  ## 
    

    
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
            
################################################################


class QActingPushButton(QtWidgets.QPushButton):
    """QPushButtons don't interact with their QActions. This class triggers
    every `QAction` in `self.actions()` when the `clicked` signal is emitted.
    https://stackoverflow.com/a/16703358
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.clicked.connect(self.trigger_actions)

    @QtCore.pyqtSlot()
    def trigger_actions(self) -> None:
        for act in self.actions():
            act.trigger()


###########################################################
###########################################################
### Recognition thread classs and worker
###########################################################
###########################################################
# QTTread that run always and recive events from gui




class WorkerRecognition(QtCore.QObject):

    def __init__(self, parent=None):

        super().__init__()
        self.signals = AppSignals
        self.imageFileName = None
        self.SudokuRecognition = None
        ## App signals connection
        self.signals.signal_video_change_mode.connect(self.slot_change_video_mode)
        self.signals.signal_open_image_file.connect(self.slot_load_image)

    @pyqtSlot()
    def start(self):
        print("Worker start")
        # self.SudokuRecognition = SudokuRecognition()
        pass

    @QtCore.pyqtSlot(str)
    def slot_load_image(self, imageFileName):
        if self.SudokuRecognition is None:
            print("Creating SudokuRecognition class instance")
            self.SudokuRecognition = SudokuRecognition()
        self.imageFileName = imageFileName
        self.SudokuRecognition.setNewImage(self.imageFileName)
        res = False
        try:
            res = self.SudokuRecognition.MakeImageRecognition(callback = self.callback_process)
            if res:
                self.signals.signal_detected_sudoku_board.emit( (
                        self.SudokuRecognition.BoardRecognition,
                        self.SudokuRecognition.BoardSolution) )
        except:
            logger.error("Cannot recognize image")
        finally:
            self.signals.signal_recognition_completed.emit(res)

    def callback_process(self):
        sr = self.SudokuRecognition
        status = tuple([
            sr.f_image_loaded,          ## image loaded
            sr.f_image_preprocessed,    ## image preprocessed
            sr.f_image_extracted,       ## table extracted
            sr.f_image_recognized,      ## table digits recognized
            sr.f_image_solved          ## sudoku solved
            ])
        self.signals.signal_recognition_process.emit(status)
        pass

    @QtCore.pyqtSlot(VideoMode)
    def slot_change_video_mode(self, mode):
        """Slot, catch signal from Gui to change video mode: what frame will be passed to GUI
        """
        ## TODO: !!! make functions in Recognition class to get images
        logger.info("Changing video mode: mode=" + str(mode))
        self.mode = mode
        if self.SudokuRecognition is None:
            return
        image_show = None
        if (self.mode == VideoMode.OriginalImage):
            image_show =    self.SudokuRecognition.Image_original
        elif (self.mode == VideoMode.SudokuSolution):
            image_show =    self.SudokuRecognition.Image_solved
        elif (self.mode == VideoMode.DigitRecognition):
            image_show =    self.SudokuRecognition.Image_recognized
        elif (self.mode == VideoMode.TableDetection):
            image_show =    self.SudokuRecognition.Image_extracted
        elif (self.mode == VideoMode.ImageFiltered):
            image_show =    self.SudokuRecognition.Image_preprocessed
        
        if image_show is None:
            logger.info("No image to show")
            self.signals.signal_change_processed_pixmap.emit((False, image_show))
            return
        self.signals.signal_change_processed_pixmap.emit((True, image_show))
        pass    



        

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
        self.setFixedSize(self.size())
        self.setWindowIcon(QtGui.QIcon(self.icons["icon"]))
        self.GameSize = 9 ## sudoku size
        self.setWindowTitle("Qt sudoku recognizer and solver")
        self.display_width = 320
        self.display_height = 300
        self.OriginalImage = self.generateImage("Load image first")
        ## create the label that holds the image
        self.imageLabelOriginal.resize(self.display_width, self.display_height)

        self.imageLabelProcessed.resize(self.display_width, self.display_height)       
        self.imageLabelProcessed.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.imageLabelProcessed.customContextMenuRequested.connect(self.video_on_context_menu)

        ## OpenImage action
        icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton)
        self.actionOpenImage.setIcon(icon)
        self.actionOpenImage.triggered.connect(self.act_open_image_clicked)
        self.actionStopRecognition.triggered.connect(self.act_stop_recognition)
        self.actionStopRecognition.setEnabled(False)
        # self.actionOpenImage.setEnabled(False)

        ## OpenImage button
        self.btnLoadImage = QActingPushButton(self.centralwidget)
        self.btnLoadImage.setObjectName(u"btnLoadImage")
        self.btnLoadImage.setMaximumSize(QtCore.QSize(150, 16777215))
        self.layoutButtons.addWidget(self.btnLoadImage, 0, 0, 1, 1)
        self.btnLoadImage.setText(QtCore.QCoreApplication.translate("MainWindow", u"Open image..", None))
        self.btnLoadImage.addAction(self.actionOpenImage)

        ## Solve button
        self.btnSolveSudoku = QActingPushButton(self.centralwidget)
        self.btnSolveSudoku.setObjectName(u"btnSolveSudoku")
        self.btnSolveSudoku.setMaximumSize(QtCore.QSize(150, 16777215))
        self.layoutButtons.addWidget(self.btnSolveSudoku, 1, 0, 1, 1)
        self.btnSolveSudoku.setText(QtCore.QCoreApplication.translate("MainWindow", u"Solve!", None))
        self.btnSolveSudoku.setEnabled(False)

        self.makeGameArea()
        self.signals = AppSignals

        ## connect its signal to the update_image slot
        self.signals.signal_change_original_pixmap.connect(self.update_original_image)
        self.signals.signal_change_processed_pixmap.connect(self.update_processed_image)
        self.signals.signal_detected_sudoku_board.connect(self.update_game_buttons_detection)
        self.signals.signal_recognition_process.connect(self.update_checkboxes_recognition)
        self.signals.signal_recognition_completed.connect(self.recognition_completed)

        self.signals.signal_change_original_pixmap.emit((True, self.OriginalImage))

        ##############################################
        ### Recognizer thread and worker handling and starting
        # 1 - create Worker and Thread inside the Form
        self.worker_recognition = WorkerRecognition()  # no parent!
        self.threadRecognition = QtCore.QThread()  # no parent!
        # 2 - Connect Worker`s Signals to Form method slots to post data.
        # 3 - Move the Worker object to the Thread object
        self.worker_recognition.moveToThread(self.threadRecognition)
        # 4 - Connect Worker Signals to the Thread slots
        # 5 - Connect Thread started signal to Worker operational slot method
        self.threadRecognition.started.connect(self.worker_recognition.start)
        # * - Thread finished signal will close the app if you want!
        self.threadRecognition.finished.connect(app.exit)
        # 6 - Start the thread
        self.threadRecognition.start()
        

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
        self.threadRecognition.stop()
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
            "JPG files (*.jpg; *.jpeg);; PNG Files (*.png);; All Files (*)",
        )
        logger.debug(f"OpenImage: result {openFileResult}.")
        fname = openFileResult[0]
        if fname:
            logger.debug(f'OpenImage: filename to open "{fname}".')
            self.LoadImageFromFile(fname)

    def act_stop_recognition(self):
        logger.info("StopRecognition button clicked.")
        # self.signals.signal_game_stopped.emit()

    def LoadImageFromFile(self, fname):
        img = cv.imread(fname, cv.IMREAD_COLOR)
        if img is not None:
            self.OriginalImage = img
            self.actionOpenImage.setEnabled(False)
            self.signals.signal_change_original_pixmap.emit((True, self.OriginalImage))
            self.signals.signal_change_processed_pixmap.emit((None, None))
            self.signals.signal_detected_sudoku_board.emit((None, None))
            self.signals.signal_open_image_file.emit(fname)
        pass

    def recognition_completed(self):
        logger.info("Recognition completed.")
        self.actionOpenImage.setEnabled(True)

    def btn_game_pressed(self):
        btn = self.sender()
        logger.info("GameButton pressed." + 'r=' + str(btn.row) + 'c=' + str(btn.col))
        print("")
        pass

    @QtCore.pyqtSlot()
    def slot_show_image_loaded(self):
        """request to show image loaded
        """
        print("slot_show_image_loaded")
        self.signals.signal_video_change_mode.emit(VideoMode.OriginalImage)
        pass

    @QtCore.pyqtSlot()
    def slot_show_image_processed(self):
        """request to show image processed: filtered into BW
        """
        print("slot_show_image_processed")
        self.signals.signal_video_change_mode.emit(VideoMode.ImageFiltered)
        pass

    @QtCore.pyqtSlot()
    def slot_show_image_transformed(self):
        """request to show image with selected sudoku and transformed to square
        """
        print("slot_show_image_transformed")
        self.signals.signal_video_change_mode.emit(VideoMode.TableDetection)
        pass
    @QtCore.pyqtSlot()
    def slot_show_image_recognized(self):
        """request to show image with recognized sudoku digits
        """
        print("slot_show_image_recognized")
        self.signals.signal_video_change_mode.emit(VideoMode.DigitRecognition)
        pass
        
    @QtCore.pyqtSlot()
    def slot_show_image_solved(self):
        """request to show image with solved sudoku digits and solution
        """
        print("slot_show_image_solved")
        self.signals.signal_video_change_mode.emit(VideoMode.SudokuSolution)
        pass
        
    @QtCore.pyqtSlot(tuple)
    def update_checkboxes_recognition(self, status):
        """update checkboxes with recognition status
        """
        print("update_checkboxes_recognition")
        checkboxes = [self.cbxLoaded,
                      self.cbxPreprocessed,
                      self.cbxTransformed,
                      self.cbxRecognized,
                      self.cbxSolved]
        modes = [VideoMode.OriginalImage,
                 VideoMode.ImageFiltered,
                 VideoMode.TableDetection,
                 VideoMode.DigitRecognition,
                 VideoMode.SudokuSolution]
        for wd, st, md in zip(checkboxes, status, modes):
            wd.setChecked(st)
            print(f"st={st}")
            if st:
                mode = md
        self.signals.signal_video_change_mode.emit(mode)
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
                btn.setText("")
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


    @QtCore.pyqtSlot(tuple)
    def update_game_buttons_detection(self, boards):
        """Updates the values of game buttons"""
        GBrec, GBsolved = boards
        if GBrec is None:
            GBrec = np.zeros((self.GameSize, self.GameSize))
        if GBsolved is None:
            GBsolved = GBrec
        for i in reversed(range(self.gameLayout.count())): 
            btn = self.gameLayout.itemAt(i).widget()
            num = int(GBrec[btn.row, btn.col])
            useSolv = not bool(0 < num < 10)
            ## table selection
            print(f"Num: {num},  Usesolved: { str(useSolv)}")
            if useSolv:
                num = int(GBsolved[btn.row, btn.col])
                color = "(0,0,169)"
            else:
                num = int(GBrec[btn.row, btn.col])
                color = "(0,169,0)"
            ## show number 1...9
            if 0<num<10:
                text = str(num)
            else:
                text = ""
            if color != None:
                style = "color:rgb" + color + ";" + "font-size:24px;"
            else:
                style = ""
            btn.setStyleSheet(style)
            btn.setText(text)




    @QtCore.pyqtSlot(tuple)
    def update_original_image(self, values):
        """Updates the image_label with a new opencv image"""
        if values[0] is None or values[0] == False:
            cv_img = self.generateImage("no image")
        else:
            cv_img = values[1]
        qt_img = self.convert_cv_qt(cv_img)
        self.imageLabelOriginal.setPixmap(qt_img)

    @QtCore.pyqtSlot(tuple)
    def update_processed_image(self, values):
        """Updates the imageLabelProcessed with a new opencv image"""
        if values[0] is None or values[0] == False:
            cv_img = self.generateImage("no image")
        else:
            cv_img = values[1]
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




    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        scaled_Qt_image = convert_to_Qt_format.scaled(self.display_width, self.display_height, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        qtpix = QtGui.QPixmap.fromImage(scaled_Qt_image)
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
    os.makedirs('./debug', exist_ok=True)

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