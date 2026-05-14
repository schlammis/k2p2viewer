import os,sys

# --- startup diagnostics (active in both dev and PyInstaller builds) ---
_appdir = os.path.dirname(sys.executable) if getattr(sys,'frozen',False) \
          else os.path.dirname(os.path.abspath(__file__))
_logpath = os.path.join(_appdir,'k2viewer_startup.log')
_logfile = open(_logpath,'w',buffering=1)

def _log(msg):
    _logfile.write(msg+'\n')
    _logfile.flush()

_log('=== k2viewer startup ===')
_log(f'python   : {sys.version}')
_log(f'frozen   : {getattr(sys,"frozen",False)}')
_log(f'exe      : {sys.executable}')
_log(f'cwd      : {os.getcwd()}')
if getattr(sys,'frozen',False):
    _log(f'MEIPASS  : {sys._MEIPASS}')

try:
    from appconfig import AppConfig
    _log('import appconfig OK')
except Exception as e:
    _log(f'import appconfig FAILED: {e}')
    raise
import threading
import traceback
import ctypes
import gzip
import pickle
import xlwt,xlrd,xlutils.copy
import datetime
from shutil import copyfile
_log('stdlib imports OK')

##https://www.pythonguis.com/tutorials/pyqt-basic-widgets/
##
##https://www.geeksforgeeks.org/pyqt5-qtabwidget/
#https://realpython.com/python-pyqt-qthread/
#https://stackoverflow.com/questions/6783194/background-thread-with-qthread-in-pyqt
#
from PyQt5.QtCore import (
    Qt,
    QSize,
    QMutex,
    QObject,
    QThread,
    QTimer,
    pyqtSignal,
    pyqtSlot )

from PyQt5.QtGui import QFont,QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QSpacerItem,
    QSizePolicy,
    QStatusBar,
    QSpinBox,
    QDoubleSpinBox,
    QAbstractSpinBox,
    QProgressBar,
    QPlainTextEdit,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QHeaderView,
    QMenu,
    QLineEdit
)
try:
    import pyi_splash  # type: ignore
    pyi_splash.update_text('Loading interface...')
    try:
        pyi_splash._SPLASH_SCREEN.tk.attributes('-topmost', False)
    except Exception:
        pass
    _splash = pyi_splash
except Exception:
    _splash = None

def _splash_msg(msg):
    if _splash:
        _splash.update_text(msg)

import mplwidget
_log('mplwidget OK')
_splash_msg('Loading numpy...')
import numpy as np
import time
import sqlite3
_splash_msg('Loading dataset...')
import k2dataset
_log('k2dataset OK')
_splash_msg('Loading tools...')
import k2toolsnew as k2tools
_log('k2tools OK')
_splash_msg('Starting application...')

def _read_version():
    for candidate in [_appdir, os.path.join(_appdir, '..')]:
        try:
            with open(os.path.join(candidate, 'version.txt')) as _f:
                return _f.read().strip()
        except Exception:
            pass
    return "unknown"
APP_VERSION = _read_version()
LOG_LEVEL   = AppConfig().loglevel

mutex = QMutex()
kda =   k2dataset.k2Set(mutex)
kda.setcoverage(2)
kda.clear()



_runtimelog = open(os.path.join(_appdir, 'k2viewer_runtime.log'), 'w', buffering=1)

class DiagStream(QObject):
    newText = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._buf = ''

    def write(self, text):
        self._buf += text
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            if line:
                ts = datetime.datetime.now().strftime('%y%m%d %H%M%S')
                _runtimelog.write(f'{ts} {line}\n')
                _runtimelog.flush()
                self.newText.emit(line)

    def flush(self):
        if self._buf.strip():
            self.newText.emit(self._buf)
            self._buf = ''


class TableLoader(QObject):
    activeRow = pyqtSignal(str, str, str, str, str)   # balance, run, title, mass, unc
    greyRow   = pyqtSignal(str, str, str, str, str)
    finished  = pyqtSignal()

    def __init__(self, bd, balance_name, known_balances, force_scan=False):
        super().__init__()
        self.bd = bd
        self.balance_name = balance_name
        self.known_balances = known_balances  # all balance names from config.ini
        self.force_scan = force_scan
        self._abort = False

    def abort(self):
        self._abort = True

    @pyqtSlot()
    def run(self):
        try:
            self._load()
        except Exception as e:
            print(f'TableLoader error: {e}')
        finally:
            self.finished.emit()

    @staticmethod
    def _write_run_error(run_dir, reason):
        try:
            ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(os.path.join(run_dir, 'k2readerror.dat'), 'w') as f:
                f.write(f'k2p2viewer v{APP_VERSION}  {ts}\n')
                f.write(f'Directory: {run_dir}\n')
                f.write(f'{reason}\n')
            print(f'[error] wrote k2readerror.dat in {run_dir}: {reason}')
        except Exception as we:
            print(f'[error] could not write k2readerror.dat: {we}')

    def _load(self):
        c = k2dataset.MyConfig()
        conn = sqlite3.connect('k2viewer.db')
        cur = conn.cursor()
        try:
            for s1 in sorted([f.path for f in os.scandir(self.bd) if f.is_dir()], reverse=True):
                if self._abort:
                    return
                yymm = os.path.split(s1)[-1]
                if len(yymm) != 4:
                    continue
                for s2 in sorted([f.path for f in os.scandir(s1) if f.is_dir()], reverse=True):
                    if self._abort:
                        return
                    day = os.path.split(s2)[-1]
                    if len(day) != 2:
                        continue
                    for s3 in sorted([f.path for f in os.scandir(s2) if f.is_dir()], reverse=True):
                        if self._abort:
                            return
                        letter = os.path.split(s3)[-1]
                        if len(letter) != 1:
                            continue
                        if not self.force_scan and os.path.isfile(os.path.join(s3, 'k2readerror.dat')):
                            continue
                        if not os.path.isfile(os.path.join(s3, 'config.ini')):
                            self._write_run_error(s3, 'No config.ini found in run directory')
                            continue
                        try:
                            c.setbd0(s3)
                            err_file = os.path.join(s3, 'k2readerror.dat')
                            if os.path.isfile(err_file):
                                os.remove(err_file)
                                print(f'[error] removed stale k2readerror.dat from {s3}')
                            run = yymm + day + letter
                            title = str(c.title)
                            rows = cur.execute(
                                "SELECT value,uncertainty FROM k2data WHERE run=? AND balance=?",
                                (run, self.balance_name)).fetchall()
                            if rows:
                                val, unc = rows[0]
                                mass_str = '{0:,.4f}'.format(val) if val > -9e96 else 'n/a'
                                unc_str  = '{0:6.4f}'.format(unc) if unc > -9e96 else 'n/a'
                            else:
                                mass_str = unc_str = ''
                            self.activeRow.emit(self.balance_name, run, title, mass_str, unc_str)
                        except Exception as e:
                            self._write_run_error(
                                s3,
                                f'Error reading run: {e}\n\n'
                                f'File: {os.path.join(s3, "config.ini")}\n'
                                f'{traceback.format_exc()}')
                            print(f'Problem reading {s3}: {e}')

            if not self._abort:
                other = [b for b in self.known_balances if b != self.balance_name]
                if other:
                    placeholders = ','.join('?' * len(other))
                    for run, bal, val, unc, title in cur.execute(
                            f"SELECT run,balance,value,uncertainty,title FROM k2data "
                            f"WHERE balance IN ({placeholders}) ORDER BY run DESC", other).fetchall():
                        if self._abort:
                            return
                        mass_str = '{0:,.4f}'.format(val) if val > -9e96 else 'n/a'
                        unc_str  = '{0:6.4f}'.format(unc) if unc > -9e96 else 'n/a'
                        self.greyRow.emit(bal, run, title or '', mass_str, unc_str)
        finally:
            conn.close()


class Worker(QObject):
    finished = pyqtSignal()
    intReady = pyqtSignal(int,int,int)

    def __init__(self,excl3,order,usesinc,dropfirst=0,ignore_cache=False):
        super().__init__()
        self.order =order
        self.excl3=excl3
        self.usesinc = usesinc
        self.dropfirst = dropfirst
        self.ignore_cache = ignore_cache

    @pyqtSlot()
    def procCounter(self): # A slot takes no params
        self.intReady.emit(0,0,0)
        cache_path = os.path.join(kda.bd0, 'k2dict.pkl')
        if not self.ignore_cache and _load_run_cache(cache_path):
            self.intReady.emit(3,0,0)
        else:
            maxgrp = kda.totGrps
            kda.readEnv()
            self.intReady.emit(2,0,0)
            Npl = int((maxgrp+1)//20)
            if Npl==0: Npl=1
            for k in range(maxgrp+1):
                kda.myVelos.readGrp(k,Vmul=1000)
                kda.myOns.readGrp(k)
                kda.myOffs.readGrp(k)
                if k>=1 and k%Npl==0:
                    kda.myVelos.fitMe(order=self.order,usesinc=self.usesinc)
                    kda.myOns.aveForce()
                    kda.myOffs.aveForce()
                if k>=1 and k%Npl==0:
                    kda.calcMass(dropfirst=self.dropfirst)
                if k>Npl:
                    self.intReady.emit(1,k+1,maxgrp+1)
            _save_run_cache(cache_path)
        kda.myVelos.fitMe(order=self.order,usesinc=self.usesinc)
        kda.myOns.aveForce()
        kda.myOffs.aveForce()
        kda.calcMass(excl3=self.excl3,dropfirst=self.dropfirst)

        self.intReady.emit(99,0,0)

        self.finished.emit()
        
def _save_run_cache(path):
    try:
        d = {
            'app_version':      APP_VERSION,
            'velos_adata':      np.array(kda.myVelos.adata),
            'velos_maxGrpMem':  kda.myVelos.maxGrpMem,
            'ons_data':         np.array(kda.myOns.data),
            'ons_maxS':         kda.myOns.maxS,
            'ons_maxgrp':       kda.myOns.maxgrp,
            'ons_maxGrpMem':    kda.myOns.maxGrpMem,
            'offs_data':        np.array(kda.myOffs.data),
            'offs_maxS':        kda.myOffs.maxS,
            'offs_maxgrp':      kda.myOffs.maxgrp,
            'offs_maxGrpMem':   kda.myOffs.maxGrpMem,
            'env_edata':        np.array(kda.myEnv.edata),
            'env_hasEnv':       kda.myEnv.hasEnv,
        }
        with gzip.open(path, 'wb') as f:
            pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'[cache] saved {path}')
    except Exception as e:
        print(f'[cache] save failed: {e}')


def _load_run_cache(path):
    if not os.path.isfile(path):
        return False
    try:
        with gzip.open(path, 'rb') as f:
            d = pickle.load(f)
        cached_major = str(d.get('app_version', '')).split('.')[0]
        current_major = str(APP_VERSION).split('.')[0]
        if cached_major != current_major:
            print(f'[cache] major version mismatch ({d.get("app_version")} vs {APP_VERSION}), ignoring')
            return False
        kda.myVelos.adata       = d['velos_adata']
        kda.myVelos.maxGrpMem   = d['velos_maxGrpMem']
        kda.myOns.data          = d['ons_data']
        kda.myOns.maxS          = d['ons_maxS']
        kda.myOns.maxgrp        = d['ons_maxgrp']
        kda.myOns.maxGrpMem     = d['ons_maxGrpMem']
        kda.myOffs.data         = d['offs_data']
        kda.myOffs.maxS         = d['offs_maxS']
        kda.myOffs.maxgrp       = d['offs_maxgrp']
        kda.myOffs.maxGrpMem    = d['offs_maxGrpMem']
        kda.myEnv.edata         = d['env_edata']
        kda.myEnv.hasEnv        = d['env_hasEnv']
        print(f'[cache] loaded {path}')
        return True
    except Exception as e:
        print(f'[cache] load failed: {e}')
        return False


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    updateAvailable = pyqtSignal(str, str)  # latest version, download url

    def __init__(self):
        super().__init__()
        self.updateAvailable.connect(self._show_update_dialog)
        self.foBig     = QFont('Arial',10)
        self.foBigBold = QFont('Arial',10)
        self.foBigBold.setBold(True)

        

        
        self.setWindowIcon(QIcon('k2viewer.png'))
        cfg = AppConfig()
        self.bd = cfg.datapath
        self.balance_name = cfg.balance_name

        self.idle=True
        self.statust=time.time()
        self._table_thread = None
        self._table_loader = None
        self._table_loading = False

        self.thread = QThread()
        title_balance = f"  [{self.balance_name}]" if self.balance_name else ""
        self.setWindowTitle(f"Kibb-g2 Viewer  v{APP_VERSION}{title_balance}")
        self.setMinimumSize(QSize(1300, 600))
        
        self._tab_balances = []
        self._tab_tables = []
        self._populated = set()
        self._last_tab_index = 0
        self._loading_table = None

        self.balanceTabs = QTabWidget()
        self.balanceTabs.setMinimumWidth(400)
        self.balanceTabs.setMaximumWidth(600)
        self.balanceTabs.tabBar().setContextMenuPolicy(Qt.CustomContextMenu)
        self.balanceTabs.tabBar().customContextMenuRequested.connect(self._on_tab_context_menu)
        self.balanceTabs.tabBar().tabBarClicked.connect(self._on_tab_bar_clicked)
        self._build_tabs()

        self.Brefresh = QPushButton("Reload")
        
        ### The plot windows
        
        self.mplfor      = mplwidget.MplWidget2()
        self.mplenv      = mplwidget.MplWidget4()
        self.mplvel      = mplwidget.MplWidget(rightax=True)
        self.mplmass     = mplwidget.MplWidget(rightax=True)
        self.mplprofile  = mplwidget.MplWidget(rightax=True)
        
        ### check and spin boxes
        
        self.cbShowVolt    = QCheckBox()
        self.cbUseSync     = QCheckBox()
        self.cbMvsZ        = QCheckBox()
        self.cbExc3sig     = QCheckBox()
        self.cbIgnoreCache = QCheckBox()
        self.cbShowPpm     = QCheckBox()
        self.cbForceScan   = QCheckBox()
        self.rbDrop0     = QRadioButton('drop 0')
        self.rbDrop1     = QRadioButton('drop 1')
        self.rbDrop2     = QRadioButton('drop 2')
        self.rgDrop      = QButtonGroup()
        self.rgDrop.addButton(self.rbDrop0, 0)
        self.rgDrop.addButton(self.rbDrop1, 1)
        self.rgDrop.addButton(self.rbDrop2, 2)
        self.sbOrder     = QSpinBox()
        self.sbMass      = QDoubleSpinBox()
        self.sbOrder.setMinimum(1)
        self.sbOrder.setMaximum(10)
        self.sbOrder.setValue(cfg.fit_order)
        self.cbUseSync.setChecked(cfg.use_sinc)
        self.rgDrop.button(max(0, min(2, cfg.drop_n))).setChecked(True)
        self.cbExc3sig.setChecked(True)
        self.cbForceScan.setChecked(cfg.force_scan)
      
        self.sbMass.setMinimumWidth(100)
        self.sbMass.setMinimum(0)
        self.sbMass.setMaximum(99999)
        self.sbMass.setDecimals(4)
        self.sbMass.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.sbMass.setKeyboardTracking(False)
        if cfg.ref_mass != 0:
            self.sbMass.setValue(cfg.ref_mass)
        
        ### global labels
        
        self.sblabel  = QLabel("Click on a run") 
        self.lares    = QLabel("") 
        self.laUncTot = QLabel("n/a")
        self.laMass   = QLabel("")
        self.laUnc    = QLabel("")
        self.laUncB    = QLabel("")
        self.laMass2   = QLabel("")
        self.laTotUnc  = QLabel("")
        self.lacov     = QLabel("(k={0})".format(kda.covk))
        self.laNoEnv   = QLabel("")

        
        self.laUa= []
        self.laUaMaxRows =8
        self.laUaMaxCols =3
        for i in range(self.laUaMaxRows):
            row=[]
            for j in range(self.laUaMaxCols):
                row.append( QLabel(""))
            self.laUa.append(row)
        for i in range(self.laUaMaxRows):
            for j in range(self.laUaMaxCols):
                self.laUa[i][j].setText('')

                
        self.Uncdict ={'Resistance':0.21,'Voltage': 1.0,'Mass position': 1.0,
                       'g': 2.0, 'Verticality': 0.5,
                       'Type A': -1, 'Total': -1, 'Balance mechanics': -1}
        
        
        self.resultLabels=[
            'Serial Number',
            'Weight designation',
            'True mass',
            'Assumed density',
            'Conventional mass',
            'Deviation from nominal',
            'Total uncertainy',
            'Tolerance (Class 3)',
            'Temperature',
            'Barometric pressure',
            'Humidity'\
            ]
        self.laResult=[]

        for i in self.resultLabels:
            row=[]
            for j in range(3):
                if j==0:
                    la =QLabel(i)
                else:
                    la=QLabel('')
                    la.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                la.setFont(self.foBig)
                row.append(la )
            self.laResult.append(row)
                
        
        self.laMass.setFont(self.foBigBold)
        self.laUnc.setFont(self.foBigBold)
        self.laUncB.setFont(self.foBigBold)
        
        ### Status bar

        self.progressBar = QProgressBar()
        self.progressBar.setMaximumWidth(400)
        self.tabWidget = MyTabWidget(self) 
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        widget = QWidget(self)
        widget.setLayout(QHBoxLayout())
        widget.layout().addWidget(self.sblabel)
        widget.layout().addWidget(self.progressBar)
   
        #self.statusBar.setStyleSheet("border 
        
        self.statusBar.addPermanentWidget(widget) 
        
        
        
        self.statusBar.showMessage('Welcome to k2viewer',5000)

        self.diagStream = DiagStream()
        self.diagStream.newText.connect(self.appendDiag)
        sys.stdout = self.diagStream

        layout   = QVBoxLayout()
        hlayout  = QHBoxLayout()
        hlayout2  = QHBoxLayout()
        vlayout1 = QVBoxLayout()
        vlayout2 = QVBoxLayout()

        layout.setContentsMargins(10, 10, 10, 6)
        layout.setSpacing(6)
        hlayout.setSpacing(10)
        vlayout1.setSpacing(6)
        vlayout2.setSpacing(4)
        hlayout2.setSpacing(10)
        hlayout2.setContentsMargins(0, 2, 0, 2)

        widget = QWidget()
        widget.setLayout(layout)
        layout.addLayout(hlayout)
        hlayout.addLayout(vlayout1)
        hlayout.addLayout(vlayout2)

        vlayout2.addLayout(hlayout2)
        vlayout2.addWidget(self.tabWidget)

        vlayout1.addWidget(self.balanceTabs)
        vlayout1.addWidget(self.Brefresh)

        hlayout2.addWidget(QLabel("order"))
        hlayout2.addWidget(self.sbOrder)
        hlayout2.addWidget(self.cbUseSync)
        hlayout2.addWidget(QLabel('use sinc'))
        hlayout2.addWidget(self.rbDrop0)
        hlayout2.addWidget(self.rbDrop1)
        hlayout2.addWidget(self.rbDrop2)
        hlayout2.addWidget(self.cbIgnoreCache)
        hlayout2.addWidget(QLabel('ignore cache'))
        hlayout2.addWidget(self.cbForceScan)
        hlayout2.addWidget(QLabel('force scan all'))

        hSpacer = QSpacerItem(20, 2, QSizePolicy.Expanding, QSizePolicy.Minimum)
        hlayout2.addItem(hSpacer)
     
        self.setCentralWidget(widget)
        self._update_sblabel()
        QTimer.singleShot(0, self.loadTable)

        self.Brefresh.clicked.connect(self.loadTable)
        self.balanceTabs.currentChanged.connect(self._on_tab_changed)
        self.cbShowVolt.clicked.connect(self.plotForce)
        self.cbUseSync.clicked.connect(self.recalcvelo)
        self.rgDrop.buttonClicked.connect(self.recalcvelo)
        self.cbMvsZ.clicked.connect(self.plotMass)
        self.cbExc3sig.clicked.connect(self.plotMass)
        self.cbShowPpm.clicked.connect(self.plotMass)
        self.tabWidget.tabs.currentChanged.connect(self.replot)
        self.sbOrder.valueChanged.connect(self.recalcvelo)
        self.sbMass.valueChanged.connect(self.gotmassval)
        self.sbOrder.valueChanged.connect(self._save_ui_settings)
        self.cbUseSync.clicked.connect(self._save_ui_settings)
        self.rgDrop.buttonClicked.connect(self._save_ui_settings)
        self.cbForceScan.clicked.connect(self._save_force_scan)
        self.cbForceScan.clicked.connect(self.loadTable)
        

    def appendDiag(self, text):
        self.tabWidget.diagText.appendPlainText(text)
        self.tabWidget.diagText.verticalScrollBar().setValue(
            self.tabWidget.diagText.verticalScrollBar().maximum())

    def createdb(self):
        connection = sqlite3.connect('k2viewer.db')
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE k2data (run TEXT, balance TEXT, value FLOAT,
            uncertainty FLOAT, title TEXT, time TIMESTAMP, airdens FLOAT,
            UNIQUE(run, balance))""")
        connection.close()

    def _migratedb(self):
        connection = sqlite3.connect('k2viewer.db')
        cursor = connection.cursor()
        cols = [r[1] for r in cursor.execute("PRAGMA table_info(k2data)").fetchall()]
        if 'balance' not in cols:
            cursor.execute("ALTER TABLE k2data ADD COLUMN balance TEXT")
            connection.commit()
        connection.close()



    @staticmethod
    def _find_dataroot(path):
        """Return the directory that contains YYMM (4-digit) subfolders.

        Resolution order:
          1. The selected path itself.
          2. A 'data' subdirectory of the selected path.
          3. Walk up toward the root (up to 5 levels).
        Falls back to the original path if nothing is found.
        """
        def _has_yymm(d):
            try:
                return any(f.is_dir() and len(f.name) == 4 and f.name.isdigit()
                           for f in os.scandir(d))
            except Exception:
                return False

        p = os.path.abspath(path)

        if _has_yymm(p):
            return p

        data_sub = os.path.join(p, 'data')
        if os.path.isdir(data_sub) and _has_yymm(data_sub):
            return data_sub

        parent = p
        for _ in range(5):
            parent = os.path.dirname(parent)
            if parent == os.path.dirname(parent):
                break
            if _has_yymm(parent):
                return parent

        return path

    def _reset_display(self):
        if self._table_thread and self._table_thread.isRunning():
            self._table_loader.abort()
            self._table_thread.quit()
            self._table_thread.wait()
        self._populated.clear()
        for t in self._tab_tables:
            t.clearContents()
            t.setRowCount(0)
        kda.clear()
        self.idle = True
        self.calcrow = -1
        self.runid = ''
        self.sbMass.clear()
        for ax in [self.mplfor.canvas.ax1, self.mplfor.canvas.ax2]:
            ax.clear(); ax.figure.canvas.draw()
        for ax in [self.mplenv.canvas.ax1, self.mplenv.canvas.ax2,
                   self.mplenv.canvas.ax3, self.mplenv.canvas.ax4]:
            ax.clear(); ax.figure.canvas.draw()
        for widget in [self.mplvel, self.mplmass, self.mplprofile]:
            widget.canvas.ax1.clear(); widget.canvas.ax1.figure.canvas.draw()

    # --------------------------------------------------------- tab context menu
    def _on_tab_context_menu(self, pos):
        if self._check_busy():
            return
        idx = self.balanceTabs.tabBar().tabAt(pos)
        if idx < 0 or idx >= len(self._tab_balances):
            return
        menu = QMenu(self)
        act_rename = menu.addAction('Rename')
        act_delete = menu.addAction('Delete')
        act_chdir  = menu.addAction('Change Directory...')
        action = menu.exec_(self.balanceTabs.tabBar().mapToGlobal(pos))
        if action == act_rename:
            self._tab_start_rename(idx)
        elif action == act_delete:
            self._tab_delete(idx)
        elif action == act_chdir:
            self._tab_change_directory(idx)

    def _tab_start_rename(self, tab_idx):
        tab_bar = self.balanceTabs.tabBar()
        old_name = self._tab_balances[tab_idx]
        rect = tab_bar.tabRect(tab_idx)
        editor = QLineEdit(old_name, tab_bar)
        editor.setGeometry(rect)
        editor.selectAll()
        editor.show()
        editor.setFocus()

        committed = [False]

        def commit():
            if committed[0]:
                return
            committed[0] = True
            new_name = editor.text().strip()
            editor.deleteLater()
            if not new_name or new_name == old_name:
                return
            if '[' in new_name or ']' in new_name:
                QMessageBox.warning(self, 'Invalid name', 'Balance name may not contain [ or ] characters.')
                return
            if new_name in self._tab_balances:
                QMessageBox.warning(self, 'Already exists', f'A balance named "{new_name}" already exists.')
                return
            AppConfig.rename_balance(old_name, new_name)
            if os.path.isfile('k2viewer.db'):
                conn = sqlite3.connect('k2viewer.db')
                conn.execute("UPDATE k2data SET balance=? WHERE balance=?", (new_name, old_name))
                conn.commit()
                conn.close()
            self._tab_balances[tab_idx] = new_name
            self.balanceTabs.setTabText(tab_idx, new_name)
            if old_name in self._populated:
                self._populated.discard(old_name)
                self._populated.add(new_name)
            if self.balance_name == old_name:
                self.balance_name = new_name
                self.setWindowTitle(f"Kibb-g2 Viewer  v{APP_VERSION}  [{new_name}]")
            self._populated.discard(new_name)
            if self.balanceTabs.currentIndex() == tab_idx:
                self._start_table_load(tab_idx)

        def cancel():
            if not committed[0]:
                editor.deleteLater()

        editor.returnPressed.connect(commit)
        editor.editingFinished.connect(cancel)

    def _tab_delete(self, tab_idx):
        name = self._tab_balances[tab_idx]
        reply = QMessageBox.warning(
            self, 'Confirm Delete',
            f'Delete balance "{name}"?\n\nThis will remove it from the config and '
            f'delete all its database entries. This cannot be undone.',
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
        if reply != QMessageBox.Yes:
            return
        AppConfig.delete_balance(name)
        if os.path.isfile('k2viewer.db'):
            conn = sqlite3.connect('k2viewer.db')
            conn.execute("DELETE FROM k2data WHERE balance=?", (name,))
            conn.commit()
            conn.close()
        self._tab_balances.pop(tab_idx)
        self._tab_tables.pop(tab_idx)
        self._populated.discard(name)

        self.balanceTabs.blockSignals(True)
        self.balanceTabs.removeTab(tab_idx)
        if not self._tab_balances:
            self.balanceTabs.blockSignals(False)
            self.balance_name = ''
            self.bd = ''
            self.setWindowTitle(f"Kibb-g2 Viewer  v{APP_VERSION}")
            self._update_sblabel()
            return
        new_idx = min(max(0, tab_idx - 1), len(self._tab_balances) - 1)
        self.balanceTabs.setCurrentIndex(new_idx)
        self.balanceTabs.blockSignals(False)
        self._last_tab_index = new_idx
        new_balance = self._tab_balances[new_idx]
        new_bd = AppConfig.get_all_balances().get(new_balance, '')
        self.balance_name = new_balance
        self.bd = new_bd
        self.setWindowTitle(f"Kibb-g2 Viewer  v{APP_VERSION}  [{new_balance}]")
        AppConfig.save(new_balance, new_bd, AppConfig().loglevel)
        if new_balance not in self._populated:
            self._start_table_load(new_idx)

    def _tab_change_directory(self, tab_idx):
        name = self._tab_balances[tab_idx]
        balances = AppConfig.get_all_balances()
        current_path = balances.get(name, '')
        start = current_path if os.path.isdir(current_path) else os.path.expanduser('~')
        path = QFileDialog.getExistingDirectory(
            self, f'Select new data directory for "{name}"', start)
        if not path:
            return
        path = self._find_dataroot(path)
        AppConfig.update_datapath(name, path)
        self._populated.discard(name)
        if name == self.balance_name:
            self.bd = path
            self._reset_display()
        if self.balanceTabs.currentIndex() == tab_idx:
            self._start_table_load(tab_idx)

    def _menu_add_balance(self):
        start = self.bd if os.path.isdir(self.bd) else os.path.expanduser('~')
        path = QFileDialog.getExistingDirectory(self, 'Select data directory for new balance', start)
        if not path:
            return
        path = self._find_dataroot(path)

        hint = ('...' + path[-40:]) if len(path) > 40 else path
        existing = AppConfig.get_all_balances()
        while True:
            name, ok = QInputDialog.getText(self, 'Add Balance',
                                            f'Data path: {hint}\n\nEnter a name for this balance\n(no [ or ] characters allowed):',
                                            text='Balance1')
            if not ok:
                return
            name = name.strip()
            if not name:
                QMessageBox.warning(self, 'Invalid name', 'Balance name cannot be empty.')
                continue
            if '[' in name or ']' in name:
                QMessageBox.warning(self, 'Invalid name', 'Balance name may not contain [ or ] characters.')
                continue
            if name in existing:
                QMessageBox.warning(self, 'Already exists', f'A balance named "{name}" already exists.')
                continue
            break

        AppConfig.save(name, path, AppConfig().loglevel)
        new_idx = self._add_tab(name)
        self.balanceTabs.blockSignals(True)
        self.balanceTabs.setCurrentIndex(new_idx)
        self.balanceTabs.blockSignals(False)
        self._last_tab_index = new_idx
        self.balance_name = name
        self.bd = path
        self.setWindowTitle(f"Kibb-g2 Viewer  v{APP_VERSION}  [{name}]")
        self._start_table_load(new_idx)
        self._update_sblabel()

    def _save_ui_settings(self, *_):
        AppConfig.save_ui_settings(
            int(self.sbOrder.value()),
            self.cbUseSync.isChecked(),
            self._dropN())

    def _save_force_scan(self, *_):
        AppConfig.save_force_scan(self.cbForceScan.isChecked())

    def _prompt_datapath(self):
        if not self.bd:
            msg = 'No balance configured yet.\n\nPlease select its data directory.'
        else:
            msg = (f'The configured data path does not exist:\n\n{self.bd}\n\n'
                   'Please select the data directory.')
        QMessageBox.information(self, 'Balance setup', msg)

        start = self.bd if os.path.isdir(self.bd) else os.path.expanduser('~')
        path = QFileDialog.getExistingDirectory(self, 'Select data directory', start)
        if not path:
            return
        path = self._find_dataroot(path)

        hint = ('...' + path[-40:]) if len(path) > 40 else path
        existing = AppConfig.get_all_balances()
        while True:
            name, ok = QInputDialog.getText(self, 'Balance name',
                                            f'Data path: {hint}\n\nEnter a name for this balance\n(no [ or ] characters allowed):',
                                            text=self.balance_name or 'Balance1')
            if not ok:
                return
            name = name.strip()
            if not name:
                QMessageBox.warning(self, 'Invalid name', 'Balance name cannot be empty.')
                continue
            if '[' in name or ']' in name:
                QMessageBox.warning(self, 'Invalid name', 'Balance name may not contain [ or ] characters.')
                continue
            if name in existing and name != self.balance_name:
                QMessageBox.warning(self, 'Already exists', f'A balance named "{name}" already exists.')
                continue
            break

        self.bd = path
        self.balance_name = name
        AppConfig.save_datapath(name, path)

    # ------------------------------------------------------------------ tabs --
    def _make_table(self):
        t = QTableWidget(0, 4)
        t.setHorizontalHeaderItem(0, QTableWidgetItem("run"))
        t.setHorizontalHeaderItem(1, QTableWidgetItem("title"))
        t.setHorizontalHeaderItem(2, QTableWidgetItem("mass/mg"))
        t.setHorizontalHeaderItem(3, QTableWidgetItem("unc/ug"))
        t.setColumnWidth(0, 80)
        t.setColumnWidth(2, 80)
        t.setColumnWidth(3, 80)
        t.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        t.setAlternatingRowColors(True)
        t.setSelectionBehavior(QTableWidget.SelectRows)
        t.verticalHeader().setVisible(False)
        t.clicked.connect(self.on_table_clicked)
        return t

    def _build_tabs(self):
        balances = AppConfig.get_all_balances()
        active = self.balance_name
        active_idx = 0
        for i, (name, _) in enumerate(balances.items()):
            table = self._make_table()
            self.balanceTabs.addTab(table, name)
            self._tab_balances.append(name)
            self._tab_tables.append(table)
            if name == active:
                active_idx = i
        self.balanceTabs.addTab(QWidget(), '+')
        self.balanceTabs.blockSignals(True)
        self.balanceTabs.setCurrentIndex(active_idx)
        self.balanceTabs.blockSignals(False)
        self._last_tab_index = active_idx

    def _add_tab(self, name):
        table = self._make_table()
        plus_idx = self.balanceTabs.count() - 1
        self.balanceTabs.insertTab(plus_idx, table, name)
        self._tab_balances.insert(plus_idx, name)
        self._tab_tables.insert(plus_idx, table)
        return plus_idx  # new tab's index

    def _current_table(self):
        idx = self.balanceTabs.currentIndex()
        if idx < len(self._tab_tables):
            return self._tab_tables[idx]
        return None

    def _set_table_loading(self, loading):
        self._table_loading = loading
        self.Brefresh.setEnabled(not loading)

    def _check_busy(self):
        """Return True and show a status message if any background task is running."""
        if self._table_loading:
            self.statusBar.showMessage('Wait until table is populated', 3000)
            return True
        if not self.idle:
            self.statusBar.showMessage('Wait until run is loaded', 3000)
            return True
        return False

    def _update_sblabel(self):
        if self._tab_balances:
            self.sblabel.setText('Click on a run')
        else:
            self.sblabel.setText('Click on + to add a balance with data directory')

    def _on_tab_bar_clicked(self, index):
        if self._check_busy():
            return
        if index == self.balanceTabs.count() - 1:
            self._menu_add_balance()

    def _on_tab_changed(self, index):
        if index >= len(self._tab_balances):
            return  # "+" tab
        if self._check_busy():
            self.balanceTabs.blockSignals(True)
            self.balanceTabs.setCurrentIndex(self._last_tab_index)
            self.balanceTabs.blockSignals(False)
            return
        self._last_tab_index = index
        balance = self._tab_balances[index]
        balances = AppConfig.get_all_balances()
        self.balance_name = balance
        self.bd = balances.get(balance, '')
        self.setWindowTitle(f"Kibb-g2 Viewer  v{APP_VERSION}  [{balance}]")
        AppConfig.save(balance, self.bd, AppConfig().loglevel)
        if balance not in self._populated:
            self._start_table_load(index)

    # --------------------------------------------------------------- loading --
    def _start_table_load(self, tab_index):
        if tab_index >= len(self._tab_balances):
            return
        balance = self._tab_balances[tab_index]
        balances = AppConfig.get_all_balances()
        bd = balances.get(balance, '')
        if not bd or not os.path.isdir(bd):
            self.statusBar.showMessage(f'Data path not configured for {balance}', 4000)
            return

        table = self._tab_tables[tab_index]
        table.clearContents()
        table.setRowCount(0)

        if not os.path.isfile('k2viewer.db'):
            self.createdb()
        self._migratedb()

        if self._table_thread and self._table_thread.isRunning():
            self._table_loader.abort()
            self._table_thread.quit()
            self._table_thread.wait()

        self._loading_table = table
        self._set_table_loading(True)
        self._table_loader = TableLoader(bd, balance, list(balances.keys()),
                                         force_scan=self.cbForceScan.isChecked())
        self._table_thread = QThread()
        self._table_loader.moveToThread(self._table_thread)
        self._table_loader.activeRow.connect(self._on_table_active_row)
        self._table_loader.finished.connect(self._table_thread.quit)
        self._table_loader.finished.connect(lambda: self._populated.add(balance))
        self._table_loader.finished.connect(lambda: self._set_table_loading(False))
        self._table_thread.started.connect(self._table_loader.run)
        self._table_thread.start()

    def loadTable(self):
        if self._check_busy():
            return
        idx = self.balanceTabs.currentIndex()
        if idx >= len(self._tab_balances):
            return
        self._populated.discard(self._tab_balances[idx])
        self._start_table_load(idx)

    @pyqtSlot(str, str, str, str, str)
    def _on_table_active_row(self, balance, run, title, mass_str, unc_str):
        t = self._loading_table
        if t is None:
            return
        row = t.rowCount()
        t.insertRow(row)
        t.setItem(row, 0, QTableWidgetItem(run))
        t.setItem(row, 1, QTableWidgetItem(title))
        if mass_str:
            nitem = QTableWidgetItem(mass_str)
            nitem.setTextAlignment(int(Qt.AlignRight | Qt.AlignVCenter))
            t.setItem(row, 2, nitem)
        if unc_str:
            nitem = QTableWidgetItem(unc_str)
            nitem.setTextAlignment(int(Qt.AlignRight | Qt.AlignVCenter))
            t.setItem(row, 3, nitem)
         
    def plotForce(self):
        if kda.myOffs.maxGrpMem<0:
            return
        self.mplfor.canvas.ax1.clear()
        self.mplfor.canvas.ax2.clear()
        if self.cbShowVolt.isChecked():
            scale = kda.myOffs.c.R/1000
            yla1='U(on)/mV'
            yla2='U(off)/mV'          
        else:
            scale =1
            yla1='I(on)/uA'
            yla2='I(off)/uA'
        mutex.lock()
        tmul,tla = kda.tmul()
        # if kda.myOns.maxS>=1:
        #     p1,=self.mplfor.canvas.ax1.plot(
        #         kda.myOns.data[:,0]*tmul,\
        #         kda.myOns.data[:,2]-mon,'r.')
        # if kda.myOffs.maxS>=1:
        #     p2,=self.mplfor.canvas.bx1.plot(\
        #         kda.myOffs.data[:,0]*tmul,\
        #         kda.myOffs.data[:,2]-mof,'b.')
        def _drop_first(adata, n):
            if n == 0 or len(adata) == 0:
                return adata
            remaining = adata[n:]
            return remaining if len(remaining) else adata

        drop_n = self._dropN()
        if kda.myOns.adatalen>0:
            d = _drop_first(kda.myOns.adata, drop_n)
            if len(d):
                _=self.mplfor.canvas.ax1.errorbar(
                    d[:,0]*tmul, d[:,2]*scale, d[:,7]*scale, fmt='ro')
        if kda.myOffs.adatalen>0:
            d = _drop_first(kda.myOffs.adata, drop_n)
            if len(d):
                _=self.mplfor.canvas.ax2.errorbar(
                    d[:,0]*tmul, d[:,2]*scale, d[:,7]*scale, fmt='bs')
            
        self.mplfor.canvas.setsamexscale()
        mutex.unlock()
        self.mplfor.canvas.ax1.set_ylabel(yla1)
        self.mplfor.canvas.ax2.set_ylabel(yla2)
        self.mplfor.canvas.ax2.set_xlabel(tla)
        self.mplfor.canvas.draw() 
            
    def plotEnv(self):
        self.mplenv.canvas.ax1.clear()
        self.mplenv.canvas.ax2.clear()
        self.mplenv.canvas.ax3.clear()
        self.mplenv.canvas.ax4.clear()
        if kda.myEnv.hasEnv==False:
            return
        mutex.lock()
        tmul,tla = kda.tmul()
        self.mplenv.canvas.ax1.plot(kda.myEnv.edata [:,0]*tmul,\
                                    kda.myEnv.edata[:,1],'r.')
        self.mplenv.canvas.ax2.plot(kda.myEnv.edata [:,0]*tmul,\
                                    kda.myEnv.edata[:,2],'b.')
        self.mplenv.canvas.ax3.plot(kda.myEnv.edata [:,0]*tmul,\
                                    kda.myEnv.edata[:,3],'g.')
        self.mplenv.canvas.ax4.plot(kda.myEnv.edata [:,0]*tmul,\
                                    kda.myEnv.edata[:,4],'m.')
        mutex.unlock()
        self.mplenv.canvas.ax1.ticklabel_format(useOffset=False)
        self.mplenv.canvas.ax2.ticklabel_format(useOffset=False)
        self.mplenv.canvas.ax3.ticklabel_format(useOffset=False)
        self.mplenv.canvas.ax4.ticklabel_format(useOffset=False)
        self.mplenv.canvas.ax1.xaxis.set_ticklabels([])
        self.mplenv.canvas.ax3.xaxis.set_ticklabels([])
        self.mplenv.canvas.ax1.tick_params(axis='x',direction='inout')
        self.mplenv.canvas.ax2.tick_params(axis='x',direction='inout')
        self.mplenv.canvas.ax3.tick_params(axis='x',direction='inout')
        self.mplenv.canvas.ax4.tick_params(axis='x',direction='inout')
        self.mplenv.canvas.ax1.set_ylabel('rel. humid (%)')
        self.mplenv.canvas.ax2.set_ylabel('press. (hPa)')
        self.mplenv.canvas.ax3.set_ylabel('temp (degC)')
        self.mplenv.canvas.ax4.set_ylabel('air dens. (kg/m^3)')
        self.mplenv.canvas.ax2.set_xlabel(tla)  
        self.mplenv.canvas.ax4.set_xlabel(tla)  
        self.mplenv.canvas.draw() 


    def plotVelocity(self):
        self.mplvel.canvas.ax1.clear()
        mutex.lock()
        tmul,tla = kda.tmul()
        if len(kda.myVelos.blfit)>0:
            self.mplvel.canvas.ax1.plot(
                kda.myVelos.blfit[:,0]*tmul,\
                kda.myVelos.blfit[:,1],'b.')
            if kda.myVelos.maxgrp>=1:
                tt = np.linspace(kda.myVelos.tmin,kda.myVelos.tmax,400)
                val,unc =  kda.myVelos.getBlAndUnc(tt)
                self.mplvel.canvas.ax1.plot(
                    tt*tmul,\
                    val,'k-')
                self.mplvel.canvas.ax1.fill_between(
                    tt*tmul,val-unc,val+unc,fc='r',alpha=0.1)
            
        
        #tmul,tla = kda.tmul()
        
#        if len(kda.vblv)>=2:
 #           p1,=self.mplvel.canvas.ax1.plot(
  #              kda.vblt*tmul,\
   #             np.abs(kda.vblv)*1e3,'b.')
    #        p2,=self.mplvel.canvas.ax1.plot(
     #           kda.sblt*tmul,\
      #          np.abs(kda.sblv)*1e3,'r-')
        mutex.unlock()
        self.mplvel.canvas.bx1.ticklabel_format(useOffset=False)
        self.mplvel.canvas.ax1.ticklabel_format(useOffset=False)        
        be,en =self.mplvel.canvas.ax1.get_ylim()
        self.mplvel.canvas.draw() 
        me =0.5*(be+en)
        self.mplvel.canvas.bx1.set_ylim((be/me-1)*1e6,(en/me-1)*1e6)
        self.mplvel.canvas.ax1.set_xlabel(tla)    
        self.mplvel.canvas.ax1.set_ylabel('Bl/Tm')
        self.mplvel.canvas.bx1.set_ylabel('rel. change /ppm')
        self.mplvel.canvas.draw() 

    def plotMass(self):
        if kda.Mass==0:
            return
        kda.calcMass(excl3=self.cbExc3sig.isChecked(),dropfirst=self._dropN())
        if kda.Mass==0:
            return
        self.populateUnc()
        self.mplmass.canvas.ax1.clear()
        mutex.lock()
        tmul,tla = kda.tmul()
        if self.cbMvsZ.isChecked()==False:
            self.mplmass.canvas.ax1.errorbar(
                kda.Mass.dif_d[:,0]*tmul,kda.Mass.dif_d[:,2],\
                    kda.Mass.dif_d[:,3],fmt='bo')
        else:
            self.mplmass.canvas.ax1.errorbar(
                kda.Mass.dif_d[:,1],kda.Mass.dif_d[:,2],\
                    kda.Mass.dif_d[:,3],fmt='bo')
            tla='z/mm'
        mutex.unlock()
        be,en =self.mplmass.canvas.ax1.get_xlim()
        mean = kda.Mass.avemass
        sig  =  kda.Mass.uncmass
        sigB = self.totuncB/1000
        sigAll = self.totuncabs/1000
        self.laMass.setText('{0:8.4f} mg'.format(mean))
        self.laUnc.setText('\u00B1 {0:4.1f} \u00B5g (Type A)'.format(sig*1000))
        self.laUncB.setText('\u00B1 {0:4.1f} \u00B5g (Type B)'.format(sigB*1000))
        if kda.myEnv.hasEnv==2:
            self.laNoEnv.setText('No Env data available')
        else:
            self.laNoEnv.setText('')
            
        self.mplmass.canvas.ax1.plot((be,en),\
                        (mean,mean),c='k',linestyle='dashed',lw=2)
        self.mplmass.canvas.ax1.plot((be,en),(mean+sig,mean+sig),\
                                      c='r',linestyle='dashdot')
        self.mplmass.canvas.ax1.plot((be,en),(mean-sig,mean-sig),\
                                      c='r',linestyle='dashdot')
      
        self.mplmass.canvas.ax1.plot((be,en),(mean-sigAll,mean-sigAll),\
                                      c='b',linestyle='dashdot')
      
        self.mplmass.canvas.ax1.plot((be,en),(mean+sigAll,mean+sigAll),\
                                      c='b',linestyle='dashdot')
        self.mplmass.canvas.ax1.fill_between((be,en), (mean-sigAll,mean-sigAll),\
                                 (mean+sigAll,mean+sigAll),color='b', alpha=0.2)
      
        self.mplmass.canvas.ax1.fill_between((be,en), (mean-sig,mean-sig),\
                                 (mean+sig,mean+sig),color='r', alpha=0.2)
        self.mplmass.canvas.ax1.set_xlim(be,en)
        if kda.hasRefMass==True:
             be,en =self.mplmass.canvas.ax1.get_xlim()
             self.mplmass.canvas.ax1.plot((be,en),
                                          (kda.refMass,kda.refMass),c='m',
                                          linestyle='dotted',lw=4)
             self.mplmass.canvas.ax1.set_xlim(be,en)
            
        be,en =self.mplmass.canvas.ax1.get_ylim()

        me = mean
        if kda.hasRefMass:
            me = kda.refMass
        if self.cbShowPpm.isChecked() and me != 0:
            self.mplmass.canvas.bx1.set_ylim((be-me)/me*1e6,(en-me)/me*1e6)
            self.mplmass.canvas.bx1.set_ylabel('deviation  /ppm')
        else:
            self.mplmass.canvas.bx1.set_ylim((be-me)*1e3,(en-me)*1e3)
            self.mplmass.canvas.bx1.set_ylabel('deviation  /\u00B5g')
        self.mplmass.canvas.ax1.set_xlabel(tla)
        self.mplmass.canvas.ax1.set_ylabel('mass /mg')
        self.mplmass.canvas.bx1.ticklabel_format(useOffset=False)
        self.mplmass.canvas.ax1.ticklabel_format(useOffset=False)
        self.mplmass.canvas.draw() 
        
        
    def plotProfile(self):
        if kda.myVelos.maxGrpMem<0:
            return
        if len(kda.myVelos.fit_pars)<2:
            return
        self.mplprofile.canvas.ax1.clear()
        z=np.linspace(-1.5,1.5,200)
        bl=k2tools.calcProfile(kda.myVelos.fit_pars,kda.myVelos.order,\
            z,kda.myVelos.zmin,kda.myVelos.zmax,withOffset=True)
        self.mplprofile.canvas.ax1.plot(z,bl,'r-')
        be,en =self.mplprofile.canvas.ax1.get_ylim()
        me =0.5*(be+en)
        self.mplprofile.canvas.bx1.set_ylim((be/me-1)*1e6,(en/me-1)*1e6)
        self.mplprofile.canvas.ax1.set_ylabel('Bl (Tm)')
        self.mplprofile.canvas.bx1.set_ylabel('rel. change /ppm')
        self.mplprofile.canvas.ax1.set_xlabel('z (mm)')
        self.mplprofile.canvas.draw() 
        
    def populateUnc(self):
        if kda.Mass==0:
            return
        mean = kda.Mass.avemass
        sig  =  kda.Mass.uncmass
        self.Uncdict['Balance mechanics']=1e-3/mean*1e6
        self.Uncdict['Type A']=sig/mean*1e6/kda.covk
        cumAll=0
        cumB=0
        for k,v in self.Uncdict.items():
            if k=='Total':
                continue
            if k!='Type A':
                cumB += v*v
            cumAll+=v*v
        self.Uncdict['Total'] = np.sqrt(cumAll)
        self.totuncrel = np.sqrt(cumAll)
        self.totuncB = np.sqrt(cumB)*1e-6*mean*1000*kda.covk
        self.totuncabs = np.sqrt(cumAll)*1e-6*mean*1000*kda.covk
        
        
    def plotUnc(self):
        if kda.Mass==0:
            return
        self.populateUnc()
        row=0
        mean = kda.Mass.avemass
        for cat,rel in sorted(self.Uncdict.items(),\
                              key=lambda item: item[1],reverse=True):
            if cat=='Total':
                prow =self.laUaMaxRows-1
            else:
                prow=row
                row=row+1
            self.laUa[prow][0].setText('{0:14}'.format(cat))
            self.laUa[prow][1].setText('{0:4.1f} ppm'.format(rel*kda.covk))
            self.laUa[prow][2].setText('{0:6.1f} \u00B5g'.format(rel*1e-3*mean*kda.covk))
            if cat=='Total':
                myFont=self.foBigBold
            else:
                myFont=self.foBig
            self.laUa[prow][0].setFont(myFont)
            self.laUa[prow][1].setFont(myFont)
            self.laUa[prow][2].setFont(myFont)
            
        self.laMass2.setText('{0:8.4f} mg'.format(mean))
        self.laTotUnc.setText('\u00B1  {0:6.1f} \u00B5g (k={1})'\
                              .format( self.totuncabs,kda.covk))

            
        


    def gotmassval(self,val):
        AppConfig.save_ref_mass(val)
        if val!=0:
            kda.setRefMass(val)
            self.plotMass()
        


    def _show_update_dialog(self, latest, download_url):
        import webbrowser
        reply = QMessageBox.question(
            self,
            "Update available",
            f"A new version is available: v{latest}\nYou have: v{APP_VERSION}\n\nOpen the download page?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            webbrowser.open(download_url)

    def _dropN(self):
        return self.rgDrop.checkedId()

    def recalcvelo(self):
        order = int(self.sbOrder.value() )
        if kda.myVelos.maxGrpMem>0:
            kda.myVelos.fitMe(order,usesinc=self.cbUseSync.isChecked())
            kda.calcMass(excl3=self.cbExc3sig.isChecked(),dropfirst=self._dropN())
            self.replot()

    def convmass(self,truemass,density):
        aird = 1.2
        steeld =8000
        conv = truemass*(1-aird/density+aird/steeld)
        return conv
    
    _TOLERANCES = {
        20: 0.35, 10: 0.25, 5: 0.18, 3: 0.15, 2: 0.13, 1: 0.1,
        0.5: 0.08, 0.3: 0.07, 0.2: 0.06, 0.1: 0.05,
        0.05: 0.042, 0.02: 0.035, 0.01: 0.030,
    }

    def getol(self, nom):
        """Return Class-3 tolerance in mg; nom is nominal mass in g."""
        return self._TOLERANCES.get(nom, 0)
        
    
    def WriteExcelHdr(self,fifn):
        if os.path.isfile('templ.xls'):
            copyfile('templ.xls',fifn)
            return
        book = xlwt.Workbook(encoding="utf-8")

        sheet1 = book.add_sheet("Sheet 1")

        sheet1.write(0,  0, "Serial Number")
        sheet1.write(0,  1, "Designated Weight")
        sheet1.write(0,  2, "True Mass")
        sheet1.write(0,  3, "Assumed Dens.")
        sheet1.write(0,  4, "Conv. Mass")
        sheet1.write(0,  5, "Dev. from Nom.")
        sheet1.write(0,  6, "Total Unc.")
        sheet1.write(0,  7, "Tolerance")
        sheet1.write(0,  8, "Temp.")
        sheet1.write(0 , 9, "Press.")
        sheet1.write(0, 10, "Humid.")

        sheet1.write(1,  0, "")
        sheet1.write(1,  1, "")
        sheet1.write(1,  2, "g")
        sheet1.write(1,  3, "g/cm3")
        sheet1.write(1,  4, "g")
        sheet1.write(1,  5, "mg")
        sheet1.write(1,  6, "mg")
        sheet1.write(1,  7, "mg")
        sheet1.write(1,  8, "degC")
        sheet1.write(1,  9, "mmHg")
        sheet1.write(1, 10, "%rel")
        
        book.save(fifn)
            
    def WriteExcel(self):
        if kda.Mass==0:
            return
        now = datetime.datetime.now()
        fn = 'kibbg2_{0}.xls'.format(now.strftime('%Y%m'))
        fifn=os.path.join(r'..\results',fn)
        if os.path.isfile(fifn)==False:
            self.WriteExcelHdr(fifn)
        rb = xlrd.open_workbook(fifn,formatting_info=True)
        r_sheet = rb.sheet_by_index(0) 
        r = r_sheet.nrows
        wb = xlutils.copy.copy(rb) 
        sheet = wb.get_sheet(0) 
        ser = kda.c.mydict['SerialNo']
        sheet.write(r,0,now.strftime('%m/%d/%Y'))
        sheet.write(r,1,now.strftime('%H:%M:%S'))
        sheet.write(r,2, ser)
        nom =kda.c.mydict['Nominal']
        if nom>=0.995:
            sheet.write(r,3, '{0}'.format(nom))
        else:
            sheet.write(r,3, '{0}'.format(nom))            
        m =kda.Mass.avemass
        if m>=995:
            sheet.write(r,4, '{0:10.7f}'.format(m/1000))
        else:
            sheet.write(r,4, '{0:10.9f}'.format(m/1000))
            
        conv = self.convmass(m,kda.c.dens)
        sheet.write(r,5, '{0:10.7f}'.format(kda.c.dens/1000))
        if conv>=995:
            sheet.write(r,6, '{0:10.7f}'.format(conv/1000))
        else:
            sheet.write(r,6, '{0:10.9f}'.format(conv/1000))
        sheet.write(r,7, '{0:8.4f}'.format(conv-nom*1000))
        unc =self.totuncabs
        sheet.write(r,8, '{0:6.4f}'.format(unc/1000))
        tol = self.getol(nom)            
        sheet.write(r,9, '{0:6.4f}'.format(tol/1000))
        if kda.myEnv.hasEnv==2:
            sheet.write(r,10, 'nominal')
            sheet.write(r,11, 'nominal')
            sheet.write(r,12,'nominal')
        else:
            temp = np.mean(kda.myEnv.edata[:,3])
            sheet.write(r,10, '{0:6.3f}'.format(temp))          
            press = np.mean(kda.myEnv.edata[:,2])/1.33322
            sheet.write(r,11, '{0:6.3f}'.format(press))
            hum = np.mean(kda.myEnv.edata[:,1])
            sheet.write(r,12, '{0:6.2f}'.format(hum))
        wb.save(fifn)
    
    def plotReport(self):    
        if kda.Mass==0:
            for i in self.laResult:
                i[1].setText('')
                i[2].setText('')
            return
        else:
            self.populateUnc()
            ser = kda.c.mydict['SerialNo']
            self.laResult[0][1].setText(ser)
            nom =kda.c.mydict['Nominal']
            if nom>=0.995:
                self.laResult[1][1].setText('{0}'.format(nom))
                self.laResult[1][2].setText('g')
            else:                
                self.laResult[1][1].setText('{0}'.format(nom*1000))
                self.laResult[1][2].setText('mg')
            m =kda.Mass.avemass
            if m>=995:
                self.laResult[2][1].setText('{0:10.7f}'.format(m/1000))
                self.laResult[2][2].setText('g')
            else:
                self.laResult[2][1].setText('{0:10.4f}'.format(m))
                self.laResult[2][2].setText('mg')
            conv = self.convmass(m,kda.c.dens)
            self.laResult[3][1].setText('{0:4.1f}'.format(kda.c.dens/1000))
            self.laResult[3][2].setText('g/cm\u00B3')
            if conv>=995:
                self.laResult[4][1].setText('{0:10.7f}'.format(conv/1000))
                self.laResult[4][2].setText('g')
            else:
                self.laResult[4][1].setText('{0:10.4f}'.format(conv))
                self.laResult[4][2].setText('mg')
            
            self.laResult[5][1].setText('{0:8.4f}'.format(conv-nom*1000))
            self.laResult[5][2].setText('mg')
               
            unc =self.totuncabs
            self.laResult[6][1].setText('{0:6.4f}'.format(unc/1000))
            self.laResult[6][2].setText('mg')
            tol = self.getol(nom)            
            self.laResult[7][1].setText('{0:6.4f}'.format(tol))
            self.laResult[7][2].setText('mg')

            if kda.myEnv.hasEnv==2:
                suf=' (n/a)'
            else:
                suf=''
            
            temp = np.mean(kda.myEnv.edata[:,3])
            self.laResult[8][1].setText('{0:6.3f}'.format(temp))
            self.laResult[8][2].setText('\u00b0C'+suf)
                
            press = np.mean(kda.myEnv.edata[:,2])/1.33322
            self.laResult[9][1].setText('{0:6.3f}'.format(press))
            self.laResult[9][2].setText('mm Hg'+suf)
            hum = np.mean(kda.myEnv.edata[:,1])
            self.laResult[10][1].setText('{0:6.2f}'.format(hum))
            self.laResult[10][2].setText('% rel'+suf)
            
                
        
        


    def replot(self):
        currentIndex=self.tabWidget.tabs.currentIndex()
        tat = self.tabWidget.tabs.tabText(currentIndex)
        if tat=='Force':
            self.plotForce()
        elif tat=='Environmentals':
            self.plotEnv()
        elif tat=='Velocity':
            self.plotVelocity()
        elif tat=='Mass':
            self.plotMass()
        elif tat=='Profile':
            self.plotProfile()
        elif tat=='Uncertainty':
            self.plotUnc()    
        elif tat=='Report':
            self.plotReport()
   
            
    def updateTable(self):
        if kda.Mass==0:
            return
        self.populateUnc()
        mass = kda.Mass.avemass
        massunc = self.totuncabs
        title =  kda.c.title
        title = title.replace('"', '\'')
        nitem =QTableWidgetItem('{0:,.4f}'.format(mass) )
        nitem.setTextAlignment(int(Qt.AlignRight | Qt.AlignVCenter))
        t = self._current_table()
        if t:
            t.setItem(self.calcrow, 2, nitem)
            nitem = QTableWidgetItem('{0:6.4f}'.format(massunc))
            nitem.setTextAlignment(int(Qt.AlignRight | Qt.AlignVCenter))
            t.setItem(self.calcrow, 3, nitem)
            t.setItem(self.calcrow, 1, QTableWidgetItem(title))

        connection = sqlite3.connect('k2viewer.db')
        cursor = connection.cursor()
        cursor.execute(
            "REPLACE INTO k2data (run,balance,value,uncertainty,title) VALUES (?,?,?,?,?)",
            (self.runid, self.balance_name, mass, massunc, title))
        connection.commit()
        connection.close()    
   
    
   
    def readStatus(self,ix,cur,tot):
        if ix==0: 
            self.start=time.time()
            return
        elif ix==1:
            self.progressBar.setValue(int(100*cur/tot))
            self.sblabel.setText('reading {0} sets'.format(tot))
        elif ix==2:
            self.progressBar.setValue(0)
            self.sblabel.setText('reading Environmentals')
        elif ix==3:
            self.progressBar.setValue(0)
            self.sblabel.setText('Loading from cache...')
        elif ix==99:
            self.sblabel.setText('reading of {0} done'.format(kda.bd0))
            self.statusBar.showMessage('Data available',5000)
            self.progressBar.setValue(0)
            self.populateUnc()
            self.updateTable()
            try:
                self.WriteExcel()
            except Exception:
                pass
            ref = self.sbMass.value()
            if ref != 0 and kda.Mass != 0:
                if abs(kda.Mass.avemass - ref) / ref > 0.001:
                    self.sbMass.setValue(0)
                    AppConfig.save_ref_mass(0)
                else:
                    kda.setRefMass(ref)
            self.replot()
            self.idle=True
            return

        self.replot()
#           
            
        

    def on_table_clicked(self, item):
        if self._table_loading:
            self.statusBar.showMessage('Wait until table is populated', 3000)
            return
        t = self._current_table()
        if t is None:
            return
        if self.idle:
            self.idle = False
            self.calcrow = item.row()
            self.runid = t.item(item.row(), 0).text()
            filePath = os.path.join(self.bd,self.runid[0:4],self.runid[4:6],self.runid[6])
            
            kda.clear()
            kda.setbd0(filePath)
            order = int(self.sbOrder.value() )
            usesinc=self.cbUseSync.isChecked()
            excl3=self.cbExc3sig.isChecked()
            dropfirst=self._dropN()

            self.obj = Worker(excl3,order,usesinc,dropfirst,
                              ignore_cache=self.cbIgnoreCache.isChecked())  # no parent!
            self.thread = QThread()  # no parent!
            self.obj.intReady.connect(self.readStatus)
            self.obj.moveToThread(self.thread)
            self.obj.finished.connect(self.thread.quit)
            self.thread.started.connect(self.obj.procCounter)
            self.thread.start()
        else:
            self.statusBar.showMessage('Wait till last run is processed',5000)
         
class MyTabWidget(QWidget): 
    def __init__(self, parent): 
        super(QWidget, self).__init__(parent)  
        self.layout = QVBoxLayout(self) 
        # Initialize tab screen 
        self.tabs     = QTabWidget() 
        self.tabForce = QWidget() 
        self.tabEnv   = QWidget() 
        self.tabVelo  = QWidget() 
        self.tabMass  = QWidget() 
        self.tabUnc     = QWidget() 
        self.tabProfile = QWidget() 
        self.tabReport = QWidget() 
        self.tabs.resize(300, 200) 
        # Add tabs 
  
        # Create Force tab 
        self.tabForce.layout = QHBoxLayout()
        tab1ctrl = QVBoxLayout()
        l1 = QLabel() 
        l1.setText("Forcemode") 
        h1 = QHBoxLayout()
        l2 = QLabel() 
        l2.setText("show voltage")
        h1.addWidget(l2)
        h1.addWidget(parent.cbShowVolt)   
        verticalSpacer = QSpacerItem(20, 40, 
                                     QSizePolicy.Minimum, 
                                     QSizePolicy.Expanding)

        tab1ctrl.addWidget(l1)
        tab1ctrl.addLayout(h1)
        tab1ctrl.addItem(verticalSpacer)
        
        
        self.tabForce.layout.addLayout(tab1ctrl)
        self.tabForce.layout.addWidget(parent.mplfor) 
        self.tabForce.setLayout(self.tabForce.layout) 
        
        # Create Env tab 
        self.tabEnv.layout = QHBoxLayout()
        self.tabEnv.layout.addWidget(parent.mplenv)
        self.tabEnv.setLayout(self.tabEnv.layout) 
    
        # Create tVelo tab 
        self.tabVelo.layout = QHBoxLayout()
        tabVeloctrl = QVBoxLayout()
        l1 = QLabel() 
        l1.setText("Velocitymode") 
        h1 = QHBoxLayout()      
        verticalSpacer = QSpacerItem(20, 40, 
                                     QSizePolicy.Minimum, 
                                     QSizePolicy.Expanding)
        tabVeloctrl.addWidget(l1)
        tabVeloctrl.addLayout(h1)
        tabVeloctrl.addItem(verticalSpacer)
        self.tabVelo.layout.addLayout(tabVeloctrl)
        self.tabVelo.layout.addWidget(parent.mplvel) 
        self.tabVelo.setLayout(self.tabVelo.layout) 

        # Create Mass tab 
        self.tabMass.layout = QHBoxLayout()
        tabMassctrl = QVBoxLayout()
        l4b = QLabel() 
        l4b.setText("ref. mass (mg):")    
        verticalSpacer = QSpacerItem(20, 40, 
                                     QSizePolicy.Minimum, 
                                     QSizePolicy.Expanding)
        l4a = QLabel() 
        l4a.setText("mass")
        h1 = QHBoxLayout()

        h1.addWidget(parent.cbExc3sig)
        h1.addWidget(QLabel('Excl. outlier'))
        h2 = QHBoxLayout()
        h2.addWidget(parent.cbShowPpm)
        h2.addWidget(QLabel('show ppm'))
        tabMassctrl.addWidget(l4a)
        tabMassctrl.addLayout(h1)
        tabMassctrl.addLayout(h2) 

        tabMassctrl.addWidget(l4b)
        tabMassctrl.addWidget(parent.sbMass)
        tabMassctrl.addWidget(QLabel("Measured Mass:")) 
        tabMassctrl.addWidget(parent.laMass)
        tabMassctrl.addWidget(parent.laUnc)
        tabMassctrl.addWidget(parent.laUncB)
        tabMassctrl.addWidget(parent.lacov)
        tabMassctrl.addWidget(parent.laNoEnv)
        tabMassctrl.addItem(verticalSpacer)
   
        self.tabMass.layout.addLayout(tabMassctrl)
        self.tabMass.layout.addWidget(parent.mplmass)
        self.tabMass.setLayout(self.tabMass.layout) 
    
        # Create Uncertainty tab 
        self.tabUnc.layout =  QGridLayout()
        
        self.tabUnc.layout.setColumnStretch(parent.laUaMaxRows+1, parent.laUaMaxCols)
  
        la10 = QLabel('Item')
        la11 = QLabel('rel. unc.')
        la12 = QLabel('uncertainty')
        la10.setFont(parent.foBig)
        la11.setFont(parent.foBig)
        la12.setFont(parent.foBig)
 
        la00 = QLabel('measured mass')
        la00.setFont(parent.foBigBold)
        parent.laMass2.setFont(parent.foBigBold)
        parent.laTotUnc.setFont(parent.foBigBold)

        self.tabUnc.layout.addWidget(la00,0,0)
        self.tabUnc.layout.addWidget(parent.laMass2,0,1)
        self.tabUnc.layout.addWidget(parent.laTotUnc,0,2)

        self.tabUnc.layout.addWidget(la10,1,0)
        self.tabUnc.layout.addWidget(la11,1,1)
        self.tabUnc.layout.addWidget(la12,1,2)
        for i in range(parent.laUaMaxRows):
            for j in range(parent.laUaMaxCols):
                self.tabUnc.layout.addWidget(parent.laUa[i][j],i+2,j)

        self.tabUnc.setLayout(self.tabUnc.layout)
        # Create tab6 tab 
        self.tabProfile.layout = QHBoxLayout()
        self.tabProfile.layout.addWidget(parent.mplprofile)
        self.tabProfile.setLayout(self.tabProfile.layout) 
        
        # Create Report tab 
        self.tabReport.layout =  QGridLayout()
        
        self.tabReport.layout.setColumnStretch(len(parent.resultLabels), 2)
        for n,i in enumerate(parent.laResult):
            for m,j in enumerate(i):
                self.tabReport.layout.addWidget(j,n,m)
        
        self.tabReport.setLayout(self.tabReport.layout) 
  
        # Add tabs to widget 
        self.layout.addWidget(self.tabs) 
        self.setLayout(self.layout) 
        self.tabs.addTab(self.tabMass, "Mass")
        self.tabs.addTab(self.tabForce, "Force")
        self.tabs.addTab(self.tabEnv, "Environmentals")
        self.tabs.addTab(self.tabVelo, "Velocity")
        self.tabs.addTab(self.tabProfile, "Profile")
        self.tabs.addTab(self.tabUnc, "Uncertainty")
        self.tabs.addTab(self.tabReport, "Report")

        self.tabDiag = QWidget()
        self.diagText = QPlainTextEdit()
        self.diagText.setReadOnly(True)
        diagLayout = QVBoxLayout()
        diagLayout.addWidget(self.diagText)
        self.tabDiag.setLayout(diagLayout)
        self.tabs.addTab(self.tabDiag, "Diagnostics")



def excepthook(exc_type, exc_value, exc_tb):
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    try:
        print("error caught:")
        print(tb)
    except Exception:
        sys.__stderr__.write("error caught (stdout unavailable):\n" + tb)
    _log("EXCEPTION CAUGHT:\n" + tb)
    _logfile.flush()
    if getattr(sys, 'frozen', False):
        dbgpath = os.path.join(os.path.dirname(sys.executable), 'k2viewer.dbg')
    else:
        dbgpath = 'k2viewer.dbg'
    with open(dbgpath, 'w') as fi:
        fi.write("error catched!:\n")
        fi.write("error message:\n" + tb)
    QApplication.quit()
    # or QtWidgets.QApplication.exit(0)

try:
    myappid = 'mycompany.myproduct.subproduct.version'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    _log('creating QApplication')
    app = QApplication(sys.argv)
    app.setFont(QFont('Segoe UI', 10))
    app.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: #F5F7FA;
            color: #2D3748;
        }
        QTableWidget {
            background-color: #FFFFFF;
            alternate-background-color: #EDF2F7;
            gridline-color: #E2E8F0;
            border: 1px solid #CBD5E0;
            border-radius: 4px;
            selection-background-color: #BEE3F8;
            selection-color: #1A202C;
        }
        QHeaderView::section {
            background-color: #EDF2F7;
            color: #4A5568;
            padding: 5px 8px;
            border: none;
            border-bottom: 2px solid #CBD5E0;
            font-weight: bold;
        }
        QPushButton {
            background-color: #FFFFFF;
            color: #4A5568;
            border: 1px solid #CBD5E0;
            border-radius: 4px;
            padding: 5px 18px;
        }
        QPushButton:hover  { background-color: #EDF2F7; border-color: #A0AEC0; }
        QPushButton:pressed { background-color: #E2E8F0; }
        QTabWidget::pane {
            border: 1px solid #CBD5E0;
            border-radius: 4px;
            background-color: #FFFFFF;
            top: -1px;
        }
        QTabBar::tab {
            background-color: #EDF2F7;
            color: #718096;
            padding: 7px 18px;
            border: 1px solid #CBD5E0;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: #FFFFFF;
            color: #2D3748;
            font-weight: bold;
        }
        QTabBar::tab:hover:!selected { background-color: #E2E8F0; }
        QStatusBar {
            background-color: #EDF2F7;
            color: #718096;
            border-top: 1px solid #CBD5E0;
            font-size: 9pt;
        }
        QProgressBar {
            background-color: #E2E8F0;
            border: none;
            border-radius: 3px;
            max-height: 6px;
            text-align: center;
        }
        QProgressBar::chunk { background-color: #68A4C4; border-radius: 3px; }
        QCheckBox { spacing: 6px; }
        QCheckBox::indicator {
            width: 15px; height: 15px;
            border: 1px solid #CBD5E0;
            border-radius: 3px;
            background-color: #FFFFFF;
        }
        QCheckBox::indicator:checked {
            background-color: #68A4C4;
            border-color: #68A4C4;
        }
        QSpinBox, QDoubleSpinBox {
            background-color: #FFFFFF;
            border: 1px solid #CBD5E0;
            border-radius: 4px;
            padding: 3px 6px;
        }
        QSpinBox:focus, QDoubleSpinBox:focus { border-color: #68A4C4; }
        QPlainTextEdit {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 4px;
            font-family: Consolas, monospace;
            font-size: 9pt;
            color: #4A5568;
        }
        QLabel { background-color: transparent; }
    """)
    _log('QApplication OK')
    sys.excepthook = excepthook
    _log('creating MainWindow')
    window = MainWindow()
    _log('MainWindow OK')
    window.showMaximized()
    window.raise_()
    window.activateWindow()
    if _splash:
        _splash.close()
    _log('window shown — entering event loop')

    def _parse_version(v):
        parts = (v.strip().split('.') + ['0', '0', '0'])[:3]
        try:
            return tuple(int(x) for x in parts)
        except ValueError:
            return (0, 0, 0)

    def _check_for_update():
        import urllib.request, json
        try:
            url = "https://api.github.com/repos/schlammis/k2p2viewer/releases/latest"
            with urllib.request.urlopen(url, timeout=5) as r:
                data = json.loads(r.read())
            latest = data["tag_name"].lstrip("v")
            gh = _parse_version(latest)
            me = _parse_version(APP_VERSION)
            if gh > me:
                print(f"Update available: v{latest}  (you have v{APP_VERSION})")
                if (gh[0], gh[1]) > (me[0], me[1]):
                    window.updateAvailable.emit(latest, data['html_url'])
                else:
                    print(f"Patch update only — download at: {data['html_url']}")
            else:
                print(f"Version check: up to date (v{APP_VERSION})")
        except Exception as e:
            print(f"Version check failed: {e}")
    
    threading.Thread(target=_check_for_update, daemon=True).start()

    app.exec()
    _log('event loop exited normally')
except Exception:
    _log('FATAL ERROR:\n' + traceback.format_exc())
    raise
finally:
    _logfile.close()
sys.exit()
