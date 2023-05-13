import os
import json
from User import User
from editor import CodeEditor
from file_view import FileManager
from resutls_view import ResultPage
from PyQt6 import QtGui
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import *
from threading import Thread
from itertools import count
from time import perf_counter

from enum import Enum
from pathlib import Path


class MainWindow(QMainWindow):
    class Page(Enum):
        IDE = 0
        RESULTS = 1
        INFO = 2

    def __init__(self, server_ip="localhost", constants=None):
        super().__init__()

        # Connect to the cluster server
        self.server_ip = server_ip
        self.request_id = count()
        self.sock = User(server_ip)
        self.sock.init_connection()

        # Init constants
        if not constants:
            self.constants = "./constants.json"
        else:
            self.constants = constants
        with open(self.constants, "r") as f:
            self.constants_json = json.load(f)["constants"]

        # Set window
        self.setObjectName("main_window")
        self.window_title = "Cluster Manager"
        self.filename_template = "{filename} - " + self.window_title
        self.setWindowTitle(self.window_title)
        self.resize(*self.constants_json["main-window"]["sizes"]["base"])
        self.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.setStyleSheet(open("style.qss", "r").read())

        # Font
        self.main_font = QFont(
            self.constants_json["main-window"]["font"].get("family", "Consolas"),
            self.constants_json["main-window"]["font"].get("font-size", 12),
            QFont.Weight.Normal,
            self.constants_json["main-window"]["font"].get("italic", False)
        )
        self.setFont(self.main_font)
        self.encoding_type = "utf-8"

        editor = self.create_editor()
        self.setCentralWidget(editor)

        # Additional
        self.model = None
        self.tab_view = None
        self.sidebar = None
        self.stacked = None
        self.splitter = None
        self.results_view = None
        self.run_button = None
        self.file_view = None
        self.file_manager = None
        self.current_filepath = None

        self.setup_body()
        self.setup_menu()

    @property
    def current_request_id(self):
        return next(self.request_id)

    @property
    def current_directory(self):
        if not self.current_filepath:
            return os.getcwd()
        return str(self.current_filepath.parent)

    @property
    def get_filename(self):
        filename = self.current_filepath.stem
        return self.results_view.get_title(filename)

    def change_page(self, page):
        self.stacked.setCurrentIndex(page.value)

    def create_editor(self, path=None):
        editor = CodeEditor(self, self.constants_json["editor"], path)
        return editor

    def setup_menu(self):
        file_menu = self.menuBar().addMenu("File")

        open_action = QtGui.QAction("Open", self)
        open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        save_action = QtGui.QAction("Save", self)
        save_action.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)
        save_as_action = QtGui.QAction("Save As", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self.save_file_as)
        file_menu.addAction(save_as_action)
        run_action = QtGui.QAction("Run", self)
        run_action.setShortcut("Ctrl+R")
        run_action.triggered.connect(self.run_file)
        file_menu.addAction(run_action)

    def setup_body(self):
        # Main body
        body_frame = QFrame(parent=self)
        size_policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        size_policy.setHeightForWidth(body_frame.sizePolicy().hasHeightForWidth())
        body_frame.setSizePolicy(size_policy)
        body_frame.setFrameShape(QFrame.Shape.StyledPanel)
        body_frame.setFrameShadow(QFrame.Shadow.Plain)
        body_frame.setObjectName("body_frame")
        body_frame.setLineWidth(0)
        body_frame.setMidLineWidth(0)
        body_frame.setContentsMargins(0, 0, 0, 0)
        body_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        body_horizontal_layout = QHBoxLayout(body_frame)
        body_horizontal_layout.setContentsMargins(0, 0, 0, 0)
        body_horizontal_layout.setSpacing(0)
        body_horizontal_layout.setObjectName("body_horizontal_layout")

        # Tab view
        self.tab_view = QTabWidget()
        self.tab_view.setContentsMargins(0, 0, 0, 0)
        self.tab_view.setTabsClosable(True)
        self.tab_view.setMovable(True)
        self.tab_view.setDocumentMode(True)
        self.tab_view.currentChanged.connect(self.change_tab)
        self.tab_view.tabCloseRequested.connect(self.close_tab)
        size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        size_policy.setHeightForWidth(self.tab_view.sizePolicy().hasHeightForWidth())
        self.tab_view.setSizePolicy(size_policy)
        self.tab_view.setCurrentIndex(0)
        self.tab_view.setObjectName("tab_view")

        editor = self.create_editor()
        self.new_tab(editor)

        # Side bar
        self.sidebar = QFrame()
        size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        size_policy.setHeightForWidth(self.sidebar.sizePolicy().hasHeightForWidth())
        self.sidebar.setSizePolicy(size_policy)
        self.sidebar.setFrameShape(QFrame.Shape.StyledPanel)
        self.sidebar.setFrameShadow(QFrame.Shadow.Plain)
        self.sidebar.setStyleSheet(f'''
        QFrame {{
            background-color: {self.constants_json["sidebar"]["paper-color"]};
            color: {self.constants_json["sidebar"]["color"]};
        }}
        
        QPushButton {{
            border-radius: 5px;
            color: {self.constants_json["sidebar"]["color"]};
            text-align: left;
            font-family: {self.constants_json["sidebar"]["button-font"]["family"]};
            font-size: {self.constants_json["sidebar"]["button-font"]["font-size"]}px;
            padding: {self.constants_json["sidebar"]["button-padding"]};
        }}

        QPushButton:hover {{
            background-color: {self.constants_json["sidebar"]["button-hovered"]};
        }}

        QPushButton:pressed {{
            background-color: {self.constants_json["sidebar"]["button-pressed"]};
        }}
        ''')
        self.sidebar.setMaximumSize(QSize(*self.constants_json["sidebar"]["sizes"]["max"]))
        self.sidebar.setMinimumSize(QSize(*self.constants_json["sidebar"]["sizes"]["min"]))
        self.sidebar.setObjectName("side_bar")
        sidebar_vertical_layout = QVBoxLayout(self.sidebar)
        sidebar_vertical_layout.setContentsMargins(*self.constants_json["sidebar"]["sizes"]["margins"])
        sidebar_vertical_layout.setSpacing(0)
        sidebar_vertical_layout.setObjectName("sidebar_vertical_layout")

        # Sidebar labels
        menu_button = self.add_sidebar_component("Menu", "sources/menu.png", self.toggle_sidebar)
        sidebar_vertical_layout.addWidget(menu_button)
        folder_button = self.add_sidebar_component("Folder", "sources/folder.png", self.toggle_file_view)
        sidebar_vertical_layout.addWidget(folder_button)
        home_button = self.add_sidebar_component("Home", "sources/home.png", lambda: self.change_page(self.Page.IDE))
        sidebar_vertical_layout.addWidget(home_button)
        results_button = self.add_sidebar_component("Completed", "sources/completed.png",
                                                    lambda: self.change_page(self.Page.RESULTS))
        sidebar_vertical_layout.addWidget(results_button)
        search_button = self.add_sidebar_component("Search", "sources/search.png", lambda: None)
        sidebar_vertical_layout.addWidget(search_button)
        self.run_button = self.add_sidebar_component("Run", "sources/run.png", self.run_file)
        sidebar_vertical_layout.addWidget(self.run_button)
        spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        sidebar_vertical_layout.addItem(spacer)

        # File manager
        self.file_view = QFrame()
        self.file_view.setMaximumSize(QSize(*self.constants_json["file-view"]["sizes"]["max"]))
        self.file_view.setMinimumSize(QSize(*self.constants_json["file-view"]["sizes"]["min"]))
        self.file_view.setStyleSheet(f'''
        QFrame {{
            background-color: {self.constants_json["file-view"]["paper-color"]};
        }}''')
        self.file_view.setObjectName("file_view")
        file_view_layout = QVBoxLayout(self.file_view)
        file_view_layout.setContentsMargins(*self.constants_json["file-view"]["sizes"]["margins"])
        file_view_layout.setSpacing(0)
        self.file_manager = FileManager(
            main_window=self,
            tab_view=self.tab_view,
            set_new_tab=self.open_tab,
            constants=self.constants_json["file-view"]
        )
        file_view_layout.addWidget(self.file_manager)
        file_view_layout.setObjectName("file_view_layout")

        # Results page
        self.results_view = ResultPage(self.constants_json["list-view"])

        # Setup container widgets
        self.stacked = QStackedWidget()
        self.stacked.addWidget(self.tab_view)
        self.stacked.addWidget(self.results_view)
        self.stacked.setCurrentIndex(1)

        self.splitter = QSplitter(self)
        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(self.file_view)
        self.splitter.addWidget(self.stacked)
        self.splitter.setChildrenCollapsible(True)
        self.splitter.setHandleWidth(0)
        size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        size_policy.setHeightForWidth(self.splitter.sizePolicy().hasHeightForWidth())
        self.splitter.setSizePolicy(size_policy)
        self.setCentralWidget(self.splitter)

    def add_sidebar_component(self, name, icon_path, function):
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(icon_path), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        push_button = QPushButton(parent=self.sidebar)
        push_button.setIcon(icon)
        push_button.setIconSize(QSize(*self.constants_json["sidebar"]["sizes"]["icon"]))
        push_button.setText(" " * 4 + name)
        push_button.clicked.connect(function)

        return push_button

    def toggle_sidebar(self):
        max_width, max_height = tuple(self.constants_json["sidebar"]["sizes"]["max"])
        min_width, min_height = tuple(self.constants_json["sidebar"]["sizes"]["min"])

        sizes = self.splitter.sizes()
        if sizes[0] == max_width:
            sizes[0], sizes[2] = min_width, sum(sizes) - min_width - sizes[1]
        else:
            sizes[0], sizes[2] = max_width, sum(sizes) - max_width - sizes[1]
        self.splitter.setSizes(sizes)

    def toggle_file_view(self):
        max_width, max_height = tuple(self.constants_json["file-view"]["sizes"]["max"])
        min_width, min_height = tuple(self.constants_json["file-view"]["sizes"]["min"])

        sizes = self.splitter.sizes()
        if sizes[1] == max_width:
            sizes[1], sizes[2] = min_width, sum(sizes) - min_width - sizes[0]
        else:
            sizes[1], sizes[2] = max_width, sum(sizes) - max_width - sizes[0]
        self.splitter.setSizes(sizes)

    def new_tab(self, editor):
        self.tab_view.addTab(editor, editor.filename)
        self.setWindowTitle(self.filename_template.format(filename=editor.filename))
        self.tab_view.setCurrentIndex(self.tab_view.count() - 1)

    def open_tab(self, path, is_new_file=False):
        if path.is_dir():
            return
        editor = self.create_editor(path)

        if is_new_file:
            self.new_tab(editor)
            self.current_filepath = path
            return

        # check if file already open
        for i in range(self.tab_view.count()):
            if self.tab_view.tabText(i) == path.name or self.tab_view.tabText(i) == "*" + path.name:
                self.tab_view.setCurrentIndex(i)
                self.current_filepath = path
                return

        # create new tab
        self.tab_view.addTab(editor, path.name)
        editor.setText(path.read_text(encoding=self.encoding_type))
        self.setWindowTitle(f"{path.name} - {self.window_title}")
        self.current_filepath = path
        self.tab_view.setCurrentIndex(self.tab_view.count() - 1)

    def close_tab(self, index):
        editor = self.tab_view.widget(index)
        if editor.file_changed:
            pressed = self.show_dialog("Close", f"Do you want to save the changes you made to {editor.filename}?")
            if pressed == QMessageBox.StandardButton.Yes:
                self.save_file()
            elif pressed == QMessageBox.StandardButton.No:
                pass
            else:
                return

        if self.tab_view.count() == 1:
            self.new_tab(self.create_editor())
        self.tab_view.removeTab(index)

    def change_tab(self):
        editor = self.tab_view.currentWidget()
        self.current_filepath = editor.filepath
        filename = editor.filename + "*" if editor.file_changed else editor.filename
        self.setWindowTitle(self.filename_template.format(filename=filename))

    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open File", self.current_directory, "Python Files (*.py)")

        if filename:
            with open(filename, "r") as f:
                code = f.read()
            self.current_filepath = Path(filename)
            editor = self.create_editor(self.current_filepath)
            self.new_tab(editor)
            self.tab_view.currentWidget().setText(code)

    def save_file(self):
        editor = self.tab_view.currentWidget()
        program = editor.text()
        if not program:
            return
        if editor.file_not_saved:
            self.save_file_as()
            return

        self.current_filepath.write_text(program)
        editor.save_file(self.current_filepath)

    def save_file_as(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save File", self.current_directory, "Python Files (*.py)")
        if filename == "":
            return

        new_path = Path(filename)
        editor = self.tab_view.currentWidget()
        program = editor.text()

        if new_path.exists():
            for i in range(self.tab_view.count()):
                if self.tab_view.widget(i).filepath == new_path:
                    self.tab_view.removeTab(i)
                    break
        if editor.file_not_saved:
            editor.save_file(new_path)
        else:
            new_editor = self.create_editor(new_path)
            new_editor.setText(program)
            self.new_tab(new_editor)

        self.current_filepath = new_path
        self.current_filepath.write_text(program)

    def run_file(self):
        self.run_button.setEnabled(False)
        request_id = self.current_request_id
        if self.current_filepath:
            with open(self.current_filepath) as file:
                code = file.read()
                self.sock.send_input_file(code)
                Thread(target=lambda: self.exec_file(code, request_id)).start()
            Thread(target=lambda: self.await_result(request_id)).start()
        else:
            self.run_button.setEnabled(True)

    def await_result(self, request_id):
        output = self.sock.recv_final_output()
        self.results_view.cluster_result(self.get_filename, request_id, *output)
        self.run_button.setEnabled(True)

    def exec_file(self, code, request_id):
        start_time = perf_counter()
        exec(code, {'builtins': globals()['__builtins__']})
        finish_time = perf_counter()
        runtime = finish_time - start_time
        self.results_view.user_result(request_id, runtime)

    def show_dialog(self, title, text):
        dialog = QMessageBox(self)
        dialog.setWindowTitle(title)
        # dialog.setWindowIcon(QIcon(":/icons/close-icon.svg"))
        dialog.setText(text)
        dialog.setStandardButtons(QMessageBox.StandardButton.Yes
                                  | QMessageBox.StandardButton.No
                                  | QMessageBox.StandardButton.Cancel)
        dialog.setDefaultButton(QMessageBox.StandardButton.Cancel)
        dialog.setIcon(QMessageBox.Icon.Warning)
        return dialog.exec()


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow("localhost")
    window.show()
    app.exec()
