import os
from User import User
from PyQt5.QtCore import QFile
from PyQt6 import QtWidgets, QtGui, Qsci
from PyQt6.QtWidgets import QFileDialog


class CodeEditor(Qsci.QsciScintilla):
    def __init__(self, parent=None):
        super().__init__()

        self.setLexer(Qsci.QsciLexerPython(self))
        self.setUtf8(True)
        self.setAutoIndent(True)
        self.setIndentationsUseTabs(False)
        self.setTabWidth(4)
        self.setIndentationWidth(4)
        self.setStyleSheet(open('style.css').read())


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.sock = User("localhost")
        self.sock.init_connection()
        self.initial_dir = os.path.join(os.getcwd(), "..")
        self.file_name = None

        # Create a QScintilla code editor
        self.editor = CodeEditor()
        self.setCentralWidget(self.editor)

        file_menu = self.menuBar().addMenu("File")

        open_action = QtGui.QAction("Open", self)
        open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        save_action = QtGui.QAction("Save", self)
        save_action.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)
        run_action = QtGui.QAction("Run", self)
        run_action.triggered.connect(self.run_file)
        file_menu.addAction(run_action)

    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open File", self.initial_dir, "Python Files (*.py)")
        self.file_name = filename

        # If a file was selected, load its contents into the code editor
        if filename:
            with open(filename, "r") as f:
                text = f.read()
                self.editor.setText(text)

    def save_file(self):
        program = self.editor.text()
        if not program:
            return

        with open(self.file_name, "w") as f:
            f.write(program)

    def run_file(self):
        self.sock.send_input_file(self.file_name)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
