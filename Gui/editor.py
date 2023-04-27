from PyQt6.Qsci import QsciScintilla
from PyQt6.QtGui import QColor, QFont
from lexer import PyCustomLexer
from PyQt6 import QtGui, Qsci


class CodeEditor(QsciScintilla):
    def __init__(self, main_window, constants=None, filepath=None):
        super().__init__()
        self.constants = constants
        self.filepath = filepath
        self.main_window = main_window
        self.first_launch = True
        self.file_not_saved = True if not filepath else False
        self.filename_template = main_window.filename_template
        self.original_code = self.text()
        self._file_changed = False
        self.textChanged.connect(self.text_changed)

        # Editor
        # self.cursorPositionChanged.connect(self._cusorPositionChanged)
        # self.textChanged.connect(self._textChanged)

        # Encoding
        self.setUtf8(True)

        # Font
        self.main_font = QFont(
            self.constants["font"].get("family", "Consolas"),
            self.constants["font"].get("font-size", 12),
            QtGui.QFont.Weight.Normal,
            self.constants["font"].get("italic", False)
        )
        self.setFont(self.main_font)

        # Brace matching
        # self.setBraceMatching(QsciScintilla.BraceMatch.SloppyBraceMatch)

        # Indentation
        self.setTabWidth(4)
        self.setIndentationGuides(True)
        self.setIndentationsUseTabs(True)
        self.setAutoIndent(True)

        # Autocomplete
        self.setAutoCompletionSource(QsciScintilla.AutoCompletionSource.AcsAll)
        self.setAutoCompletionThreshold(1)
        self.setAutoCompletionCaseSensitivity(False)
        self.setAutoCompletionUseSingle(QsciScintilla.AutoCompletionUseSingle.AcusNever)

        # Caret
        self.setCaretLineVisible(True)
        self.setCaretWidth(2)
        self.setCaretForegroundColor(QColor(self.constants["caret-foreground-color"]))
        self.setCaretLineBackgroundColor(QColor(self.constants["caret-background-color"]))

        # EOL
        self.setEolMode(QsciScintilla.EolMode.EolWindows)
        self.setEolVisibility(False)

        # Highlighted text
        self.setSelectionBackgroundColor(QColor(self.constants["selection-color"]))

        # Scroll bars
        # self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # line numbers
        self.setMarginType(0, QsciScintilla.MarginType.NumberMargin)
        self.setMarginWidth(0, "00000")
        self.setMarginsForegroundColor(QColor(self.constants["lines-foreground-color"]))
        self.setMarginsBackgroundColor(QColor(self.constants["lines-background-color"]))
        self.setMarginsFont(self.main_font)

        self.python_lexer = PyCustomLexer(self)
        self.python_lexer.setDefaultFont(self.main_font)

        self.__api = Qsci.QsciAPIs(self.python_lexer)
        # self.auto_completer = Qsci.QScintilla.AutoCompleter(self.full_path, self.__api)
        # self.auto_completer.finished.connect(self.loaded_autocomplete)  # you can use this callback to do something

        self.setLexer(self.python_lexer)

    @property
    def filename(self):
        if not self.filepath:
            return "untitled"
        return self.filepath.name

    @property
    def file_changed(self):
        return self._file_changed

    @file_changed.setter
    def file_changed(self, bool_value):
        index = self.main_window.tab_view.currentIndex()
        if bool_value:
            self.main_window.tab_view.setTabText(index, self.filename + "*")
            self.main_window.setWindowTitle(self.filename_template.format(filename=self.filename + "*"))
        else:
            self.main_window.tab_view.setTabText(index, self.filename)
            self.main_window.setWindowTitle(self.filename_template.format(filename=self.filename))

        self._file_changed = bool_value

    def text_changed(self):
        if self.first_launch and not self.file_not_saved:
            self.first_launch = False
        elif self.original_code == self.text():
            self.file_changed = False
        elif not self.file_changed:
            self.file_changed = True

    def save_file(self, path):
        self.filepath = path
        self.original_code = self.text()
        self.file_not_saved = False
        self.file_changed = False

    def setText(self, text):
        self.original_code = text
        super().setText(text)
