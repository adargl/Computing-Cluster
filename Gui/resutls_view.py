from PyQt6 import QtGui
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import *
from enum import Enum
from threading import Lock


class Status(Enum):
    ONGOING = "ongoing"
    FAILED = "failed"
    COMPLETED = "completed"


class CustomDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        editor.textChanged.connect(self.updateEditor)
        return editor

    def updateEditor(self, index):
        view = self.parent()
        view.update(index)


class Item(QtGui.QStandardItem):
    def __init__(self, name, index=None, result=None, info=None):
        super().__init__(name)
        self.name = name
        self.index = index
        self.result = result
        self.info = info
        self.setEditable(False)


class ResultTree(QTreeView):
    def __init__(self, constants=None):
        super().__init__()
        self.constants = constants
        self.all_results = dict()
        self.lock = Lock()

        self.model = QtGui.QStandardItemModel()
        self.setModel(self.model)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.header().hide()
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setSelectionBehavior(QTreeView.SelectionBehavior.SelectRows)
        self.setEditTriggers(QTreeView.EditTrigger.NoEditTriggers)

        self.exec_status = Item('Execution status')
        self.final_result = Item('Final result')
        self.runtime_info = Item('Runtime info')
        self.communication = Item('Communication')
        self.model.appendRow(self.exec_status)
        self.model.appendRow(self.final_result)
        self.model.appendRow(self.runtime_info)
        self.model.appendRow(self.communication)

        size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        size_policy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(size_policy)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.setStyleSheet('''
        QTreeView {
            background-color: transparent;
            font-family: calibri;
            font-size: 16px;
            color: #b1b1b1;
            border-style: none;
        }

        QTreeView::item {
            background-color: transparent;
            border-radius: 5px;
            padding: 10px;
        }

        QTreeView::item:hover{
            color: #e6e6ee;
        }''')

    def add_user_runtime(self, index, runtime, currently_displayed):
        current_result = self.all_results.get(index)
        if current_result:
            with self.lock:
                self.all_results[index]['user_runtime'] = runtime
        else:
            with self.lock:
                self.all_results[index] = {'user_runtime': runtime}

        if currently_displayed:
            self.display_result(index)

    def add_result(self, index, status, runtime, results, communication):
        if not self.all_results.get(index):
            new_result = {'user_runtime': Status.ONGOING}
        else:
            with self.lock:
                new_result = self.all_results.get(index)
        new_result['status'], new_result['results'] = status, results
        new_result['runtime'], new_result['communication'] = runtime, communication

        with self.lock:
            self.all_results[index] = new_result

    def display_result(self, index):
        with self.lock:
            current_result = self.all_results[index]
        for item, function in [(self.exec_status, self.handle_status), (self.runtime_info, self.handle_runtime),
                               (self.final_result, self.handle_results),
                               (self.communication, self.handle_communication)]:

            values = function(current_result)
            for _ in range(item.rowCount()):
                item.takeRow(0)
            for i in values:
                new_item = Item(i)
                new_item.setSelectable(False)
                item.appendRow(new_item)

    def handle_status(self, current_result):
        return [current_result['status']]

    def handle_runtime(self, current_result):
        seconds = "second(s)"
        user_runtime = current_result.get('user_runtime')
        runtime = current_result.get('runtime')
        item1 = f"Cluster finished in {runtime} {seconds}"
        if user_runtime == Status.ONGOING:
            item2 = f"Control run is still {user_runtime.value}"
            return [item1, item2]
        elif user_runtime:
            item2 = f"Control run finished in {user_runtime} {seconds}"
            item3 = f"Saved a total time of {user_runtime - runtime} {seconds}"
            return [item1, item2, item3]
        else:
            return [item1]

    def handle_results(self, current_result):
        return [current_result['results']]

    def handle_communication(self, current_result):
        return [current_result['communication']]


class ResultPage(QFrame):
    def __init__(self, constants=None):
        super().__init__()

        self.displayed_filenames = dict()
        self.header_label_value = "Additional Info"
        self.constants = constants
        self.initial_item = True

        # List view
        self.list_view = QListView()
        self.list_view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.list_view.setAlternatingRowColors(True)
        self.list_view.setObjectName("list_view")
        self.model = QtGui.QStandardItemModel(self.list_view)
        self.model.appendRow(Item("No results yet"))
        self.list_view.setModel(self.model)
        self.list_view.selectionModel().selectionChanged.connect(self.listview_selection_changed)

        info_frame = QFrame()
        size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        size_policy.setHeightForWidth(info_frame.sizePolicy().hasHeightForWidth())
        info_frame.setSizePolicy(size_policy)
        infoview_vertical_layout = QVBoxLayout(info_frame)
        infoview_vertical_layout.setContentsMargins(0, 0, 0, 0)
        infoview_vertical_layout.setSpacing(8)
        infoview_vertical_layout.addWidget(self.list_view)

        self.info_header_label = QLabel()
        font = QFont(
            self.constants["font"].get("family", "calibri"),
            self.constants["font"].get("font-size", 18),
            QFont.Weight.Bold,
            self.constants["font"].get("italic", False)
        )
        self.info_header_label.setFont(font)
        self.info_header_label.setText(self.header_label_value)
        self.tree = ResultTree(self.constants)

        infoview_vertical_layout.addWidget(self.info_header_label)
        infoview_vertical_layout.addWidget(self.tree)

        self.setStyleSheet(f'''
        QFrame {{
            background-color: {self.constants["paper-color"]};
        }}''')
        list_view_layout = QVBoxLayout(self)
        list_view_layout.setContentsMargins(*self.constants["sizes"]["margins"])
        list_view_layout.setSpacing(14)
        list_view_layout.setObjectName("file_view_layout")

        list_view_layout.addWidget(self.list_view, 1)
        list_view_layout.addWidget(info_frame, 3)

    def get_title(self, filename):
        number = self.displayed_filenames.get(filename, 0)
        self.displayed_filenames[filename] = number + 1
        if number:
            return filename + f" ({number})"
        return filename

    def listview_selection_changed(self):
        if self.initial_item:
            return
        index = self.list_view.currentIndex()
        item = self.list_view.model().itemFromIndex(index)
        self.info_header_label.setText(f"{self.header_label_value} - {item.name}")
        self.display_result(index.row())

    def cluster_result(self, text, index, *results):
        if self.initial_item:
            self.model.clear()
            self.initial_item = False
        item = Item(text, index)
        self.model.appendRow(item)
        self.add_result(index, *results)

    def user_result(self, index, runtime):
        currently_displayed = self.list_view.currentIndex().row() == index
        self.tree.add_user_runtime(index, runtime, currently_displayed)

    def add_result(self, index, *result):
        self.tree.add_result(index, *result)

    def display_result(self, index=None):
        if index is None:
            index = self.list_view.currentIndex().row()
        self.tree.display_result(index)
