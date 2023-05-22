from PyQt6 import QtGui
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import *
from enum import Enum
from threading import Lock


class Status(Enum):
    COMPLETED = "completed"
    ONGOING = "ongoing"
    REJECTED = "rejected"
    FAILED = "failed"

    def __str__(self):
        return self.value


class Colors(Enum):
    NORMAL = "#868079"
    FAILURE = "#DE463C"
    WARNING = "#937907"
    SUCCESS = "#678420"


class CustomItemDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)

        level = self.get_item_level(index)
        if level == 2:
            item = self.get_item(index)
            color = QtGui.QColor(item.color)
            option.palette.setColor(option.palette.ColorRole.Text, color)

    def sizeHint(self, option, index):
        desired_padding = 15
        size = super().sizeHint(option, index)
        level = self.get_item_level(index)

        if level == 1:
            size.setHeight(size.height() + desired_padding)

        return size

    def get_item(self, index):
        return index.model().itemFromIndex(index)

    def get_item_level(self, index):
        level = 1
        while index.parent().isValid():
            index = index.parent()
            level += 1
        return level


class Item(QtGui.QStandardItem):
    def __init__(self, name, index=None, color=None):
        super().__init__(name)
        self.name = name
        self.index = index
        self.color = color.value if color else None
        self.setEditable(False)


class ResultTree(QTreeView):
    def __init__(self, constants=None):
        super().__init__()
        self.constants = constants
        self.all_results = dict()
        self.lock = Lock()

        self.model = QtGui.QStandardItemModel()
        self.delegate = CustomItemDelegate()
        self.setItemDelegate(self.delegate)
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
            padding: 2px;
        }

        QTreeView::item:hover{
            color: none;
        }''')

    def add_user_runtime(self, index, runtime, serial_exec_status, currently_displayed):
        current_result = self.all_results.get(index)
        if current_result:
            with self.lock:
                self.all_results[index]['user_runtime'] = runtime
                self.all_results[index]['user_status'] = serial_exec_status
        else:
            with self.lock:
                self.all_results[index] = {'user_runtime': runtime, 'user_status': serial_exec_status}

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

    def add_unsuccessful(self, index, status):
        self.all_results[index] = {'status': status}

    def display_result(self, index):
        with self.lock:
            current_result = self.all_results[index]

        execution_unsuccessful = False
        cluster_status, user_status = current_result['status'], current_result.get('user_status', Status.COMPLETED)
        for item, function in [(self.exec_status, self.handle_status), (self.runtime_info, self.handle_runtime),
                               (self.final_result, self.handle_results),
                               (self.communication, self.handle_communication)]:

            values = function(current_result) if not execution_unsuccessful else []
            for _ in range(item.rowCount()):
                item.takeRow(0)
            for i in values:
                color = Colors.NORMAL
                if isinstance(i, tuple):
                    i, color = i
                new_item = Item(i, color=color)
                new_item.setSelectable(False)
                item.appendRow(new_item)

            if cluster_status != Status.COMPLETED or user_status != Status.COMPLETED:
                execution_unsuccessful = True

    def handle_status(self, current_result):
        status = current_result.get('user_status', Status.COMPLETED)
        if status == Status.FAILED:
            color = Colors.FAILURE
        else:
            status = current_result['status']
            if status == Status.COMPLETED:
                color = Colors.SUCCESS
            elif status == Status.REJECTED:
                color = Colors.WARNING
            elif status == Status.FAILED:
                color = Colors.FAILURE
            else:
                color = Colors.NORMAL
        item = (f"Execution {str(status)}", color)
        return [item]

    def handle_runtime(self, current_result):
        seconds = "secs"
        user_runtime = current_result.get('user_runtime')
        runtime = current_result.get('runtime')
        item1 = f"{runtime} {seconds} on Cluster"
        if user_runtime == Status.ONGOING:
            item2 = f"Control run is still {str(user_runtime)}"
            return [item1, item2]
        elif user_runtime:
            item2 = f"{user_runtime} {seconds} Control Run"
            time_saved = user_runtime - runtime
            item3 = f"{time_saved} {seconds} saved in total"
            item3 = (item3, Colors.SUCCESS) if time_saved > 0 else (item3, Colors.FAILURE)
            return [item1, item2, item3]
        else:
            return [item1]

    def handle_results(self, current_result):
        return [current_result['results']]

    def handle_communication(self, current_result):
        nodes_info = current_result['communication']
        template = "{ip} executed approximately {task_amount} task(s)"
        return [template.format(ip=ip, task_amount=task_amount) for ip, task_amount in nodes_info.items()]


class ResultPage(QFrame):
    def __init__(self, constants=None):
        super().__init__()

        self.displayed_filenames = dict()
        self.header_label_value = "Additional Info"
        self.constants = constants
        self.initial_item = True

        # List view
        self.list_view = QListView()
        self.list_view.setStyleSheet('''
        QListView {
            background-color: transparent;
            alternate-background-color: transparent;
            font-family: calibri;
            color: #b1b1b1;
            border-style: none;
            show-decoration-selected: 1;
        }

        QListView::item {
            background-color: #41404A;
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
        }

        QListView::item:alternate {
            background-color: transparent;
        }

        QListView::item:selected,
        QListView::item:selected:alternate {
            color: #e6e6ee;
            background-color: #504F5D;
        }

        QListView::item:hover,
        QListView::item:hover:alternate {
            border: 1px solid #e6e6ee;
        }
        ''')
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

    def cluster_result(self, text, index, *results, execution_finished=True):
        if self.initial_item:
            self.model.clear()
            self.initial_item = False
        item = Item(text, index)
        self.model.appendRow(item)
        if execution_finished:
            self.add_result(index, *results)
        else:
            execution_status, *_ = results
            self.add_unsuccessful(index, execution_status)

    def user_result(self, index, runtime, serial_exec_status):
        currently_displayed = self.list_view.currentIndex().row() == index
        self.tree.add_user_runtime(index, runtime, serial_exec_status, currently_displayed)

    def add_result(self, index, *result):
        self.tree.add_result(index, *result)

    def add_unsuccessful(self, index, status):
        self.tree.add_unsuccessful(index, status)

    def display_result(self, index=None):
        if index is None:
            index = self.list_view.currentIndex().row()
        self.tree.display_result(index)
