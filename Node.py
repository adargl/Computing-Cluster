import pickle
import socket
from struct import pack, unpack
from Client import BaseClient

import logging


class CustomFormatter(logging.Formatter):
    """Logging colored formatter, adapted from:
     https://alexandra-zaharia.github.io/posts/make-your-own-custom-color-formatter-with-python-logging/"""

    dark_grey = "\x1b[38;5;245m"
    bright_grey = "\x1b[38;5;250m"
    yellow = '\x1b[38;5;136m'
    red = '\x1b[31;20m'
    green = '\x1b[38;5;64m'
    reset = '\x1b[0m'

    def __init__(self, level_fmt, time_fmt):
        super().__init__()
        self.level_fmt = level_fmt
        self.time_fmt = time_fmt
        self.FORMATS = {
            logging.DEBUG: self.dark_grey + self.level_fmt + self.reset,
            logging.INFO: self.dark_grey + self.level_fmt + self.reset,
            logging.WARNING: self.yellow + self.level_fmt + self.reset,
            logging.ERROR: self.red + self.level_fmt + self.reset,
            logging.CRITICAL: self.green + self.level_fmt + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, self.time_fmt)
        return formatter.format(record)


class ExecutableTree:
    def __init__(self, tree, params_name, params):
        self.tree = tree
        self.params_name = params_name
        self.params = params

    def exec_tree(self, file_name=''):
        exec(compile(self.tree, file_name, 'exec'), {self.params_name: self.params})


class Node(BaseClient):
    def __init__(self, server_ip, max_threads=2, server_port=55555, send_format="utf-8", buffer_size=1024):
        super().__init__(server_port, send_format, buffer_size)
        self.server_ip = server_ip
        self.max_threads = max_threads
        self.params = None

    def init_connection(self):
        super().init_connection()
        self.connect_as_node()
        self.handle_connection()

    def handle_connection(self):
        self.declare_ready()
        while True:
            packet = self.recv_msg(self.conn_sock)
            if packet:
                action, template_id, task_id, executable_tree = packet
                if action == self.Actions.SEND_TASK_TO_NODE:
                    self.params = executable_tree.params
                    logger.info(f"[DATA RECEIVED] request  (id={template_id}): {executable_tree.params}")
                    executable_tree.exec_tree()
                    self.send_response(template_id, task_id)
                    self.declare_ready()
            else:
                self.conn_sock.close()
                break

    def declare_ready(self):
        self.send_msg(self.conn_sock, self.Actions.NODE_AVAILABLE, self.max_threads)

    def send_response(self, template_id, task_id):
        self.send_msg(self.conn_sock, self.Actions.TASK_RESPONSE, self.params, template_id, task_id)
        logger.info(f"[RESPONSE SENT] response (id={template_id}): {self.params}")

    def connect_as_node(self):
        super().connect_as_node()
        logger.info(f"[CONNECTION REQUEST] request sent to connect as a node")


if __name__ == '__main__':
    logging_file = 'Server.log'
    fmt = '%(name)s %(asctime)s.%(msecs)03d %(message)s', '%I:%M:%S'
    with open(logging_file, 'w'):
        pass

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logger.level)
    stream_handler.setFormatter(CustomFormatter(*fmt))
    logger.addHandler(stream_handler)

    client = Node("localhost")
    client.init_connection()
