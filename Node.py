import ast
import pickle
import socket
from struct import pack, unpack

import astpretty
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


class Node:
    def __init__(self, server_ip, server_port=55555, send_format="utf-8", buffer_size=1024):
        self.server_ip = server_ip
        self.server_port = server_port
        self.server_addr = (self.server_ip, self.server_port)
        self.format = send_format
        self.buffer = buffer_size
        self.conn_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.executed_count = 0
        self.file = "shared.txt"

    def init_connection(self):
        self.conn_sock.connect(self.server_addr)
        self.handle_connection()

    def handle_connection(self):
        self.declare_ready()
        while True:
            packet = self.recv_msg(self.conn_sock)
            if packet:
                op_code, task_id, tree = packet
                if op_code == 5:
                    exec_tree(tree)
                    self.send_response(task_id)
                    self.executed_count += 1
                    self.declare_ready()
                logger.info(f"[DATA RECEIVED] server: {tree}")
            else:
                self.conn_sock.close()
                break

    def send_msg(self, sock, op_code, msg, reserved=0):
        pickled_msg = pickle.dumps(msg)
        pickled_msg = pack('>III', len(pickled_msg), op_code, reserved) + pickled_msg
        sock.sendall(pickled_msg)

    def recv_msg(self, sock):
        raw_header = self.recv_limited_bytes(sock, 12)
        if not raw_header:
            return None
        msg_len, op_code, reserved = unpack('>III', raw_header)
        pickled_msg = self.recv_limited_bytes(sock, msg_len)
        return op_code, reserved, pickle.loads(pickled_msg)

    def recv_limited_bytes(self, sock, n):
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def declare_ready(self):
        op_code = 4
        self.send_msg(self.conn_sock, op_code, self.executed_count)

    def send_response(self, task_id):
        op_code = 6
        with open(self.file) as file:
            params = ast.literal_eval(file.read())
        self.send_msg(self.conn_sock, op_code, params, task_id)


def exec_tree(tree, file_name=''):
    exec(compile(tree, file_name, 'exec'), globals())


def print_tree(tree):
    astpretty.pprint(tree)


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

    file_handler = logging.FileHandler(logging_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(*fmt))
    logger.addHandler(file_handler)

    client = Node("192.168.68.112")
    client.init_connection()
