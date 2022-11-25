import ast
import pickle
import socket
from struct import pack, unpack

import astpretty


class Client:
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
                tree, op_code = packet
                if op_code == 5:
                    exec_tree(tree)
                    self.send_response()
                    self.executed_count += 1
                    self.declare_ready()
                print(f"[DATA RECEIVED] server: {tree}")
            else:
                self.conn_sock.close()
                break

    def send_msg(self, sock, op_code, msg):
        pickled_msg = pickle.dumps(msg)
        pickled_msg = pack('>II', len(pickled_msg), op_code) + pickled_msg
        sock.sendall(pickled_msg)

    def recv_msg(self, sock):
        raw_header = self.recv_limited_bytes(sock, 8)
        if not raw_header:
            return None
        msg_len, op_code = unpack('>II', raw_header)
        pickled_msg = self.recv_limited_bytes(sock, msg_len)
        return pickle.loads(pickled_msg), op_code

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

    def send_response(self):
        op_code = 6
        with open(self.file) as file:
            global_vars = ast.literal_eval(file.read())
        self.send_msg(self.conn_sock, op_code, global_vars)


def exec_tree(tree, file_name=''):
    exec(compile(tree, file_name, 'exec'))


def print_tree(tree):
    astpretty.pprint(tree)


if __name__ == '__main__':
    client = Client("192.168.68.117")
    client.init_connection()
