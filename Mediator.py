import pickle
import socket
from struct import pack, unpack
import logging

class Mediator:
    def __init__(self, server_port=55555, send_format="utf-8", buffer_size=1024):
        self.server_ip = socket.gethostbyname(socket.gethostname())
        self.server_port = server_port
        self.server_addr = (self.server_ip, self.server_port)
        self.format = send_format
        self.buffer = buffer_size
        self.conn_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.conn_sock.connect(self.server_addr)

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

    def send_request(self, template_id, params):
        op_code = 1
        self.send_msg(self.conn_sock, op_code, params, template_id)

    def get_results(self, task_id):
        op_code = 3
        self.send_msg(self.conn_sock, op_code, task_id)
        op_code, reserved, response = self.recv_msg(self.conn_sock)
        return response


class Handler:
    def __init__(self):
        self.mediator = Mediator()
        self.concurrent = 0

    def set_connection_concurrent(self):
        self.concurrent = 1

    def handle_connection(self):
        if self.concurrent:
            pass
