import pickle
import socket
from struct import pack, unpack


class ClusterMediator:
    def __init__(self, server_port=55555, send_format="utf-8", buffer_size=1024):
        self.server_ip = socket.gethostbyname(socket.gethostname())
        self.server_port = server_port
        self.server_addr = (self.server_ip, self.server_port)
        self.format = send_format
        self.buffer = buffer_size
        self.conn_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.conn_sock.connect(self.server_addr)

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

    def new_request(self, global_variables):
        op_code = 1
        self.send_msg(self.conn_sock, op_code, global_variables)

    def await_response(self):
        response, op_code = self.recv_msg(self.conn_sock)
        return response

    def template_finished(self, is_last=False):
        op_code = 3
        self.send_msg(self.conn_sock, op_code, is_last)
        return self.await_response()
