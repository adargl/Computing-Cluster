import socket
import pickle
from struct import pack, unpack
from enum import Enum


class BaseClient:
    class Actions(Enum):
        RESERVED = 0
        PROCESSING_REQUEST = 1
        WHILE_PROCESSING_REQUEST = 2
        PROCESSING_RESPONSE = 3
        GET_RESULTS_REQUEST = 4
        NODE_AVAILABLE = 5
        SEND_TASK_TO_NODE = 6
        TASK_RESPONSE = 7
        TASK_FAILED = 8
        CONNECT_AS_MEDIATOR = 9
        CONNECT_AS_NODE = 10
        CONNECT_AS_USER = 11
        USER_INPUT_FILE = 12
        FINAL_RESPONSE = 13
        UNDEFINED_CODE = 14

    def __init__(self, server_ip="localhost", server_port=55555, send_format="utf-8", buffer_size=1024):
        self.server_ip = server_ip
        self.server_port = server_port
        self.server_addr = (self.server_ip, self.server_port)
        self.format = send_format
        self.buffer = buffer_size
        self.conn_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def init_connection(self):
        """Initialize the connection with the server."""

        self.conn_sock.connect(self.server_addr)

    def send_msg(self, sock, action, msg=None, optional=0, reserved=0):
        """Main message sending protocol over the sockets.

        Each message is sent in 5 fields: | length | op code | optional | reserved | message |
        'send_msg' utilizes the serializing functionalities of struct.pack and pickle.dumps.

        """

        op_code = action.value
        pickled_msg = pickle.dumps(msg)
        pickled_msg = pack('>4I', len(pickled_msg), op_code, optional, reserved) + pickled_msg
        sock.sendall(pickled_msg)

    def recv_msg(self, sock):
        """Main message receiving protocol over the sockets.

        Each message is received in 5 fields: | length | op code | optional | reserved | message |
        'recv_msg' utilizes the deserializing functionalities of struct.unpack and pickle.loads.

        """

        raw_header = self.recv_limited_bytes(sock, 16)
        if not raw_header:
            return None
        msg_len, op_code, optional, reserved = unpack('>4I', raw_header)
        pickled_msg = self.recv_limited_bytes(sock, msg_len)
        msg = pickle.loads(pickled_msg)
        action = self.Actions(op_code)
        return action, optional, reserved, msg

    def recv_limited_bytes(self, sock, n):
        """Return a limited amount of bytes waiting on the socket.

        This is a helper function that allows the reading of limited amounts of bytes.

        """

        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def connect_as_mediator(self, user_sock_id):
        """Send a request to connect as a mediator."""

        self.send_msg(self.conn_sock, self.Actions.CONNECT_AS_MEDIATOR, user_sock_id)

    def connect_as_user(self):
        """Send a request to connect as a user."""

        self.send_msg(self.conn_sock, self.Actions.CONNECT_AS_USER)

    def connect_as_node(self):
        """Send a request to connect as a node."""

        self.send_msg(self.conn_sock, self.Actions.CONNECT_AS_NODE)
