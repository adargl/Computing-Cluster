from Client import BaseClient
from Logger import CustomFormatter
from struct import unpack
from pickle import loads
import logging


class User(BaseClient):
    def __init__(self, server_ip="localhost", server_port=55555, send_format="utf-8", buffer_size=1024):
        super().__init__(server_ip, server_port, send_format, buffer_size)

    def init_connection(self):
        """Initialize the connection with the server."""

        super().init_connection()
        self.connect_as_user()

    def cluster_exec(self, file):
        """Send a request to use the cluster services."""

        self.send_msg(self.conn_sock, self.Actions.USER_INPUT_FILE, file)

    def recv_final_output(self):
        """Return the final response of cluster."""

        raw_header = self.recv_limited_bytes(self.conn_sock, 16)
        if not raw_header:
            return None
        status_len, runtime_len, result_len, communication_len = unpack('>4I', raw_header)
        status = loads(self.recv_limited_bytes(self.conn_sock, status_len))
        runtime = loads(self.recv_limited_bytes(self.conn_sock, runtime_len))
        result = loads(self.recv_limited_bytes(self.conn_sock, result_len))
        communication = loads(self.recv_limited_bytes(self.conn_sock, communication_len))

        return status, runtime, result, communication

    def connect_as_user(self):
        """Send a request to connect as a user."""

        super().connect_as_user()
        logger.info(f"[CONNECTION REQUEST] request sent to connect as a user")


fmt = '%(name)s %(asctime)s.%(msecs)03d %(message)s', '%I:%M:%S'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logger.level)
stream_handler.setFormatter(CustomFormatter(*fmt))
logger.addHandler(stream_handler)

if __name__ == '__main__':
    client = User("localhost")
    client.init_connection()
    file_name = "Examples/Crisper.py"
    with open(file_name) as f:
        client.cluster_exec(f.read())
