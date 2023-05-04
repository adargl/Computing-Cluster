from Client import BaseClient
from Logger import CustomFormatter
import logging


class User(BaseClient):
    def __init__(self, server_ip, server_port=55555, send_format="utf-8", buffer_size=1024):
        super().__init__(server_port, send_format, buffer_size)
        self.server_ip = server_ip

    def init_connection(self):
        super().init_connection()
        self.connect_as_user()

    def send_input_file(self, file_name):
        self.send_msg(self.conn_sock, self.Actions.USER_INPUT_FILE, file_name)

    def connect_as_user(self):
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
    client.send_input_file("Examples/Xrisper.py")
