from Client import BaseClient
from Logger import CustomFormatter
import logging


class Mediator(BaseClient):
    def __init__(self, server_port=55555, send_format="utf-8", buffer_size=1024):
        super().__init__(server_port=server_port, send_format=send_format, buffer_size=buffer_size)
        self.init_connection()

    def init_connection(self):
        super().init_connection()
        self.connect_as_mediator()

    def send_request(self, template_id, params):
        self.send_msg(self.conn_sock, self.Actions.PROCESSING_REQUEST, params, template_id)

    def send_while_request(self, template_id, params):
        self.send_msg(self.conn_sock, self.Actions.WHILE_PROCESSING_REQUEST, params, template_id)

    def get_results(self, task_id):
        self.send_msg(self.conn_sock, self.Actions.GET_RESULTS_REQUEST, None, task_id)
        action, optional, reserved, response = self.recv_msg(self.conn_sock)
        return response

    def connect_as_mediator(self):
        super().connect_as_mediator()
        logger.info(f"[CONNECTION REQUEST] request sent to connect as a mediator")


fmt = '%(name)s %(asctime)s.%(msecs)03d %(message)s', '%I:%M:%S'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logger.level)
stream_handler.setFormatter(CustomFormatter(*fmt))
logger.addHandler(stream_handler)
