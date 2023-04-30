from Client import BaseClient
import logging


class CustomFormatter(logging.Formatter):
    """Logging colored formatter, adapted from:
     https://alexandra-zaharia.github.io/posts/make-your-own-custom-color-formatter-with-python-logging/"""

    dark_grey = "\x1b[38;5;245m"
    bright_grey = "\x1b[38;5;250m"
    yellow = '\x1b[38;5;136m'
    red = '\x1b[31;20m'
    bold_red = '\x1b[31;1m'
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
            logging.CRITICAL: self.bold_red + self.level_fmt + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, self.time_fmt)
        return formatter.format(record)


class Mediator(BaseClient):
    def __init__(self, server_port=55555, send_format="utf-8", buffer_size=1024):
        super().__init__(server_port, send_format, buffer_size)
        self.init_connection()

    def init_connection(self):
        super().init_connection()
        self.connect_as_mediator()

    def send_request(self, template_id, params):
        self.send_msg(self.conn_sock, self.Actions.PROCESSING_REQUEST, params, template_id)

    def get_results(self, task_id):
        self.send_msg(self.conn_sock, self.Actions.GET_RESULTS_REQUEST, None, task_id)
        action, optional, reserved, response = self.recv_msg(self.conn_sock)
        return response

    def change_template(self):
        self.send_msg(self.conn_sock, self.Actions.CHANGE_TEMPLATE)

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
