import ast
import concurrent.futures
from threading import Semaphore
from Client import BaseClient

import logging
from Logger import CustomFormatter


class ExecutableTree:
    def __init__(self, tree, template_id, params_name, params):
        self.tree = tree
        self.template_id = template_id
        self.params_name = params_name
        self.params = params

    def exec_tree(self):
        """Execute the object's ast tree."""

        exec(ast.unparse(self.tree), {self.params_name: self.params})


class Node(BaseClient):
    def __init__(self, server_ip="localhost", max_threads=1, server_port=55555, send_format="utf-8", buffer_size=1024):
        super().__init__(server_ip, server_port, send_format, buffer_size)
        self.max_threads = max_threads
        self.semaphore = Semaphore(max_threads)

    def init_connection(self):
        """Initialize the connection with the server."""

        super().init_connection()
        self.connect_as_node()
        self.handle_connection()

    def recv_msg(self, sock):
        """Override the message receiving function to handle server disconnections."""

        msg = super().recv_msg(sock)
        if not msg:
            return
        elif isinstance(msg, Exception):
            exception = msg
            logger.error(f"[ERROR ENCOUNTERED] while receiving a message encountered: {exception}")
            logger.error(f"[CONNECTION CLOSED] connection with server from {self.server_ip}, "
                         f"{self.server_port} had been closed")
            return

        return msg

    def handle_connection(self):
        """Handle all processing requests.

        This function uses a ThreadPoolExecutor in order to handle several requests at once.

        """

        for _ in range(self.max_threads):
            self.declare_ready()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            while True:
                packet = self.recv_msg(self.conn_sock)
                if packet:
                    action, template_id, task_id, executable_tree = packet
                    if action == self.Actions.SEND_TASK_TO_NODE:
                        executor.submit(self.execute_task, executable_tree, template_id, task_id)
                else:
                    self.conn_sock.close()
                    executor.shutdown(wait=False)
                    break

    def thread_failure(func):
        """Handle thread failure.

        Called as a decorator of 'execute_task'. Tries executing a task and if exception is encountered
        sends back a message that indicates the task had failed.

        """

        def wrapper(self, *args, is_root=True, call_count=0, max_calls=2, **kwargs):
            try:
                func(self, *args, **kwargs)
            except Exception as e:
                logger.warning(f"[ERROR ENCOUNTERED] attempt {call_count + 1} raised: {e}")
                if call_count + 1 < max_calls:
                    wrapper(self, *args, is_root=False, call_count=call_count + 1, max_calls=max_calls, **kwargs)
                else:
                    self.send_msg(self.conn_sock, self.Actions.TASK_FAILED, *args)
                    logger.error("[ERROR ENCOUNTERED] passed the allowed amount of function calls")
            finally:
                if is_root:
                    self.declare_ready()

        return wrapper

    @thread_failure
    def execute_task(self, executable_tree, template_id, task_id):
        """Execute a task.

        A separate thread on the node that executes a task. A semaphore object is utilized so that the amount of
        threads won't exceed the maximum.

        """

        with self.semaphore:
            params = executable_tree.params
            logger.info(f"[DATA RECEIVED] request  (id={template_id}): {params}")
            executable_tree.exec_tree()
            self.send_response(template_id, task_id, params)

    def declare_ready(self):
        """Send a message indicating that there is room for one more task. Called each time an execution thread
         finishes."""

        self.send_msg(self.conn_sock, self.Actions.NODE_AVAILABLE, self.max_threads)

    def send_response(self, template_id, task_id, params):
        """Send task response aka a processing response."""

        self.send_msg(self.conn_sock, self.Actions.TASK_RESPONSE, params, template_id, task_id)
        logger.info(f"[RESPONSE SENT] response (id={template_id}): {params}")

    def connect_as_node(self):
        """Send a request to connect as a node."""

        super().connect_as_node()
        logger.info(f"[CONNECTION REQUEST] request sent to connect as a node")


if __name__ == '__main__':
    fmt = '%(name)s %(asctime)s.%(msecs)03d %(message)s', '%I:%M:%S'

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logger.level)
    stream_handler.setFormatter(CustomFormatter(*fmt))
    logger.addHandler(stream_handler)

    client = Node("localhost", 10)
    client.init_connection()
