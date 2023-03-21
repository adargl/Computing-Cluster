import ast
import pickle
import queue
import select
import socket
import threading
from enum import Enum
from copy import deepcopy
from struct import pack, unpack
from io import StringIO
from contextlib import redirect_stdout

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


class ClusterVisitor(ast.NodeVisitor):
    class Node:
        def __init__(self, node, container):
            self.node = node
            self.container = container
            if isinstance(node, ast.AST):
                self.lineno = node.lineno
                self.end_lineno = node.end_lineno
            else:
                self.lineno = 0
                self.end_lineno = 0
            self.snapshot = None

        def freeze(self):
            snapshot = ClusterVisitor.Node(deepcopy(self.node), deepcopy(self.container))
            self.snapshot = snapshot
            return snapshot

    class Loop(Node):
        def __init__(self, node, container, loop_type):
            super().__init__(node, container)
            self.loop_type = loop_type
            self.is_within_func = False
            self.function = None
            self.is_nested = False

        def __str__(self):
            return f"{self.loop_type}, Within_func: {self.is_within_func}, Nested: {self.is_nested}"

        @classmethod
        def empty_instance(cls):
            return cls(None, None, None)

        def set_within_func(self, function):
            self.is_within_func = True
            self.function = function

        def set_nested(self):
            self.is_nested = True

    class Function(Node):
        def __init__(self, node, container, name):
            super().__init__(node, container)
            self.name = name
            self.is_used = False
            self.modified_node = None

        def __str__(self):
            return f"Name: {self.name}, Used: {self.is_used}"

        @classmethod
        def empty_instance(cls):
            return cls(None, None, None)

        def set_used(self):
            self.is_used = True

    class Thread(Node):
        def __init__(self, node, container, args, func_name, name=None):
            super().__init__(node, container)
            self.name = name
            self.args = args
            self.func_name = func_name
            self.start_call = None
            self.join_call = None
            self.node_to_container = {node: container}

        def __str__(self):
            if self.name:
                return f"Name: {self.name}"
            return "Unnamed Thread"

        def take_snapshot(self):
            snapshots = [super().freeze(), self.start_call.freeze(), self.join_call.freeze()]
            return snapshots

        def set_start_call(self, call_node, container):
            self.start_call = ClusterVisitor.Node(call_node, container)

        def set_join_call(self, call_node, container):
            self.join_call = ClusterVisitor.Node(call_node, container)

    def __init__(self):
        self.names = {"parameters": "params", "mediator_class": "Mediator",
                      "mediator_object": "cluster", "processing_request": "send_request",
                      "template_change": "template_change", "await": "get_results",
                      "client_file": "shared.txt"}
        self.module_node = None

        self.current_func = self.Function.empty_instance()
        self.current_loop = self.Loop.empty_instance()
        self.current_container = None

        self.imports = {"import": [], "from": []}
        self.loops = list()
        self.threads = list()
        self.functions = dict()
        self.parameters = set()
        self.builtin_funcs = set()

    @staticmethod
    def has_body(ast_node):
        return isinstance(ast_node, (ast.Module, ast.For, ast.While, ast.FunctionDef, ast.If, ast.With, ast.Try,
                                     ast.ExceptHandler))

    def is_builtin(self, func_node):
        if not isinstance(func_node, ast.Call):
            return False
        return not (isinstance(func_node.func, ast.Name) and func_node.func.id in self.functions.keys())

    def generic_visit(self, node):
        if self.has_body(node):
            self.current_container = node
        super().generic_visit(node)

    def visit_Module(self, node):
        self.module_node = node
        self.generic_visit(node)

    def visit_Import(self, node):
        self.imports["import"].append(node)

    def visit_ImportFrom(self, node):
        self.imports["from"].append(node)

    def visit_FunctionDef(self, node, initial=True):
        # If the function is called from within the code (initial=False), perform recursive visiting
        # Otherwise if it's called on a function definition (initial=True) save its information

        # Update the current function variable with every visit
        if initial:
            function = self.Function(node, self.current_container, node.name)
            self.current_func = function
            # Save the function node and name
            self.functions[node.name] = function
        else:
            self.current_func = self.functions[node.name]
            self.generic_visit(node)

    def visit_For(self, node):
        self.visit_loop(node, ObjectType.FOR)

    def visit_While(self, node):
        self.visit_loop(node, ObjectType.WHILE)

    def visit_Assign(self, node):
        # Find out weather the assignment expression contains a thread
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) \
                and node.value.func.attr == 'Thread':
            name = node.targets[0]
            if isinstance(name, ast.Name):
                self.create_thread(node, name.id)
        else:
            if not self.current_loop.lineno < node.lineno <= self.current_loop.end_lineno:
                self.parameters.update(
                    [child_node.id for child_node in node.targets if isinstance(child_node, ast.Name) and
                     child_node.id not in self.builtin_funcs]
                )
            self.generic_visit(node)

    def visit_Expr(self, node):
        # Find out weather the expression contains a start() or join() call on a thread
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            function = node.value.func
            if isinstance(function.value, ast.Name):
                if function.attr == 'start' or function.attr == 'join':
                    for thread in self.threads:
                        if thread.name == function.value.id:
                            thread.set_start_call(node, self.current_container) \
                                if function.attr == 'start' else thread.set_join_call(node, self.current_container)
        self.generic_visit(node)

    def visit_Call(self, node):
        if not self.is_builtin(node):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            function = self.functions[func_name]
            function.set_used()

            if self.current_loop.lineno < node.lineno <= self.current_loop.end_lineno:
                loop = self.loops[-1]
                loop.set_nested()
                self.find_nested_functions(function.node)
            else:
                self.visit_FunctionDef(function.node, initial=False)
        else:
            name = node.func
            if isinstance(name, ast.Attribute) and isinstance(name.value, ast.Name):
                name = name.value
            self.builtin_funcs.add(name.id)
        self.generic_visit(node)

    def visit_loop(self, node, loop_type):
        if not self.current_loop.lineno < node.lineno <= self.current_loop.end_lineno:
            if loop_type == ObjectType.FOR:
                loop = self.Loop(node, self.current_container, ObjectType.FOR)
                if isinstance(node.target, ast.Name):
                    self.parameters.add(node.target.id)
            else:
                loop = self.Loop(node, self.current_container, ObjectType.WHILE)
            self.loops.append(loop)
            self.current_loop = loop
            if self.current_func.lineno < node.lineno <= self.current_func.end_lineno:
                loop.set_within_func(self.current_func)
        else:
            loop = self.loops[-1]
            loop.set_nested()
            if isinstance(node.target, ast.Name):
                self.builtin_funcs.add(node.target.id)

        self.generic_visit(node)

    def create_thread(self, assign_node, name):
        call_node = assign_node.value
        args = func_name = None
        for keyword in call_node.keywords:
            if keyword.arg == "target" and isinstance(keyword.value, ast.Name):
                func_name = keyword.value.id
            elif keyword.arg == "args" and isinstance(keyword.value, ast.Tuple):
                args = [name.id for name in keyword.value.elts if isinstance(name, ast.Name)
                        and name.id not in self.builtin_funcs]

        # Make sure that the thread function call is not recursive
        if not self.current_func.lineno < assign_node.lineno <= self.current_func.end_lineno \
                or func_name != self.current_func.name:
            thread = self.Thread(assign_node, self.current_container, args, func_name, name)
            self.threads.append(thread)
            function = self.functions[func_name]
            self.find_nested_functions(function.node)
            function.set_used()

    def find_nested_functions(self, node):
        for child_node in ast.walk(node):
            if isinstance(child_node, ast.Call) and isinstance(child_node.func, ast.Name) \
                    and child_node.func.id in self.functions.keys():
                function = self.functions[child_node.func.id]
                function.set_used()
                self.find_nested_functions(function.node)


class ClusterModifier(ast.NodeTransformer):
    def __init__(self, visitor):
        self.visitor = visitor
        self.names = visitor.names
        self.imports = [f for i in visitor.imports.values() for f in i]
        self.modified_functions = [function.node for function in visitor.functions.values() if function.is_used]
        self.original_functions = [deepcopy(function) for function in self.modified_functions]
        self.templates_count = 0
        self.templates = dict()
        self.global_nodes = list()
        self.instructions = list()
        self.current_params = set()
        self.all_params = set()
        self.max_concurrent_loops = 4

        # Create new nodes that do not depend on variables values
        self.assign_results_node = str_to_ast_node(
            f"exec('\\n'.join([f'{{key}} = {{value}}' for key, value in {self.names['parameters']}.items()]))"
        )
        self.assign_params_node = str_to_ast_node(
            f"globals().update({self.names['parameters']})"
        )

        self.setup_tree()

    @staticmethod
    def exhaust_generator(item, generator):
        for value in generator:
            if value is item:
                break

    @staticmethod
    def lineno_difference(node1, node2):
        return abs(node2.lineno - node1.end_lineno)

    @staticmethod
    def thread_to_instructions(thread):
        func_str = f"{thread.func_name}("
        for arg in thread.args:
            func_str += f"{arg}, "
        func_str = func_str[:-2] + ")"
        expr_node = str_to_ast_node(func_str)
        return expr_node

    @staticmethod
    def loop_to_instructions(loop):
        return loop.node.body

    @property
    def param_dict(self):
        param_dict = "{"
        for parameter in self.current_params:
            param_dict += f"'{parameter}': {parameter}, "
        param_dict = param_dict[:-2] + "}"

        return param_dict

    @property
    def processing_request_node(self):
        return str_to_ast_node(
            f"{self.names['mediator_object']}.{self.names['processing_request']}"
            f"({self.templates_count}, {self.param_dict})"
        )

    @property
    def get_results_node(self):
        return str_to_ast_node(
            f"{self.names['parameters']} = {self.names['mediator_object']}.{self.names['await']}"
            f"({self.templates_count})"
        )

    def assign_results_nodes(self, param_names):
        variable_name = "param_names"
        yield str_to_ast_node(f"{variable_name} = {self.names['parameters']}.keys()")

        for name in param_names:
            yield str_to_ast_node(
                f"if '{name}' in {variable_name}:"
                f"  {name} = {self.names['parameters']}['{name}']"
            )

    def params_to_instructions(self):
        for param in self.current_params:
            assign_node = str_to_ast_node(f"{param} = {self.names['parameters']}['{param}']")
            self.instructions.append(assign_node)

    def setup_tree(self):
        module_node = self.visitor.module_node
        count = 0
        # Import the mediator class
        new_import_node = str_to_ast_node(
            f"from {self.names['mediator_class']} import {self.names['mediator_class']}")
        module_node.body.insert(count, new_import_node)

        # Create and inject the mediator object
        while isinstance(module_node.body[count], (ast.FunctionDef, ast.Import, ast.ImportFrom)):
            count += 1
        new_obj_node = str_to_ast_node(f"{self.names['mediator_object']} = {self.names['mediator_class']}()")
        module_node.body.insert(count, new_obj_node)

    def modify_loop(self, loop):
        self.instructions = self.loop_to_instructions(loop)
        loop.body = self.processing_request_node
        self.generic_visit(loop.node,
                           "is_custom_visit", "add", add_after=loop.node)

    def modify_threads(self):
        sorted_by_container = dict()
        for thread in self.visitor.threads:
            if thread.start_call and thread.join_call:
                if not sorted_by_container.get(thread.join_call.container):
                    sorted_by_container[thread.join_call.container] = [thread]
                else:
                    sorted_by_container[thread.join_call.container].append(thread)
        self.instructions = (self.thread_to_instructions(thread) for _, threads in sorted_by_container.items()
                             for thread in threads)

        for container, threads in sorted_by_container.items():
            threads.sort(key=lambda t: t.join_call.lineno)
            last_thread = threads[0]
            thread_group = [last_thread]
            self.all_params.clear()
            self.all_params.update(last_thread.args)
            body = ast.iter_child_nodes(container)
            self.exhaust_generator(last_thread.join_call.node, body)

            for thread in threads[1:]:
                if self.lineno_difference(last_thread.join_call, thread.join_call) > 1:
                    for node in body:
                        if node is thread.join_call.node:
                            break
                        else:
                            for child in ast.walk(node):
                                if isinstance(child, ast.Name):
                                    if child.id in self.all_params:
                                        self.generic_visit(container, thread_group,
                                                           "is_custom_visit", "is_container_node", "visit_threads")
                                        thread_group = list()
                                        self.all_params.clear()
                                        self.exhaust_generator(thread.join_call.node, body)
                                        self.create_template()
                                        break
                            else:
                                continue
                            break
                else:
                    try:
                        next(body)
                    except StopIteration:
                        pass
                thread_group.append(thread)
                self.all_params.update(thread.args)

            self.generic_visit(container, thread_group,
                               "is_custom_visit", "is_container_node", "visit_threads")
            self.create_template()

    def custom_visit(self, obj):
        enum_type = ObjectType(type(obj))
        method = 'modify_' + str(enum_type)
        visitor = getattr(self, method)
        visitor(obj)

    def generic_visit(self, node, modified_nodes=None, *flags, **operations):
        allowed_flags = ["is_custom_visit", "is_container_node", "visit_threads"]
        allowed_operations = ["remove", "replace", "add", "add_after", "add_before"]
        conditions = {element: False for element in allowed_flags + allowed_operations}

        # Extract the arguments that were supplied
        for flag in flags:
            if flag not in allowed_flags:
                raise ValueError(f"Invalid flag type {flag}."
                                 f" Expected one of: {', '.join(allowed_flags)}")
            else:
                conditions[flag] = True
        for operation, value in operations.items():
            if operation not in allowed_operations:
                raise ValueError(f"Invalid operation type {operation}."
                                 f" Expected one of: {', '.join(allowed_operations)}")
            else:
                conditions[operation] = True

        # Override the functionality of generic_visit()
        if conditions["is_custom_visit"]:
            if conditions["is_container_node"]:

                # If other flags are set than node.body should be created with additional logic
                if conditions["visit_threads"]:
                    new_body = []
                    last_join_call = modified_nodes[-1].join_call.node
                    operations["add"] = last_join_call
                    for n in node.body:
                        new_n = self.generic_visit(n, modified_nodes, "is_custom_visit", "visit_threads",
                                                   **operations)
                        if not new_n:
                            continue
                        elif new_n is last_join_call:
                            new_body.append(self.get_results_node)
                            new_body.extend(self.assign_results_nodes(self.all_params))
                        else:
                            new_body.append(new_n)

                    node.body = new_body

                elif conditions["add"]:
                    if is_iterable(modified_nodes):
                        raise AttributeError("'modified nodes' must not be an iterable when preforming "
                                             "add operations")

                    new_body = []
                    for n in node.body:
                        new_n = self.generic_visit(n, modified_nodes, "is_custom_visit", **operations)
                        if not new_n:
                            continue
                        elif new_n is operations.get("add_before"):
                            new_body.append(operations["add"])
                        new_body.append(new_n)
                        if new_n is operations.get("add_after"):
                            new_body.append(operations["add"])
                    node.body = new_body

                # Otherwise list comprehension is enough
                else:
                    node.body = [
                        self.generic_visit(n, modified_nodes, "is_custom_visit", **operations)
                        for n in node.body if n is not None
                    ]
            else:
                if conditions["visit_threads"]:
                    for thread in modified_nodes:
                        if node is thread.node:
                            return None
                        elif node is thread.start_call.node:
                            self.current_params = thread.args
                            return self.processing_request_node
                        elif node is operations["add"]:
                            return node
                        elif node is thread.join_call.node:
                            return None
                    return node

                elif node in modified_nodes:
                    if conditions["remove"]:
                        return None
                    elif conditions["replace"]:
                        return operations["replace"]

            return node

        # If the visit is not custom use the original generic_visit method
        return super().generic_visit(node)

    def visit_Name(self, node):
        if node.id in self.visitor.current_params and not \
                (node.id in self.visitor.functions.keys() or node.id in self.visitor.builtin_funcs):
            self.current_params.add(node.id)
        self.generic_visit(node)
        return node

    def visit_Global(self, node):
        self.global_nodes.append(node)
        self.generic_visit(node)
        return None

    def create_template(self):
        template = ast.Module(
            body=self.imports + self.original_functions + [self.assign_params_node],
            type_ignores=[]
        )
        self.templates[self.templates_count] = TaskMaker(
            template, len(self.imports + self.original_functions) + 1, self.instructions, self.names['parameters']
        )

        self.templates_count += 1

    def provide_response(self):
        for loop in self.visitor.loops:
            if loop.is_nested:
                self.generic_visit(loop.node)
                self.modify_loop(loop)
                self.create_template()
                self.clear_data()

        self.modify_threads()

    def clear_data(self):
        self.current_params.clear()
        self.global_nodes = []
        self.instructions = []


class ObjectType(Enum):
    FOR = ast.For
    WHILE = ast.While
    LOOP = ClusterVisitor.Loop
    FUNCTION = ClusterVisitor.Function
    THREAD = ClusterVisitor.Thread

    def __str__(self):
        return self.name.lower()


class ClusterServer:
    # number  |  operation code
    #  0      |  reserved
    #  1      |  processing request
    #  2      |  processing response
    #  3      |  get results
    #  4      |  node available
    #  5      |  send to cluster
    #  6      |  return response
    #  7      |  initiate connection

    def __init__(self, port=55555, send_format="utf-8", buffer_size=1024, max_queue=5):
        self.ip = "0.0.0.0"
        self.port = port
        self.addr = (self.ip, self.port)
        self.format = send_format
        self.buffer = buffer_size
        self.max_queue = max_queue
        self.main_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.client_sock = None
        self.read_socks = list()
        self.write_socks = list()
        self.templates = list()
        self.node_queue = queue.Queue()
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()

        self.task_condition = threading.Condition()
        self.result_condition = threading.Condition()
        self.lock = threading.Lock()
        self.requests = dict()
        self.responses = dict()
        self.current_result_id = 0
        self.close_connections = False

        self.init_connection()

    def set_partition(self, cluster_trees):
        self.templates = cluster_trees

    def init_connection(self):
        self.main_sock.bind(self.addr)
        self.main_sock.listen(self.max_queue)
        self.read_socks = [self.main_sock]
        self.write_socks = []
        request_handler = threading.Thread(target=self.handle_tasks)
        response_handler = threading.Thread(target=self.handle_results)
        request_handler.start()
        response_handler.start()

    def handle_connection(self):
        while not self.close_connections:
            readable, writeable, exceptions = select.select(self.read_socks, self.write_socks, self.read_socks)
            for sock in readable:
                if sock is self.main_sock:
                    connection, client_addr = sock.accept()
                    ip, port = client_addr
                    logger.info(f"[CONNECTION ESTABLISHED] connection has been established from {ip}, {port}")
                    logger.info(f"[ACTIVE CONNECTIONS] {len(self.read_socks)} active connection(s)")
                    self.read_socks.append(connection)
                else:
                    packet = self.recv_msg(sock)
                    if packet:
                        msg_fmt = None
                        if sock not in self.write_socks:
                            self.write_socks.append(sock)
                        op_code, reserved, data = packet
                        if op_code == 1:
                            template_id = reserved
                            self.task_queue.put((template_id, data))
                            with self.lock:
                                self.responses[template_id] = dict()
                            msg_fmt = f"processing request (id={template_id})"
                        elif op_code == 3:
                            task_group_id = reserved
                            logger.info(f"[DATA RECEIVED] get results request (id={task_group_id})")
                        elif op_code == 4:
                            self.node_queue.put(sock)
                            with self.task_condition:
                                self.task_condition.notify()
                            ip, port = sock.getsockname()
                            logger.info(f"[TASK FINISHED] {ip} executed {data} task(s)")
                            logger.info(f"[TASK FINISHED] {ip} is currently available")
                        elif op_code == 6:
                            task_id = reserved
                            msg_fmt = f"process result (id={task_id})"
                            with self.lock:
                                original = self.requests[sock]
                            new = {p: data[p] for p in original if p in data and not original[p] == data[p]}
                            if new:
                                with self.lock:
                                    response = self.responses[task_id]
                                    if response:
                                        for key in response.keys():
                                            org_value, new_value = original[key], new[key]
                                            if type(org_value) == type(new_value):
                                                if isinstance(org_value, list):
                                                    for i in range(len(new_value)):
                                                        if org_value[i] != new_value[i]:
                                                            response[key][i] = new_value[i]
                                                    else:
                                                        org_value.extend(new_value[i:])
                                            else:
                                                response[key] = new[key]
                                    else:
                                        response.update(new)
                        elif op_code == 7:
                            self.client_sock = sock
                            ip, port = sock.getsockname()
                            logger.info(f"[NEW CLIENT CONNECTION] {ip} connected as a client")
                        if msg_fmt:
                            logger.info(f"[DATA RECEIVED] {msg_fmt}: {data}")
                    else:
                        self.close_sock(sock)

            for sock in exceptions:
                self.close_sock(sock)
        for sock in self.read_socks:
            self.close_sock(sock)

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

    def handle_tasks(self):
        op_code = 5
        while True:
            if self.task_queue.empty():
                with self.lock:
                    response = self.responses.get(self.current_result_id)
                if response:
                    self.result_queue.put(response)
                    with self.task_condition:
                        self.task_condition.wait()
                    with self.result_condition:
                        self.result_condition.notify()
                with self.task_condition:
                    self.task_condition.wait()
            elif self.node_queue.empty():
                with self.task_condition:
                    self.task_condition.wait()
            else:
                sock = self.node_queue.get()
                if sock in self.write_socks:
                    data = self.task_queue.get()
                    if not data:
                        self.node_queue.put(sock)
                        break
                    template_id, params = data

                    # Release a result
                    if self.current_result_id < template_id:
                        with self.lock:
                            response = self.responses[self.current_result_id]
                        self.result_queue.put(response)
                        with self.result_condition:
                            self.result_condition.notify()

                    task = self.templates[template_id]
                    task.params = params
                    self.send_msg(sock, op_code, task.to_executable(), template_id)
                    with self.lock:
                        self.requests[sock] = params

    def handle_results(self):
        op_code = 2
        while True:
            with self.result_condition:
                self.result_condition.wait()
            response = self.result_queue.get()
            self.send_msg(self.client_sock, op_code, response)
            logger.info(f"[RESPONSE SENT] response parameters (id={self.current_result_id}): {response}")
            self.current_result_id += 1

    def close_sock(self, sock):
        if sock in self.write_socks:
            self.write_socks.remove(sock)
        self.read_socks.remove(sock)
        ip, port = sock.getsockname()
        sock.close()
        logger.info(f"[CONNECTION CLOSED] connection from {ip}, {port} has been closed")


class TaskMaker:
    def __init__(self, tree, inject_index, instructions, params_name):
        self.tree = tree
        self.index = inject_index
        self.instructions = instructions
        self.params_name = params_name
        self.params = None

    @property
    def instruction_block(self):
        return next(self.instructions)

    @staticmethod
    def extend_in_index(lst, iterable, index):
        if is_iterable(iterable):
            for item in iterable:
                lst.insert(index, item)
        else:
            lst.insert(index, iterable)

    def finalize(self):
        tree = deepcopy(self.tree)
        block = self.instruction_block
        self.extend_in_index(tree.body, block, self.index)

        tmp_file = "Created.py"
        with open(tmp_file, 'w') as f:
            f.write(ast.unparse(tree))

        return ast.parse(ast.unparse(tree))

    def to_executable(self):
        return ExecutableTree(self.finalize(), self.params_name, self.params)


class ExecutableTree:
    def __init__(self, tree, params_name, params):
        self.tree = tree
        self.params_name = params_name
        self.params = params

    def exec_tree(self, file_name=''):
        exec(compile(self.tree, file_name, 'exec'), {self.params_name: self.params})


def str_to_ast_node(string: str):
    module = ast.parse(string)
    if len(module.body) == 1:
        node = module.body[0]
        return node
    logger.error("String exceeded the allowed amount of ast nodes")
    return None


def is_iterable(obj: object):
    return hasattr(obj, '__iter__')


def exec_tree(tree, file_name=''):
    std_out = StringIO()
    with redirect_stdout(std_out):
        exec(compile(tree, file_name, 'exec'), {'builtins': globals()['__builtins__']})
    # Remove the last char from the output since it is always '\n'
    logger.critical("[EXECUTION FINISHED] Program output:\n" + std_out.getvalue()[:-1])


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
    file_handler.setLevel(logger.level)
    file_handler.setFormatter(logging.Formatter(*fmt))
    logger.addHandler(file_handler)

    file = "Examples/CaesarCipher.py"
    modified_file = "Modified.py"
    byproduct_file = "Created.py"
    with open(file) as source:
        ast_tree = ast.parse(source.read())

    Visitor = ClusterVisitor()
    Visitor.visit(ast_tree)

    logger.debug(f"Loops {[str(loop) for loop in Visitor.loops]}")
    logger.debug(f"Functions {[str(value) for value in Visitor.functions.values()]}")
    logger.debug(f"Threads {[str(thread) for thread in Visitor.threads]}")
    logger.debug(f"Parameters {Visitor.parameters}")

    Modifier = ClusterModifier(Visitor)
    Modifier.provide_response()
    cluster_partitions = Modifier.templates

    logger.debug(f"Templates {cluster_partitions}")

    if cluster_partitions:
        logger.warning(f"File {file} is breakable")

        server = ClusterServer()
        server.set_partition(cluster_partitions)
        server_thread = threading.Thread(target=server.handle_connection)
        server_thread.start()

        modified_code = ast.unparse(ast_tree)
        with open(modified_file, 'w') as output_file:
            output_file.write(modified_code)
        exec_tree(ast.parse(modified_code))
    else:
        logger.warning(f"File {file} is unbreakable")
