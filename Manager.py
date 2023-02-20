import ast
import pickle
import queue
import select
import socket
import time
from copy import deepcopy
from struct import pack, unpack
from threading import Thread
from enum import Enum

import astpretty
import logging


class ClusterVisitor(ast.NodeVisitor):
    class Loop:
        def __init__(self, node, start_line, end_line, container, loop_type):
            self.node = node
            self.start_line = start_line
            self.end_line = end_line
            self.container = container
            self.loop_type = loop_type
            self.is_within_func = False
            self.function = None
            self.is_nested = False
            self.node_copy = None

        def __str__(self):
            return f"{self.loop_type}, Within_func: {self.is_within_func}, Nested: {self.is_nested}"

        def set_within_func(self, function):
            self.is_within_func = True
            self.function = function

        def set_nested(self):
            self.is_nested = True

        def make_node_copy(self):
            self.node_copy = deepcopy(self.node)

    class Function:
        def __init__(self, node, name, start_line, end_line, container):
            self.node = node
            self.name = name
            self.start_line = start_line
            self.end_line = end_line
            self.container = container
            self.is_used = False
            self.modified_node = None

        def __str__(self):
            return f"Name: {self.name}, Used: {self.is_used}"

        def set_used(self):
            self.is_used = True

        def replace_node_with_copy(self):
            self.modified_node = self.node
            self.node = deepcopy(self.node)

    class CThread:
        def __init__(self, node, args, container, name=None):
            self.node = node
            self.name = name
            self.args = args
            self.container = container
            self.start_call = None
            self.join_call = None

        def __str__(self):
            if self.name:
                return f"Name: {self.name}"
            return "Unnamed Thread"

        def set_start_call(self, call_node):
            self.start_call = call_node

        def set_join_call(self, call_node):
            self.join_call = call_node

    def __init__(self):
        """
        parameters: a variable name that is used to store the parameter dictionary
        mediator_class: a name of the file and class form which the mediating object is imported
        mediator_object: a variable name that stores the mediating object instance
        processing_request: a function name of the mediating class for a processing request
        template_change: a function name of the mediating class for the a template change
        await: a function name of the mediating class for awaiting a response
        client_file: a file name in which the client stores data temporarily
        """
        self.names = {"parameters": "params", "mediator_class": "Mediator",
                      "mediator_object": "mediator", "processing_request": "processing_request",
                      "template_change": "template_change", "await": "await_response",
                      "client_file": "shared.txt"}
        self.module_node = None

        self.current_func = self.Function(None, None, 0, 0, None)
        self.current_loop = self.Loop(None, 0, 0, None, None)
        self.current_container = None

        self.imports = {"import": [], "from": []}
        self.loops = list()
        self.threads = list()
        self.functions = dict()
        self.parameters = set()
        self.builtin_funcs = set()

    def generic_visit(self, node):
        if has_body(node):
            self.current_container = node
        super().generic_visit(node)

    def visit_Module(self, node):
        self.module_node = node
        count = 0
        # Import the mediator class
        new_import_node = string_to_ast_node(
            f"from {self.names['mediator_class']} import {self.names['mediator_class']}")
        node.body.insert(count, new_import_node)

        # Create and inject the mediator object
        while isinstance(node.body[count], (ast.FunctionDef, ast.Import, ast.ImportFrom)):
            count += 1
        new_obj_node = string_to_ast_node(f"{self.names['mediator_object']} = {self.names['mediator_class']}()")
        node.body.insert(count, new_obj_node)

        self.generic_visit(node)

    def visit_Import(self, node):
        self.imports["import"].append(node)

    def visit_ImportFrom(self, node):
        self.imports["from"].append(node)

    def visit_FunctionDef(self, node, initial=True):
        # If the function is called from within the code (initial=False), perform recursive visiting
        # Otherwise if its called on a function definition (initial=True) save it's information

        # Update the current function variable with every visit
        if initial:
            function = self.Function(node, node.name, node.lineno, node.end_lineno, self.current_container)
            self.current_func = function
            # Save the function node and name
            self.functions[node.name] = function
        else:
            self.current_func = self.functions[node.name]
            self.generic_visit(node)

    def visit_For(self, node):
        self.visit_Loop(node, ObjectType.FOR)

    def visit_While(self, node):
        self.visit_Loop(node, ObjectType.WHILE)

    def visit_Loop(self, node, loop_type):
        if not self.current_loop.start_line < node.lineno <= self.current_loop.end_line:
            if loop_type == ObjectType.FOR:
                loop = self.Loop(node, node.lineno, node.end_lineno, self.current_container, ObjectType.FOR)
                if isinstance(node.target, ast.Name):
                    self.parameters.add(node.target.id)
            else:
                loop = self.Loop(node, node.lineno, node.end_lineno, self.current_container, ObjectType.WHILE)
            self.loops.append(loop)
            self.current_loop = loop
            if self.current_func.start_line < node.lineno <= self.current_func.end_line:
                loop.set_within_func(self.current_func)
        else:
            loop = self.loops[-1]
            loop.set_nested()
            if isinstance(node.target, ast.Name):
                self.builtin_funcs.add(node.target.id)

        self.generic_visit(node)

    def visit_Assign(self, node):
        # Find out weather the assignment expression contains a thread
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) \
                and node.value.func.attr == 'Thread':
            self.create_Thread(node, node.targets[0].id)
        else:
            if not self.current_loop.start_line < node.lineno <= self.current_loop.end_line:
                self.parameters.update(
                    [child_node.id for child_node in node.targets if isinstance(child_node, ast.Name)])
            self.generic_visit(node)

    def visit_Expr(self, node):
        # Find out weather the expression contains a start() or join() call on a thread
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            function = node.value.func
            if isinstance(function.value, ast.Name):
                if function.attr == 'start' or function.attr == 'join':
                    for thread in self.threads:
                        if thread.name == function.value.id:
                            thread.set_start_call(node) if function.attr == 'start' else thread.set_join_call(node)
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in self.functions.keys():
            func_name = node.func.id
            function = self.functions[func_name]
            function.set_used()

            if self.current_loop.start_line < node.lineno <= self.current_loop.end_line:
                if node.func.id in self.functions.keys():
                    loop = self.loops[-1]
                    loop.set_nested()
                    self.find_nested_functions(function.node)
                else:
                    self.builtin_funcs.add(node.func.id)
            else:
                self.visit_FunctionDef(function.node, initial=False)
        self.generic_visit(node)

    def create_Thread(self, assign_node, name):
        call_node = assign_node.value
        args = None
        for keyword in call_node.keywords:
            if keyword.arg == "args" and isinstance(keyword.value, ast.Tuple):
                args = {name_node.id for name_node in ast.walk(keyword) if isinstance(name_node, ast.Name)}
                break
        thread = self.CThread(assign_node, args, self.current_container, name)
        self.threads.append(thread)

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
        self.templates = []
        self.global_nodes = []
        self.params_assign = []
        self.func_call = None
        self.parameters = set()
        self.max_concurrent_loops = 4

        self.global_mediator_node = string_to_ast_node(f"global {self.visitor.names['mediator_object']}")
        self.update_params_node = string_to_ast_node(f"subarrays = params['subarrays']")
        self.change_template_node = string_to_ast_node(
            f"{self.visitor.names['parameters']} = {self.visitor.names['mediator_object']}."
            f"{self.visitor.names['template_change']}()")

    @property
    def param_dict(self):
        param_dict = "{"
        for parameter in self.parameters:
            param_dict += f"'{parameter}': {parameter}, "
            assign_node = string_to_ast_node(f"{parameter} = {self.visitor.names['parameters']}['{parameter}']")
            self.params_assign.append(assign_node)
        param_dict = param_dict[:-2] + "}"

        return param_dict

    @property
    def params_assign_node(self):
        return string_to_ast_node(f"{self.visitor.names['parameters']} = {self.param_dict}")

    @property
    def await_response_node(self):
        return string_to_ast_node(
            f"{self.visitor.names['parameters']} = {self.visitor.names['mediator_object']}."
            f"{self.visitor.names['await']}()"
        )

    @property
    def execute_on_cluster_node(self):
        return string_to_ast_node(
            f"{self.visitor.names['mediator_object']}.{self.visitor.names['processing_request']}({self.param_dict})"
        )

    def custom_visit(self, obj):
        enum_type = ObjectType(type(obj))
        method = 'modify_' + str(enum_type)
        visitor = getattr(self, method)
        visitor(obj)

    def modify_loop(self, loop):
        loop.make_node_copy()

        if loop.is_within_func:
            func_node = loop.function.node
            loop.function.replace_node_with_copy()
            body = func_node.body
            body.insert(0, self.global_nodes)
            body.insert(0, self.global_mediator_node)
            insertion_index = get_node_locations(func_node, loop.node)
        else:
            body = self.visitor.module_node.body
            insertion_index = get_node_locations(self.visitor.module_node, loop.node)
        body.insert(insertion_index + 1, self.change_template_node)

        if loop.loop_type == self.visitor.Type.FOR:
            body.insert(insertion_index + 1, self.update_params_node)
            loop.node.body = [self.execute_on_cluster_node]
        elif loop.loop_type == self.visitor.Type.WHILE:
            loop.node.body = [self.params_assign, self.execute_on_cluster_node, self.await_response_node,
                              self.update_params_node]

        return loop

    def modify_thread(self, thread):
        for arg_name in thread.args:
            self.parameters.add(arg_name)

        if thread.start_call and thread.join_call:
            start_node, join_node = thread.start_call, thread.join_call
            container_node = thread.container
            self.generic_visit(container_node, True, True, start_node, replacement_node=self.execute_on_cluster_node)
            self.generic_visit(container_node, True, True, join_node, replacement_node=self.await_response_node)
        elif thread.start_call:
            pass
        elif thread.join_call:
            pass

    def generic_visit(self, node, to_modify=False, is_container=False, currently_modified_node=None, remove_node=False,
                      replacement_node=None):
        if to_modify:
            if node is currently_modified_node:
                if remove_node:
                    return None
                elif replacement_node:
                    return replacement_node
            else:
                if is_container:
                    node.body = [
                        self.generic_visit(n, to_modify, False, currently_modified_node, remove_node, replacement_node)
                        for n in node.body if n is not None
                    ]
            return node

        return super().generic_visit(node)

    def visit_Name(self, node):
        if node.id in self.visitor.parameters and not \
                (node.id in self.visitor.functions.keys() or node.id in self.visitor.builtin_funcs):
            self.parameters.add(node.id)
        self.generic_visit(node)
        return node

    def visit_Global(self, node):
        self.global_nodes.append(node)
        self.generic_visit(node)
        return None

    def create_expr_from_thread(self, thread):
        func_name = thread.node.targets[0].id
        self.func_call = f"{func_name}("
        for parameter in self.parameters:
            self.func_call += f"{parameter}, "
            assign_node = string_to_ast_node(f"{parameter} = {self.visitor.names['parameters']}['{parameter}']")
            self.params_assign.append(assign_node)
        self.func_call = self.func_call[:-2] + ")"

    # def create_nodes(self):
    #     self.global_mediator_node = string_to_ast_node(f"global {self.visitor.names['mediator_object']}")
    #     self.execute_on_cluster_node = string_to_ast_node(
    #         f"{self.visitor.names['mediator_object']}.{self.visitor.names['processing_request']}"
    #         f"({self.visitor.names['parameters']})")
    #     self.await_response_node = string_to_ast_node(
    #         f"{self.visitor.names['parameters']} = {self.visitor.names['mediator_object']}."
    #         f"{self.visitor.names['await']}()")
    #     # self.update_params = string_to_ast_node(
    #     #     f"for key, val in {self.visitor.names['parameters']}.items():\n"
    #     #     f"  exec(key + '=val')")
    #     # self.update_params = string_to_ast_node(
    #     #     f"for key, val in {self.visitor.names['parameters']}.items():\n"
    #     #     f"  globals()[key] = val")
    #     self.update_params_node = string_to_ast_node(f"subarrays = params['subarrays']")
    #     self.change_template_node = string_to_ast_node(
    #         f"{self.visitor.names['parameters']} = {self.visitor.names['mediator_object']}."
    #         f"{self.visitor.names['template_change']}()")

    def create_template(self, loop=None, thread=None):
        open_file_node = [string_to_ast_node(
            f"""with open('{self.visitor.names['client_file']}', 'w') as file:
        file.write(str({self.param_dict}))""")]

        imports = [j for i in self.visitor.imports.values() for j in i]
        functions = [function.node for function in self.visitor.functions.values() if function.is_used]
        instructions = None
        if loop:
            instructions = loop.node_copy.body
        instructions = [self.func_call]

        template = ast.Module(
            body=imports + functions + self.params_assign + instructions + open_file_node,
            type_ignores=[]
        )
        self.templates.append(ExecutableTree(template, len(imports + functions), self.visitor.names['parameters']))

    def provide_response(self):
        for loop in self.visitor.loops:
            if loop.is_nested:
                self.generic_visit(loop.node)
                loop_copy = self.modify_loop(loop)
                self.create_template(loop=loop_copy)

                self.clear_data()

        for thread in self.visitor.threads:
            if thread.name:
                self.custom_visit(thread)
                self.create_expr_from_thread(thread)
                self.create_template()

                self.clear_data()

    def clear_data(self):
        self.parameters.clear()
        self.global_nodes = []
        self.params_assign = []


class ObjectType(Enum):
    FOR = ast.For
    WHILE = ast.While
    LOOP = ClusterVisitor.Loop
    FUNCTION = ClusterVisitor.Function
    THREAD = ClusterVisitor.CThread

    def __str__(self):
        return self.name.lower()


class Server:
    # number  |  operation code
    #  0      |  reserved
    #  1      |  run file request
    #  2      |  run file response
    #  3      |  change template
    #  4      |  node is ready
    #  5      |  cluster request
    #  6      |  cluster response

    def __init__(self, port=55555, send_format="utf-8", buffer_size=1024, max_queue=5):
        self.ip = socket.gethostbyname(socket.gethostname())
        self.port = port
        self.addr = (self.ip, self.port)
        self.format = send_format
        self.buffer = buffer_size
        self.max_queue = max_queue
        self.main_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.read_socks = list()
        self.write_socks = list()
        self.templates = list()
        self.insertion_index = 0
        self.clients_queue = queue.Queue()
        self.task_queue = queue.Queue()

        self.sock_to_tree = dict()
        self.responses = list()
        self.client_socks = list()

        self.operation_finished = False

    def set_partition(self, cluster_trees):
        self.templates = cluster_trees
        self.responses = [dict() for _ in range(len(self.templates))]

    def init_connection(self):
        self.main_sock.bind(self.addr)
        self.main_sock.listen(self.max_queue)

        self.read_socks = [self.main_sock]
        self.write_socks = []
        self.handle_connection()

    def handle_connection(self):
        while not self.operation_finished:
            readable, writeable, exceptions = select.select(self.read_socks, self.write_socks, self.read_socks)
            for sock in readable:
                if sock is self.main_sock:
                    connection, client_addr = sock.accept()
                    logging.info(f"[CONNECTION ESTABLISHED] connection has been established from {client_addr}...")
                    logging.info(f"[ACTIVE CONNECTIONS] {len(self.read_socks)} active connections...")
                    self.read_socks.append(connection)
                else:
                    packet = self.recv_msg(sock)
                    msg_type = None
                    if packet:
                        if sock not in self.write_socks:
                            self.write_socks.append(sock)
                        data, op_code = packet
                        if op_code == 1:
                            self.client_socks.append(sock)
                            self.task_queue.put(data)
                            msg_type = "Processing Request"
                        elif op_code == 3:
                            Thread(target=self.handle_results, args=(sock,)).start()
                            msg_type = "Change Template Request"
                        elif op_code == 4:
                            self.clients_queue.put(sock)
                            msg_type = "Available For Processing. Executed Templates"
                        elif op_code == 6:
                            msg_type = "Process result"
                            org = self.sock_to_tree[sock]
                            new = {k: data[k] for k in org if k in data and not org[k] == data[k]}
                            if new:
                                self.responses[self.insertion_index].update(new)
                            # While loop addition
                            op_code = 2
                            self.send_msg(self.client_socks[0], op_code, new)
                            logging.info(f"[DATA RECEIVED] {msg_type}: {data}")
                            logging.info(f"[RESPONSE SENT] Parameters: {new}")
                            msg_type = None
                        if msg_type:
                            logging.info(f"[DATA RECEIVED] {msg_type}: {data}")
                    else:
                        self.close_sock(sock)

            for sock in exceptions:
                self.close_sock(sock)
        for sock in self.read_socks:
            self.close_sock(sock)

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

    def handle_tasks(self):
        op_code = 5
        while True:
            time.sleep(0.01)
            if not self.clients_queue.empty():
                sock = self.clients_queue.get()
                if sock in self.write_socks:
                    params = self.task_queue.get()
                    if not params:
                        self.clients_queue.put(sock)
                        break
                    partition = self.templates[self.insertion_index]
                    partition.set_params(str(params))
                    tree = partition.finalize()
                    self.send_msg(sock, op_code, tree)
                    self.sock_to_tree[sock] = params
        self.operation_finished = True

    def handle_results(self, sock):
        while not self.task_queue.empty():
            time.sleep(0.01)
        op_code = 2
        response = self.responses[self.insertion_index]
        self.send_msg(sock, op_code, response)
        logging.info(f"[RESPONSE SENT] Parameters: {response}")

        if len(self.templates) - 1 == self.insertion_index:
            self.task_queue.put(None)
        else:
            self.insertion_index += 1

    def close_sock(self, sock):
        if sock in self.write_socks:
            self.write_socks.remove(sock)
        self.read_socks.remove(sock)
        sock.close()
        logging.info(f"[CONNECTION CLOSED] {sock} has been closed...")


class ExecutableTree:
    def __init__(self, tree, inject_index, params):
        self.tree = tree
        self.index = inject_index
        self.params_name = params
        self.params = None

    def set_params(self, params):
        self.params = params

    def finalize(self):
        tree = deepcopy(self.tree)
        new_assign_node = f"{self.params_name} = " + str(self.params)
        param_node = string_to_ast_node(new_assign_node)
        tree.body.insert(self.index, param_node)

        tmp_file = "Template_File.py"
        with open(tmp_file, 'w') as f:
            f.write(ast.unparse(tree))

        return ast.parse(ast.unparse(tree))


def string_to_ast_node(string: str):
    module = ast.parse(string)
    if len(module.body) == 1:
        return module.body[0]
    logging.error("String acceded the allowed amount of ast nodes")
    return None


def get_node_locations(tree, nodes: list, results: dict):
    for node in nodes:
        if node in tree.body:
            results[node] = (tree.body.index(node), tree.body)
            nodes = [new_node for new_node in nodes if new_node is not node]

    if not nodes:
        return

    for child_node in ast.iter_child_nodes(tree):
        if has_body(child_node):
            for node in nodes:
                if node in tree.body:
                    results[node] = (child_node.body.index(node), child_node.body)
                    nodes = [new_node for new_node in nodes if new_node is not node]
            get_node_locations(child_node, nodes, results)


def has_body(ast_node: object):
    return isinstance(ast_node, (ast.Module, ast.For, ast.While, ast.FunctionDef, ast.If, ast.With, ast.Try,
                                 ast.ExceptHandler))


def exec_tree(tree, file_name=''):
    exec(compile(tree, file_name, 'exec'), globals())


def print_tree(tree):
    astpretty.pprint(tree)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s:%(message)s', datefmt='%I:%M:%S %p', level=logging.INFO)

    file = "Examples/Threaded_MergeSort"
    modified_file = "Modified_File.py"
    template_file = "Template_File.py"
    with open(file) as source:
        ast_tree = ast.parse(source.read())

    Visitor = ClusterVisitor()
    Visitor.visit(ast_tree)
    Modifier = ClusterModifier(Visitor)
    Modifier.provide_response()
    cluster_partitions = Modifier.templates

    logging.debug(f"Loops {[str(loop) for loop in Visitor.loops]}")
    logging.debug(f"Functions {[str(value) for value in Visitor.functions.values()]}")
    logging.debug(f"Threads {[str(thread) for thread in Visitor.threads]}")
    logging.debug(f"Parameters {Visitor.parameters}")
    logging.debug(f"Templates {cluster_partitions}")

    if cluster_partitions:
        server = Server()
        server.set_partition(cluster_partitions)
        request_handler = Thread(target=server.handle_tasks)
        request_handler.start()
        server_thread = Thread(target=server.init_connection)
        server_thread.start()

        modified_code = ast.unparse(ast_tree)
        with open(modified_file, 'w') as output_file:
            output_file.write(modified_code)
        exec_tree(ast.parse(modified_code))
        # logging.warning(f"\n--- File {file} is breakable ---")
    else:
        logging.warning(f"\n--- File {file} is unbreakable ---")
