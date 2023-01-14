import ast
import pickle
import queue
import select
import socket
import time
from copy import deepcopy
from struct import pack, unpack
from threading import Thread

import astpretty
import logging


class ClusterVisitor(ast.NodeVisitor):
    class Loop:
        def __init__(self, node, start_line, end_line):
            self.node = node
            self.start_line = start_line
            self.end_line = end_line
            self.is_within_func = False
            self.function = None
            self.is_nested = False

        def __str__(self):
            return f"Within_func: {self.is_within_func}, Nested: {self.is_nested} |"

        def set_within_func(self, function):
            self.is_within_func = True
            self.function = function

        def set_nested(self):
            self.is_nested = True

    class Function:
        def __init__(self, node, name, start_line, end_line):
            self.node = node
            self.name = name
            self.start_line = start_line
            self.end_line = end_line
            self.is_used = False

        def __str__(self):
            return f"Name: {self.name}, Used: {self.is_used} |"

        def set_used(self):
            self.is_used = True

    def __init__(self):
        self.module_node = None
        self.names = {"global_vars": "global_variables", "mediator_class": "ClusterMediator",
                      "mediator_object": "mediator", "helper_func": "send_to_cluster", "request_func": "new_request",
                      "end_template_func": "template_finished", "client_file": "shared.txt"}

        self.helper_func_node = None
        self.current_func = self.Function(None, None, 0, 0)
        self.current_loop = self.Loop(None, 0, 0)

        self.imports = {"import": [], "from": []}
        self.loops = list()
        self.functions = dict()  # {"function name": (func_node, True if used False otherwise)}
        self.variables = set()  # {"variable name"}
        self.builtin_funcs = set()  # {"function name"}

    # def function_name(func):
    #     def wrapper(*args, **kwargs):
    #         logging.debug("Running:", func.__name__)
    #         func(*args, **kwargs)
    #
    #     return wrapper

    def visit_Module(self, node):
        self.module_node = node
        count = 0
        # inject import
        new_import_node = string_to_ast_node(
            f"from {self.names['mediator_class']} import {self.names['mediator_class']}")
        node.body.insert(count, new_import_node)

        while isinstance(node.body[count], (ast.Import, ast.ImportFrom)):
            count += 1
        self.helper_func_node = string_to_ast_node(
            f"""def {self.names['helper_func']}({self.names['global_vars']}):
        global {self.names['mediator_object']}
        {self.names['mediator_object']}.{self.names['request_func']}({self.names['global_vars']})""")

        node.body.insert(count, self.helper_func_node)
        while isinstance(node.body[count], ast.FunctionDef):
            count += 1
        new_obj_node = string_to_ast_node(f"{self.names['mediator_object']} = {self.names['mediator_class']}()")
        node.body.insert(count, new_obj_node)

        self.generic_visit(node)

    def visit_Import(self, node):
        self.imports["import"].append(node)

    def visit_ImportFrom(self, node):
        self.imports["from"].append(node)

    def visit_FunctionDef(self, node, initial=True):
        # Update the current function with every new definition
        if initial:
            function = self.Function(node, node.name, node.lineno, node.end_lineno)
            self.current_func = function
            # Add to tree functions
            self.functions[node.name] = function
        else:
            self.current_func = self.functions[node.name]
            self.generic_visit(node)

    def visit_For(self, node):
        if not self.current_loop.start_line < node.lineno <= self.current_loop.end_line:
            loop = self.Loop(node, node.lineno, node.end_lineno)
            self.loops.append(loop)
            self.current_loop = loop
            if self.current_func.start_line < node.lineno <= self.current_func.end_line:
                loop.set_within_func(self.current_func)
            if isinstance(node.target, ast.Name):
                self.variables.add(node.target.id)
            self.generic_visit(node)
        else:
            loop = self.loops[-1]
            loop.set_nested()
            if isinstance(node.target, ast.Name):
                self.builtin_funcs.add(node.target.id)

    def visit_Assign(self, node):
        if not self.current_loop.start_line < node.lineno <= self.current_loop.end_line:
            self.variables.update([child_node.id for child_node in node.targets if isinstance(child_node, ast.Name)])
        self.generic_visit(node)

    def visit_Call(self, node):
        # Modify node
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
                logging.debug(f"Visiting: {func_name}")
                self.visit_FunctionDef(function.node, False)
        self.generic_visit(node)

    def find_nested_functions(self, node):
        for child_node in ast.walk(node):
            if isinstance(child_node, ast.Call) and isinstance(child_node.func, ast.Name) \
                    and child_node.func.id in self.functions.keys():
                function = self.functions[child_node.func.id]
                function.set_used()
                self.find_nested_functions(function.node)

    # def find_scope_variables(self, node, global_nodes, parameters):
    #
    #     global_nodes_copy = []
    #     # Iterate over the children of the node
    #
    #     for child in ast.walk(node):
    #         if isinstance(child, ast.Name) and child.id in self.variables and not \
    #                 (child.id in self.functions.keys() or child.id in self.builtin_funcs):
    #             parameters.add(child.id)
    #
    #         if isinstance(child, ast.Global):
    #             global_nodes_copy.append(child)
    #             global_nodes.append(deepcopy(child))
    #
    #     for child in global_nodes_copy:
    #         child.parent.body.remove(child)

    # def create_partitions(self):
    #     partitions = []
    #     global_nodes = []
    #     parameters = set()  # All variables that need to be sent to the cluster client
    #     variables_dict = "{"  # Maps global vars to their values
    #     variables_assign = []
    #
    #     for loop in self.loops:
    #         if loop.is_nested:
    #             # Find all variable names
    #             loop_node = loop.node
    #             self.find_scope_variables(loop_node, global_nodes, parameters)
    #
    #             print("Parameters", parameters)
    #
    #             # Append variables to dictionary
    #             for var in parameters:
    #                 variables_dict += f"'{var}': {var}, "
    #
    #                 assign_node = string_to_ast_node(f"{var} = {self.names['global_vars']}['{var}']")
    #                 variables_assign.append(assign_node)
    #
    #             variables_dict = variables_dict[:-2] + "}"
    #
    #             assign_node = string_to_ast_node(
    #                 f"""for key, val in {self.names['global_vars']}.items():
    #             exec(key + '=val')""")
    #             await_response_node = string_to_ast_node(
    #                 f"{self.names['global_vars']} = {self.names['mediator_object']}."
    #                 f"{self.names['end_template_func']}()")
    #
    #             # Replace the for
    #             if loop.is_within_func:
    #                 func_node = loop.function.node
    #                 index = get_node_index(func_node, loop_node)
    #                 body = global_nodes + func_node.body
    #             else:
    #                 index = get_node_index(self.module_node, loop_node)
    #                 body = self.module_node.body
    #             body.insert(index + 1, assign_node)
    #             body.insert(index + 1, await_response_node)
    #             func_call = string_to_ast_node(
    #                 f"""{self.names['helper_func']}({variables_dict})""")
    #
    #             loop_copy = deepcopy(loop_node)
    #             loop_node.body = [func_call]
    #
    #             with_node = [string_to_ast_node(
    #                 f"""with open('{self.names['client_file']}', 'w') as file:
    #             file.write(str({variables_dict}))""")]
    #
    #             imports = [j for i in self.imports.values() for j in i]
    #             functions = [function.node for function in self.functions.values() if function.is_used]
    #
    #             # Create the new tree
    #             partition = ast.Module(
    #                 body=imports + functions + variables_assign + loop_copy.body + with_node,
    #                 type_ignores=[]
    #             )
    #             partitions.append(ExecutableTree(partition, len(imports + functions), self.names['global_vars']))
    #
    #             parameters.clear()
    #             variables_dict = "{"
    #             variables_assign = []
    #     return partitions


class ClusterModifier(ast.NodeTransformer):
    def __init__(self, visitor):
        self.visitor = visitor
        self.templates = []
        self.global_nodes = []
        self.variables_assign = []
        self.parameter_dict = None
        self.parameters = set()

    def visit_Name(self, node):
        if node.id in self.visitor.variables and not\
                (node.id in self.visitor.functions.keys() or node.id in self.visitor.builtin_funcs):
            self.parameters.add(node.id)
        self.generic_visit(node)
        return node

    def visit_Global(self, node):
        self.global_nodes.append(node)
        self.generic_visit(node)
        return None

    def create_parameter_dict(self):
        self.parameter_dict = "{"
        for parameter in self.parameters:
            self.parameter_dict += f"'{parameter}': {parameter}, "
            assign_node = string_to_ast_node(f"{parameter} = {self.visitor.names['global_vars']}['{parameter}']")
            self.variables_assign.append(assign_node)

        self.parameter_dict = self.parameter_dict[:-2] + "}"

    def modify_input_file(self, loop):
        await_response = string_to_ast_node(
            f"{self.visitor.names['global_vars']} = {self.visitor.names['mediator_object']}."
            f"{self.visitor.names['end_template_func']}()")

        update_parameters = string_to_ast_node(
            f"""for key, val in {self.visitor.names['global_vars']}.items():
        exec(key + '=val')""")

        if loop.is_within_func:
            func_node = loop.function.node
            body = func_node.body
            body.insert(0, self.global_nodes)
            index = get_node_index(func_node, loop.node)
        else:
            index = get_node_index(self.visitor.module_node, loop.node)
            body = self.visitor.module_node.body
        body.insert(index + 1, update_parameters)
        body.insert(index + 1, await_response)
        helper_func = string_to_ast_node(
            f"""{self.visitor.names['helper_func']}({self.parameter_dict})""")

        loop_copy = deepcopy(loop.node)
        loop.node.body = [helper_func]

        return loop_copy

    def create_template(self, loop_copy):
        open_file_node = [string_to_ast_node(
            f"""with open('{self.visitor.names['client_file']}', 'w') as file:
        file.write(str({self.parameter_dict}))""")]

        imports = [j for i in self.visitor.imports.values() for j in i]
        functions = [function.node for function in self.visitor.functions.values() if function.is_used]

        template = ast.Module(
            body=imports + functions + self.variables_assign + loop_copy.body + open_file_node,
            type_ignores=[]
        )

        self.templates.append(ExecutableTree(template, len(imports + functions), self.visitor.names['global_vars']))

    def provide_response(self):
        for loop in self.visitor.loops:
            if loop.is_nested:
                self.generic_visit(loop.node)
                self.create_parameter_dict()
                loop_copy = self.modify_input_file(loop)
                self.create_template(loop_copy)

                self.parameters.clear()
                self.global_nodes = []
                self.variables_assign = []
                self.parameter_dict = None


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
        self.cluster_partitions = list()
        self.partition_index = 0
        self.clients_queue = queue.Queue()
        self.request_queue = queue.Queue()

        self.sock_to_tree = dict()
        self.responses = list()

        self.operation_finished = False

    def set_partition(self, cluster_trees):
        self.cluster_partitions = cluster_trees
        self.responses = [dict() for _ in range(len(self.cluster_partitions))]

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
                    # ip, port = client_addr
                    # if self.ip == ip:
                    #     request_handler = Thread(target=self.send_tree)
                    #     request_handler.start()
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
                            self.request_queue.put(data)
                            msg_type = "Processing Request"
                        elif op_code == 3:
                            Thread(target=self.outgoing_response, args=(sock,)).start()
                            msg_type = "Change Template Request"
                        elif op_code == 4:
                            self.clients_queue.put(sock)
                            msg_type = "Node Is Available For Processing. Executed Requests Count"
                        elif op_code == 6:
                            org = self.sock_to_tree[sock]
                            new = {k: data[k] for k in org if k in data and not org[k] == data[k]}
                            if new:
                                self.responses[self.partition_index].update(new)
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

    def outgoing_requests(self):
        op_code = 5
        while True:
            time.sleep(0.01)
            if not self.clients_queue.empty():
                sock = self.clients_queue.get()
                if sock in self.write_socks:
                    global_vars = self.request_queue.get()
                    if not global_vars:
                        self.clients_queue.put(sock)
                        break
                    partition = self.cluster_partitions[self.partition_index]
                    partition.set_global_vars(str(global_vars))
                    tree = partition.finalize()
                    self.send_msg(sock, op_code, tree)
                    self.sock_to_tree[sock] = global_vars
        self.operation_finished = True

    def outgoing_response(self, sock):
        while not self.request_queue.empty():
            time.sleep(0.01)
        op_code = 2
        response = self.responses[self.partition_index]
        self.send_msg(sock, op_code, response)
        logging.info(f"[RESPONSE SENT] Variables: {response}")

        if len(self.cluster_partitions) - 1 == self.partition_index:
            self.request_queue.put(None)
        else:
            self.partition_index += 1

    def close_sock(self, sock):
        if sock in self.write_socks:
            self.write_socks.remove(sock)
        self.read_socks.remove(sock)
        sock.close()
        logging.info(f"[CONNECTION CLOSED] {sock} has been closed...")


class ExecutableTree:
    def __init__(self, tree, inject_index, global_variables):
        self.tree = tree
        self.index = inject_index
        self.global_vars_name = global_variables
        self.global_vars = None

    def set_global_vars(self, global_variables):
        self.global_vars = global_variables

    def finalize(self):
        tree = deepcopy(self.tree)
        new_assign_node = f"{self.global_vars_name} = " + str(self.global_vars)
        vars_node = string_to_ast_node(new_assign_node)
        tree.body.insert(self.index, vars_node)

        tmp_file = "Cluster_Partition.py"
        with open(tmp_file, 'w') as f:
            f.write(ast.unparse(tree))

        return ast.parse(ast.unparse(tree))


def string_to_ast_node(string):
    module = ast.parse(string)
    if len(module.body) == 1:
        return module.body[0]
    logging.error("String acceded the allowed amount of ast nodes")
    return None


def get_node_index(tree, node):
    return tree.body.index(node)


def exec_tree(tree, file_name=''):
    exec(compile(tree, file_name, 'exec'))


def print_tree(tree):
    astpretty.pprint(tree)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s:%(message)s', datefmt='%I:%M:%S %p', level=logging.ERROR)

    file = "Testing.py"
    modified_file = "Modified_File.py"
    cluster_file = "Cluster_Partition.py"
    with open(file) as source:
        ast_tree = ast.parse(source.read())

    Visitor = ClusterVisitor()
    Visitor.visit(ast_tree)
    Modifier = ClusterModifier(Visitor)
    Modifier.provide_response()
    cluster_partitions = Modifier.templates

    logging.debug(f"Loops | {str(*Visitor.loops)}")
    logging.debug(f"Functions | {str(*Visitor.functions.values())}")
    logging.debug(f"Variables: {Visitor.variables}")
    logging.debug(f"Templates: {cluster_partitions}")

    if cluster_partitions:
        server = Server()
        server.set_partition(cluster_partitions)
        request_handler = Thread(target=server.outgoing_requests)
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
