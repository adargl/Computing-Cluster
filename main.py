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


class ClusterModifier(ast.NodeVisitor):
    class Loop:
        def __init__(self, node):
            self.node = node
            self.is_within_func = False
            self.func_node = None
            self.is_nested = False

        def __str__(self):
            return f"Within_func: {self.is_within_func}, Nested: {self.is_nested} |"

        def set_within_func(self, func_node):
            self.is_within_func = True
            self.func_node = func_node

        def set_nested(self):
            self.is_nested = True

    def __init__(self):
        self.module_node = None
        self.names = {"global_vars": "global_variables", "mediator_class": "ClusterMediator",
                      "mediator_object": "mediator", "helper_func": "send_to_cluster", "request_func": "new_request",
                      "end_partition_func": "partition_finished", "client_file": "shared.txt"}

        self.helper_func_node = None
        self.current_func = None
        self.func_start_line = 0
        self.func_end_line = 0
        self.current_loop = None
        self.loop_start_line = 0
        self.loop_end_line = 0

        self.imports = {"import": [], "from": []}
        self.loops = list()
        self.functions = dict()  # {"function name": (func_node, True if used False otherwise)}
        self.variables = set()  # {"variable name"}
        self.builtin_funcs = set()  # {"function name"}

    def function_name(func):
        def wrapper(*args, **kwargs):
            print("Running:", func.__name__)
            func(*args, **kwargs)

        return wrapper

    @function_name
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
        {self.names['mediator_object']}.{self.names['request_func']}({self.names['global_vars']})""")

        node.body.insert(count, self.helper_func_node)
        while isinstance(node.body[count], ast.FunctionDef):
            count += 1
        new_obj_node = string_to_ast_node(f"{self.names['mediator_object']} = {self.names['mediator_class']}()")
        node.body.insert(count, new_obj_node)

        self.generic_visit(node)

    @function_name
    def visit_Import(self, node):
        self.imports["import"].append(node)

    @function_name
    def visit_ImportFrom(self, node):
        self.imports["from"].append(node)

    @function_name
    def visit_FunctionDef(self, node, initial=True):
        # Update the current function with every new definition
        if initial:
            self.current_func = node
            self.func_start_line = node.lineno
            self.func_end_line = node.end_lineno
            # Add to tree functions
            self.functions[node.name] = node, False
        else:
            self.generic_visit(node)

    @function_name
    def visit_For(self, node):
        if not self.loop_start_line < node.lineno <= self.loop_end_line:
            self.current_loop = node
            self.loop_start_line = node.lineno
            self.loop_end_line = node.end_lineno
            loop = self.Loop(node)
            self.loops.append(loop)
            if self.func_start_line < node.lineno <= self.func_end_line:
                loop.set_within_func(self.current_func)
            if isinstance(node.target, ast.Name):
                self.variables.add(node.target.id)
            self.generic_visit(node)
        else:
            loop = self.loops[-1]
            loop.set_nested()
            if isinstance(node.target, ast.Name):
                self.builtin_funcs.add(node.target.id)

    @function_name
    def visit_Assign(self, node):
        if not self.loop_start_line < node.lineno <= self.loop_end_line:
            self.variables.update([child_node.id for child_node in node.targets if isinstance(child_node, ast.Name)])
        self.generic_visit(node)

    @function_name
    def visit_Call(self, node):
        # Modify node
        if isinstance(node.func, ast.Name) and node.func.id in self.functions.keys():
            func_name = node.func.id
            func_node, is_used = self.functions[func_name]
            self.functions[func_name] = func_node, True
            if self.loop_start_line <= node.lineno <= self.loop_end_line:
                if node.func.id in self.functions.keys():
                    loop = self.loops[-1]
                    loop.set_nested()
                    self.find_nested_functions(func_node)
                else:
                    self.builtin_funcs.add(node.func.id)
            else:
                print("Visited:", func_name)
                self.visit_FunctionDef(func_node, False)
        self.generic_visit(node)

    def find_nested_functions(self, node):
        for child_node in ast.walk(node):
            if isinstance(child_node, ast.Call) and isinstance(child_node.func, ast.Name) \
                    and child_node.func.id in self.functions.keys():
                func_node, is_used = self.functions[child_node.func.id]
                self.functions[child_node.func.id] = func_node, True
                self.find_nested_functions(func_node)

    def create_partitions(self):
        partitions = []
        global_variables = set()  # All variables that need to be sent to the cluster client
        variables_dict = "{"  # Maps global vars to their values
        variables_assign = []
        for loop in self.loops:
            if loop.is_nested:
                # Find all variable names
                loop_node = loop.node
                for name in ast.walk(loop_node):
                    if isinstance(name, ast.Name) and name.id in self.variables and not \
                            (name.id in self.functions.keys() or name.id in self.builtin_funcs):
                        global_variables.add(name.id)
                # Append variables to dictionary
                for var in global_variables:
                    variables_dict += f"'{var}': {var}, "

                    assign_node = string_to_ast_node(f"{var} = {self.names['global_vars']}['{var}']")
                    variables_assign.append(assign_node)

                variables_dict = variables_dict[:-2] + "}"

                assign_node = string_to_ast_node(
                    f"""for key, val in {self.names['global_vars']}.items():
                exec(key + '=val')""")
                await_response_node = string_to_ast_node(
                    f"{self.names['global_vars']} = {self.names['mediator_object']}."
                    f"{self.names['end_partition_func']}()")

                # Replace the for
                if loop.is_within_func:
                    index = get_node_index(loop.func_node, loop_node)
                    body = loop.func_node.body
                else:
                    index = get_node_index(self.module_node, loop_node)
                    body = self.module_node.body
                body.insert(index + 1, assign_node)
                body.insert(index + 1, await_response_node)
                func_call = string_to_ast_node(
                    f"""{self.names['helper_func']}({variables_dict})""")

                loop_copy = deepcopy(loop_node)
                loop_node.body = [func_call]

                with_node = [string_to_ast_node(
                    f"""with open('{self.names['client_file']}', 'w') as file:
                file.write(str({variables_dict}))""")]

                imports_and_functions = [j for i in self.imports.values() for j in i] + \
                                        [func_node for func_node, is_used in self.functions.values() if is_used]

                # Create the new tree
                partition = ast.Module(
                    body=imports_and_functions + variables_assign + loop_copy.body + with_node,
                    type_ignores=[]
                )
                partitions.append(ExecutableTree(partition, len(imports_and_functions), self.names['global_vars']))

                global_variables.clear()
                variables_dict = "{"
                variables_assign = []
        return partitions


class Server:
    # number  |  operation code
    #  0      |  reserved
    #  1      |  cluster request
    #  2      |  cluster response
    #  3      |  end partition
    #  4      |  notify readiness
    #  5      |  exec tree request
    #  6      |  exec tree response

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
                    print(f"[CONNECTION ESTABLISHED] connection has been established from {client_addr}...")
                    print(f"[ACTIVE CONNECTIONS] {len(self.read_socks)} active connections...")
                    self.read_socks.append(connection)
                else:
                    packet = self.recv_msg(sock)
                    if packet:
                        if sock not in self.write_socks:
                            self.write_socks.append(sock)
                        data, op_code = packet
                        if op_code == 1:
                            self.request_queue.put(data)
                        elif op_code == 3:
                            Thread(target=self.outgoing_response, args=(sock,)).start()
                        elif op_code == 4:
                            self.clients_queue.put(sock)
                        elif op_code == 6:
                            org = self.sock_to_tree[sock]
                            new = {k: data[k] for k in org if k in data and not org[k] == data[k]}
                            if new:
                                self.responses[self.partition_index].update(new)
                        print(f"[DATA RECEIVED] client: {data}")
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
        print(f"[RESPONSE SENT] data: {response}")

        if len(self.cluster_partitions) - 1 == self.partition_index:
            self.request_queue.put(None)
        else:
            self.partition_index += 1

    def close_sock(self, sock):
        if sock in self.write_socks:
            self.write_socks.remove(sock)
        self.read_socks.remove(sock)
        sock.close()
        print(f"[CONNECTION CLOSED] {sock} has been closed...")


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
    print("String acceded the allowed amount of ast nodes")
    return None


def get_node_index(tree, node):
    return tree.body.index(node)


def exec_tree(tree, file_name=''):
    exec(compile(tree, file_name, 'exec'))


def print_tree(tree):
    astpretty.pprint(tree)


if __name__ == '__main__':
    file = "Testing.py"
    modified_file = "Modified_File.py"
    cluster_file = "Cluster_Partition.py"
    with open(file) as source:
        ast_tree = ast.parse(source.read())

    Modifier = ClusterModifier()
    Modifier.visit(ast_tree)
    cluster_partitions = Modifier.create_partitions()
    print("Loops:", *Modifier.loops)
    print("Variables:", Modifier.variables)
    print("Functions:", Modifier.functions)

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
        # print(f"\n--- File {file} is breakable ---")
    else:
        print(f"\n--- File {file} is unbreakable ---")
