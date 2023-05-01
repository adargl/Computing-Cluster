import ast
import astpretty
import pickle
import queue
import select
import socket
import threading
import builtins
from enum import Enum
from copy import deepcopy
from time import time, perf_counter
from struct import pack, unpack
from itertools import cycle, count
from io import StringIO
from contextlib import redirect_stdout

import logging
from Logger import CustomFormatter


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
            self.parameters = None
            self.function = None
            self.is_within_func = False
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
        def __init__(self, node, container):
            super().__init__(node, container)
            if node is not None:
                self.name = node.name
                self.args = set(arg.arg for arg in node.args.args)
                self.is_used = False

        def __str__(self):
            return f"Name: {self.name}, Used: {self.is_used}"

        @classmethod
        def empty_instance(cls):
            return cls(None, None)

        def set_used(self):
            self.is_used = True

    class Thread(Node):
        def __init__(self, node, container, parameters, func_name, name=None):
            super().__init__(node, container)
            self.name = name
            self.parameters = parameters
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
                      "template_change": "template_change", "await": "get_results", "change": "change_template"}
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

    def is_builtin(self, node):
        if not isinstance(node, (ast.Call, ast.Name)):
            return False
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                return False
            node = node.func
            if isinstance(node, ast.Attribute):
                if not isinstance(node.value, ast.Name):
                    return False
                node = node.value
        return node.id not in self.functions.keys() and (node.id in self.builtin_funcs or node.id in dir(builtins))

    def is_external_function(self, node):
        if not isinstance(node, ast.Call):
            return False
        return not (isinstance(node.func, ast.Name) and node.func.id in self.functions.keys())

    def generic_visit(self, node):
        if has_body(node):
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
            function = self.Function(node, self.current_container)
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
        if not self.is_external_function(node):
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
                self.builtin_funcs.add(name.attr)
            elif isinstance(name, ast.Name):
                self.builtin_funcs.add(name.id)
        self.generic_visit(node)

    def visit_loop(self, node, loop_type):
        if not self.current_loop.lineno < node.lineno <= self.current_loop.end_lineno:
            loop = self.Loop(node, self.current_container, loop_type)
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
                    and not self.is_builtin(child_node):
                function = self.functions[child_node.func.id]
                function.set_used()
                self.find_nested_functions(function.node)


class ClusterModifier(ast.NodeTransformer):
    class KeywordTypes(Enum):
        GLOBAL = ast.Global
        BREAK = ast.Break
        CONTINUE = ast.Continue
        PASS = ast.Pass
        RETURN = ast.Return

    class Keyword:
        def __init__(self, node, container, keyword_type):
            self.keyword_type = keyword_type
            self.node = node
            self.container = container

    def __init__(self, visitor):
        self.visitor = visitor
        self.names = visitor.names
        self.imports = [f for i in visitor.imports.values() for f in i]
        self.modified_functions = [function.node for function in visitor.functions.values() if function.is_used]
        self.original_functions = [deepcopy(function) for function in self.modified_functions]

        self.templates = dict()
        self.keywords = list()
        self.global_nodes = list()
        self.current_params = set()
        self.all_params = set()

        self.templates_count = 0
        self.max_concurrent_loops = 4
        self.current_container = None
        self.instructions = None

        # Create new nodes that do not depend on variables values
        self.assign_results_node = str_to_ast_node(
            f"exec('\\n'.join([f'{{key}} = {{value}}' for key, value in {self.names['parameters']}.items()]))"
        )
        self.assign_params_node = str_to_ast_node(
            f"globals().update({self.names['parameters']})"
        )
        self.change_template_node = str_to_ast_node(f"{self.names['mediator_object']}.{self.names['change']}()")

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
        for arg in thread.parameters:
            func_str += f"{arg}, "
        func_str = func_str[:-2] + ")"
        expr_node = str_to_ast_node(func_str)
        return expr_node

    @staticmethod
    def loop_to_instructions(loop_copy):
        yield loop_copy.node.body

    @property
    def param_dict(self):
        if not self.current_params:
            return "dict()"
        param_dict = "{"
        for parameter in self.current_params:
            param_dict += f"'{parameter}': {parameter}, "
        param_dict = param_dict[:-2] + "}"

        return param_dict

    @property
    def get_results(self):
        return str_to_ast_node(
            f"{self.names['parameters']} = {self.names['mediator_object']}.{self.names['await']}"
            f"({self.templates_count})"
        )

    @property
    def processing_request(self):
        return str_to_ast_node(
            f"{self.names['mediator_object']}.{self.names['processing_request']}"
            f"({self.templates_count}, {self.param_dict})"
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

        # Find out which parameters should be passed into the processing request
        self.generic_visit(loop.node)
        if isinstance(loop.container, ast.FunctionDef):
            function_name = loop.container.name
            function = self.visitor.functions[function_name]
            self.current_params.update(function.args)

        for node in loop.container.body:
            if has_body(node):
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and not self.visitor.is_builtin(child):
                    self.current_params.add(child.id)
        shared_params = self.all_params.intersection(self.current_params)
        self.current_params = shared_params

        loop.freeze()
        self.generic_visit(loop.snapshot.node)
        self.instructions = self.loop_to_instructions(loop.snapshot)
        if loop.loop_type == ObjectType.FOR:
            if isinstance(loop.node.target, ast.Name):
                self.current_params.add(loop.node.target.id)

        for keyword in self.keywords:
            if keyword.keyword_type == self.KeywordTypes.RETURN:
                if keyword.container is loop.node:
                    loop.node.body.append(keyword.node)
                else:
                    loop.node.body.append(keyword.container)

        loop.node.body = [self.processing_request]
        if loop.loop_type == ObjectType.FOR:
            self.generic_visit(loop.container, None,
                               "is_custom_visit", "is_container_node", "add",
                               add_after={loop.node: (self.get_results, self.assign_results_nodes(self.current_params),
                                                      self.change_template_node)})
        else:
            loop.node.body.extend((self.get_results, *self.assign_results_nodes(self.current_params)))
            self.generic_visit(loop.container, None,
                               "is_custom_visit", "is_container_node", "add",
                               add_after={loop.node: self.change_template_node})

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
            self.all_params.update(last_thread.parameters)
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
                self.all_params.update(thread.parameters)

            self.generic_visit(container, thread_group,
                               "is_custom_visit", "is_container_node", "visit_threads")
            self.create_template()

    def custom_visit(self, obj):
        enum_type = ObjectType(type(obj))
        method = 'modify_' + str(enum_type)
        visitor = getattr(self, method)
        visitor(obj)

    def generic_visit(self, node, modified_nodes=None, *flags, **operations):

        def extract_flags_and_operations():
            allowed_flags = ["is_custom_visit", "is_container_node", "visit_threads", "add"]
            allowed_operations = ["remove", "replace", "add_after", "add_before"]
            is_existing = {element: False for element in allowed_flags + allowed_operations}

            # Extract the arguments that were supplied
            for flag in flags:
                if flag not in allowed_flags:
                    raise ValueError(f"Invalid flag type {flag}."
                                     f" Expected one of: {', '.join(allowed_flags)}")
                else:
                    is_existing[flag] = True
            for operation, value in operations.items():
                if operation not in allowed_operations:
                    raise ValueError(f"Invalid operation type {operation}."
                                     f" Expected one of: {', '.join(allowed_operations)}")
                else:
                    is_existing[operation] = True

            return is_existing

        def create_body_for_threads():
            last_join_call = modified_nodes[-1].join_call.node
            operations["add_after"] = last_join_call

            new_body = []
            nonlocal node
            for n in node.body:
                new_node = self.generic_visit(n, modified_nodes, "is_custom_visit", "visit_threads",
                                              **operations)
                if not new_node:
                    continue
                elif new_node is last_join_call:
                    new_body.append(self.get_results)
                    new_body.extend(self.assign_results_nodes(self.all_params))
                else:
                    new_body.append(new_node)

            return new_body

        def create_body_for_additions():
            if conditions["add_after"] and type(operations["add_after"]) is not dict:
                raise ValueError(f"Invalid operation type, 'add_after' operation must be a dict")
            if conditions["add_before"] and type(operations["add_before"]) is not dict:
                raise ValueError(f"Invalid operation type, 'add_before' operation must be a dict")

            new_body = []
            nonlocal node
            for n in node.body:
                new_node = self.generic_visit(n, modified_nodes, "is_custom_visit", **operations)
                if not new_node:
                    continue
                elif conditions["add_before"]:
                    do_addition(new_node, new_body, "add_before")
                new_body.append(new_node)
                if conditions["add_after"]:
                    do_addition(new_node, new_body, "add_after")

            return new_body

        def do_addition(new_node, new_body, addition_type):
            added_nodes = operations[addition_type].get(new_node)
            if added_nodes:
                if is_iterable(added_nodes):
                    for added_node in added_nodes:
                        if is_iterable(added_node):
                            new_body.extend(added_node)
                        else:
                            new_body.append(added_node)
                else:
                    new_body.append(added_nodes)

        # Override the functionality of generic_visit()
        conditions = extract_flags_and_operations()

        if conditions["is_custom_visit"]:
            if conditions["is_container_node"]:
                if conditions["visit_threads"]:
                    new_b = create_body_for_threads()
                elif conditions["add"]:
                    new_b = create_body_for_additions()
                else:
                    new_b = [
                        self.generic_visit(n, modified_nodes, "is_custom_visit", **operations)
                        for n in node.body if n is not None
                    ]
                node.body = new_b
            else:
                if conditions["visit_threads"]:
                    for thread in modified_nodes:
                        if node is thread.node:
                            return None
                        elif node is thread.start_call.node:
                            self.current_params = thread.parameters
                            return self.processing_request
                        elif node is operations["add_after"]:
                            return node
                        elif node is thread.join_call.node:
                            return None
                    return node
                elif is_iterable(modified_nodes) and node in modified_nodes:
                    if conditions["remove"]:
                        return None
                    elif conditions["replace"]:
                        return operations["replace"]

            return node

        # If the visit is not custom use the original generic_visit method
        if has_body(node):
            self.current_container = node
        return super().generic_visit(node)

    def visit_Global(self, node):
        self.global_nodes.append(node)
        self.generic_visit(node)
        return None

    def visit_Return(self, node):
        self.keywords.append(self.Keyword(node, self.current_container, self.KeywordTypes.RETURN))
        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        if not (self.visitor.is_builtin(node) or node.id in self.visitor.functions.keys()):
            if self.current_params and node.id in self.current_params:
                new_node = str_to_ast_node(f"{self.names['parameters']}['{node.id}']")
                ast.copy_location(new_node, node)
                ast.fix_missing_locations(new_node)
                astpretty.pprint(new_node)
                return new_node
            self.all_params.add(node.id)
        self.generic_visit(node)
        return node

    def create_template(self):
        template = ast.Module(
            body=self.imports + self.original_functions + [self.assign_params_node],
            type_ignores=[]
        )
        self.templates[self.templates_count] = TaskMaker(
            template, len(self.imports + self.original_functions) + 1, self.instructions, self.names['parameters']
        )

        self.templates_count += 1

    def create_templates(self):
        for loop in self.visitor.loops:
            if loop.is_nested:
                self.generic_visit(loop.node)
                self.modify_loop(loop)
                self.create_template()
                self.clear_data()

        self.modify_threads()

    def clear_data(self):
        self.current_params.clear()
        self.all_params.clear()
        self.instructions = None


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

    RUN_START_TIME = None

    class Actions(Enum):
        RESERVED = 0
        PROCESSING_REQUEST = 1
        PROCESSING_RESPONSE = 2
        GET_RESULTS_REQUEST = 3
        NODE_AVAILABLE = 4
        SEND_TASK_TO_NODE = 5
        TASK_RESPONSE = 6
        CONNECT_AS_MEDIATOR = 7
        CONNECT_AS_NODE = 8
        CONNECT_AS_USER = 9
        USER_INPUT_FILE = 10
        CHANGE_TEMPLATE = 11
        UNDEFINED_CODE = 12

    class ConnectionStatus(Enum):
        USER = "User"
        NODE = "Node"
        Mediator = "Mediator"
        UNDEFINED = "None"

    class CustomSocket:
        _ids = count()
        _all_templates = queue.Queue()

        def __init__(self, sock):
            self.id = next(self._ids)
            self.task_id = count()
            ip, port = sock.getsockname()
            self.sock = sock
            self.port = port
            self.ip = ip
            self.lock = threading.Lock()
            self.initial_connection = time()
            self.time_stamps = list()
            self.connection_type = ClusterServer.ConnectionStatus.UNDEFINED

            # If connection_type is Node
            self.hash_to_sock = dict()
            self.ongoing_tasks = dict()
            self.executed_tasks = dict()
            self.unhandled_requests = 0
            self.available_threads = 0

            # If connection_type is Mediator
            self.templates = dict()
            self.results = dict()
            self.get_results = False
            self.change_template = False
            self.result_id = 0
            self.task_queue = queue.Queue()

            # If connection_type is User
            self.temp = None

            logger.info(f"[CONNECTION ESTABLISHED] connection has been established from {ip}, {port}")

        def __hash__(self):
            return self.id

        def __eq__(self, other):
            if isinstance(other, self.__class__):
                return self.id == other.id
            return False

        def __enter__(self):
            self.lock.acquire()

            return self.sock

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.lock.release()

        def set_connection_type(self, connection_type):
            def clear_user_data():
                pass

            def clear_node_data():
                self.hash_to_sock.clear()
                self.ongoing_tasks.clear()
                self.executed_tasks.clear()
                self.available_threads = 0

            def clear_mediator_data():
                self.templates.clear()
                self.results.clear()
                self.get_results = False
                self.result_id = 0
                with self.task_queue.mutex:
                    self.task_queue.queue.clear()

            if connection_type == ClusterServer.Actions.CONNECT_AS_MEDIATOR:
                clear_node_data()
                clear_user_data()
                self.templates = self._all_templates.get()
                self.results = {i: {} for i in range(len(self.templates))}
            elif connection_type == ClusterServer.Actions.CONNECT_AS_NODE:
                clear_mediator_data()
                clear_user_data()
            elif connection_type == ClusterServer.Actions.CONNECT_AS_USER:
                clear_mediator_data()
                clear_node_data()
            else:
                logger.error(
                    f"[CONNECTION DENIED] connection must be of type {ClusterServer.ConnectionStatus.NODE.value}, "
                    f"{ClusterServer.ConnectionStatus.USER.value} or {ClusterServer.ConnectionStatus.Mediator.value} "
                    f"when initialized")
                return

            self.connection_type = connection_type

        def time_stamp_action(self, action):
            self.time_stamps.append((time(), action))

        def handle_user_input(self, file_name="Examples/CaesarCipher.py"):
            file = file_name
            modified_file = "Modified.py"
            byproduct_file = "Created.py"
            with open(file) as source:
                ast_tree = ast.parse(source.read())

            visitor = ClusterVisitor()
            visitor.visit(ast_tree)
            modifier = ClusterModifier(visitor)
            modifier.create_templates()
            templates = modifier.templates

            if templates:
                logger.warning(f"File {file} is breakable")
                self.templates = templates
                self._all_templates.put(templates)

                modified_code = ast.unparse(ast_tree)
                with open(modified_file, 'w') as output_file:
                    output_file.write(modified_code)
                modified_code = ast.parse(modified_code)
                exec_tree(modified_code)
                pass
            else:
                logger.warning(f"File {file} is unbreakable")

        def add_request(self, template_id, params):
            self.task_queue.put((template_id, params))

        def get_request(self):
            if self.task_queue.empty():
                return None
            return self.task_queue.get()

        def add_task(self, user_sock, params):
            new_id = next(self.task_id)
            self.ongoing_tasks[new_id] = params
            self.hash_to_sock[new_id] = user_sock
            return new_id

        def get_task(self, template_id, params):
            task = self.templates[template_id]
            task.params = params
            return task.to_executable()

        def get_result(self):
            result_id, current_result = self.result_id, self.results[self.result_id]
            self.get_results = False
            if self.change_template:
                self.change_template_count()
            return result_id, current_result

        def change_template_count(self):
            if self.unhandled_requests == 0:
                self.change_template = False
                self.result_id += 1

        def update_result(self, original, params):
            new = {p: params[p] for p in original if p in params and not original[p] == params[p]}
            if new:
                response = self.results.get(self.result_id)
                if not response:
                    response.update(new)
                else:
                    for key in response.keys():
                        org_value, new_value = original[key], new[key]
                        if not type(org_value) == type(new_value):
                            response[key] = new[key]
                        else:
                            if isinstance(org_value, list):
                                for i in range(len(new_value)):
                                    if org_value[i] != new_value[i]:
                                        response[key][i] = new_value[i]
                                else:
                                    if len(org_value) < len(new_value):
                                        org_value.extend(new_value[i:])
                                    elif len(org_value) > len(new_value):
                                        response[key] = response[key][:-(len(org_value) - len(new_value))]
                            elif isinstance(org_value, dict):
                                key_names = org_value.keys()
                                for k in new_value.keys():
                                    if k in key_names:
                                        if org_value[k] != new_value[k]:
                                            response[key][k] = new_value[k]
                                    else:
                                        response[key][k] = new_value[k]

        def node_failure(self):
            pass

    def __init__(self, port=55555, send_format="utf-8", buffer_size=1024, max_queue=5):
        self.ip = "0.0.0.0"
        self.port = port
        self.addr = (self.ip, self.port)
        self.format = send_format
        self.buffer = buffer_size
        self.max_queue = max_queue
        self.main_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.user_task_queue = queue.Queue()
        self.user_result_queue = queue.Queue()
        self.node_queue = queue.Queue()
        self.all_socks = dict()
        self.read_socks = list()
        self.write_socks = list()

        self.task_condition = threading.Condition()
        self.result_condition = threading.Condition()
        self.lock = threading.Lock()

        self.close_connections = False
        self.RUN_START_TIME = time()
        self.init_connection()

    def init_connection(self):
        self.main_sock.bind(self.addr)
        self.main_sock.listen(self.max_queue)
        self.read_socks = [self.main_sock]
        self.write_socks = []
        connection_handler = threading.Thread(target=self.handle_connection)
        request_handler = threading.Thread(target=self.handle_tasks)
        response_handler = threading.Thread(target=self.handle_results)
        connection_handler.start()
        response_handler.start()
        request_handler.start()

    def handle_connection(self):
        actions = self.Actions
        while not self.close_connections:
            readable, writeable, exceptions = select.select(self.read_socks, self.write_socks, self.read_socks)
            for sock in readable:
                if sock is self.main_sock:
                    connection, client_addr = sock.accept()
                    self.read_socks.append(connection)
                else:
                    with self.lock:
                        custom_sock = self.all_socks.get(sock)
                        if not custom_sock:
                            custom_sock = self.CustomSocket(sock)
                            self.all_socks[sock] = custom_sock
                    packet = self.recv_msg(custom_sock)
                    if not packet:
                        self.close_sock(custom_sock)
                    else:
                        if sock not in self.write_socks:
                            self.write_socks.append(sock)
                        sock = custom_sock
                        action, optional, reserved, data = packet
                        if action == actions.UNDEFINED_CODE:
                            continue
                        if action == actions.PROCESSING_REQUEST:
                            template_id = optional
                            sock.add_request(template_id, data)
                            self.user_task_queue.put(sock)
                            with self.task_condition:
                                self.task_condition.notify()
                        elif action == actions.GET_RESULTS_REQUEST:
                            with sock:
                                sock.get_results = True
                        elif action == actions.CHANGE_TEMPLATE:
                            with sock:
                                sock.change_template = True
                        elif action == actions.NODE_AVAILABLE:
                            self.node_queue.put(sock)
                            with self.task_condition:
                                self.task_condition.notify()
                        elif action == actions.TASK_RESPONSE:
                            task_id, params = reserved, data
                            with sock:
                                original = sock.ongoing_tasks.pop(task_id)
                                sock.executed_tasks[task_id] = original
                                user_sock = sock.hash_to_sock[task_id]
                            with user_sock:
                                user_sock.update_result(original, params)
                                user_sock.unhandled_requests -= 1
                                if user_sock.get_results and user_sock.task_queue.empty() \
                                        and user_sock.unhandled_requests == 0:
                                    self.user_result_queue.put(user_sock)
                                    with self.result_condition:
                                        self.result_condition.notify()
                        elif action == actions.CONNECT_AS_MEDIATOR or action == actions.CONNECT_AS_NODE \
                                or action == actions.CONNECT_AS_USER:
                            with sock:
                                sock.set_connection_type(action)
                        elif action == actions.USER_INPUT_FILE:
                            with sock:
                                file_name = data
                                handler = threading.Thread(target=sock.handle_user_input, args=(file_name, ))
                                handler.start()

            for sock in exceptions:
                sock = self.all_socks[sock]
                self.close_sock(sock)
        for sock in self.read_socks:
            sock = self.all_socks[sock]
            self.close_sock(sock)

    def send_msg(self, sock, action, msg=None, optional=0, reserved=0):
        pickled_msg = pickle.dumps(msg)
        op_code = action.value
        pickled_msg = pack('>4I', len(pickled_msg), op_code, optional, reserved) + pickled_msg

        with sock as socket_object:
            sock.time_stamp_action(action)
            socket_object.sendall(pickled_msg)

    def recv_msg(self, sock):
        with sock as socket_object:
            raw_header = self.recv_limited_bytes(socket_object, 16)
            if not raw_header:
                return None
            msg_len, op_code, optional, reserved = unpack('>4I', raw_header)
            pickled_msg = self.recv_limited_bytes(socket_object, msg_len)
            msg = pickle.loads(pickled_msg)

            actions = self.Actions
            action = actions.UNDEFINED_CODE
            if 0 < op_code < len(actions) - 1:
                msg_fmt = None
                action = actions(op_code)
                if action == actions.PROCESSING_REQUEST:
                    template_id = optional
                    msg_fmt = f"processing request (id={template_id})"
                elif action == actions.GET_RESULTS_REQUEST:
                    task_group_id = optional
                    logger.info(f"[DATA RECEIVED] get results request (id={task_group_id})")
                elif action == actions.NODE_AVAILABLE:
                    logger.info(f"[NODE AVAILABLE] {sock.ip} executed {len(sock.executed_tasks)} task(s)")
                    logger.info(f"[NODE AVAILABLE] {sock.ip} can have {msg} threads available")
                elif action == actions.TASK_RESPONSE:
                    task_id = optional
                    msg_fmt = f"process result (id={task_id})"
                elif action == actions.CONNECT_AS_MEDIATOR:
                    logger.info(f"[NEW USER CONNECTION] {sock.ip} connected as a mediator")
                elif action == actions.CONNECT_AS_USER:
                    logger.info(f"[NEW USER CONNECTION] {sock.ip} connected as a user")
                elif action == actions.CONNECT_AS_NODE:
                    logger.info(f"[NEW NODE CONNECTION] {sock.ip} connected as a node")
                if msg_fmt:
                    logger.info(f"[DATA RECEIVED] {msg_fmt}: {msg}")

            sock.time_stamp_action(action)

        return action, optional, reserved, msg

    def recv_limited_bytes(self, sock, n):
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def handle_tasks(self):
        while True:
            if self.user_task_queue.empty() or self.node_queue.empty():
                with self.task_condition:
                    self.task_condition.wait()
            else:
                node_sock = self.node_queue.get()
                with node_sock as sock_object:
                    if sock_object not in self.write_socks:
                        continue
                user_sock = self.user_task_queue.get()
                with user_sock:
                    data = user_sock.get_request()
                if not data:
                    self.node_queue.put(node_sock)
                    logger.error(f"[ERROR ENCOUNTERED] expected a task on {user_sock.ip} but none were found")
                    continue
                template_id, params = data
                task = user_sock.get_task(template_id, params)
                task_id = node_sock.add_task(user_sock, params)
                self.send_msg(node_sock, self.Actions.SEND_TASK_TO_NODE, task, template_id, task_id)
                with user_sock:
                    user_sock.unhandled_requests += 1

    def handle_results(self):
        while True:
            with self.result_condition:
                self.result_condition.wait()
            if self.user_result_queue.empty():
                continue
            user_sock = self.user_result_queue.get()
            with user_sock:
                result_id, result = user_sock.get_result()
            self.send_msg(user_sock, self.Actions.PROCESSING_RESPONSE, result)
            logger.info(f"[RESPONSE SENT] response parameters (id={result_id}): {result}")

    def close_sock(self, sock):
        with self.lock:
            if sock in self.all_socks:
                del self.all_socks[sock]
        with sock as socket_object:
            if socket_object in self.write_socks:
                self.write_socks.remove(socket_object)
            self.read_socks.remove(socket_object)
            socket_object.close()
            sock.node_failure()
        logger.info(f"[CONNECTION CLOSED] connection from {sock.ip}, {sock.port} had been closed")


class TaskMaker:
    def __init__(self, tree, inject_index, instructions, params_name):
        self.tree = tree
        self.index = inject_index
        self.instructions = cycle(instructions)
        self.params_name = params_name
        self.params = None

    @property
    def instruction_block(self):
        return next(self.instructions)

    @staticmethod
    def extend_in_index(lst, iterable, index):
        if is_iterable(iterable):
            for count, item in enumerate(iterable):
                lst.insert(index + count, item)
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


def has_body(ast_node):
    return hasattr(ast_node, 'body') and is_iterable(ast_node.body)


def exec_tree(tree, file_name=''):
    start_time = perf_counter()
    std_out = StringIO()
    with redirect_stdout(std_out):
        exec(compile(tree, file_name, 'exec'), {'builtins': globals()['__builtins__']})
    finish_time = perf_counter()

    # Remove the last char from the output since it is always '\n'
    output = std_out.getvalue()[:-1]
    logger.critical(f"[EXECUTION FINISHED] Program output:\n{'-' * 10}Final{'-' * 10}\n" + output
                    + f"\n{'-' * 12}+{'-' * 12}\nfinished in {finish_time - start_time} second(s)")


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

    server = ClusterServer()
