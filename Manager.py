import ast
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
            self.child_stack = list()
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
            self.loop = None
            self.function = None
            self.is_within_loop = False
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

        def set_within_loop(self, loop):
            self.is_within_loop = True
            self.loop = loop

        def set_nested(self):
            self.is_nested = True

    class Function(Node):
        def __init__(self, node, container):
            super().__init__(node, container)
            if node is not None:
                self.name = node.name
                self.args = set(arg.arg for arg in node.args.args)
                self.is_used = False
                self.was_visited = False

        def __str__(self):
            return f"Name: {self.name}, Used: {self.is_used}"

        @classmethod
        def empty_instance(cls):
            return cls(None, None)

        def set_used(self):
            self.is_used = True

        def set_visited(self):
            self.was_visited = True

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

    def __init__(self, condition_name, parameters_name="params", user_sock_id="user_sock_id"):
        self.names = {"parameters": parameters_name, "mediator_class": "Mediator", "mediator_object": "cluster",
                      "processing_request": "send_request", "while_processing_request": "send_while_request",
                      "template_change": "template_change", "await": "get_results", "condition": condition_name,
                      "user_sock_id": user_sock_id, "end_connection": "close_connection"}
        self.module_node = None

        self.current_func = self.Function.empty_instance()
        self.current_loop = self.Loop.empty_instance()
        self.hierarchy_stack = list()
        self.container_stack = list()

        self.imports = {"import": [], "from": []}
        self.loops = list()
        self.threads = list()
        self.functions = dict()
        self.parameters = set()
        self.builtin_funcs = set()

    def get_current_container(self, node):
        """Return the container (ast node with a body attribute) that contains the provided node.

        If no container had been visited yet, the function returns an ast.Module object. Otherwise,
        out of the fitting containers, the most recent one is returned.

        """

        current_container = self.module_node
        while self.container_stack and not self.is_within(node, self.container_stack[-1]):
            self.container_stack.pop()
        if self.container_stack:
            current_container = self.container_stack[-1]
        self.container_stack.append(node)
        return current_container

    def update_container_stack(self, node):
        """Update the container stack. Called each time a new container is encountered."""

        while self.container_stack and not self.is_within(node, self.container_stack[-1]):
            self.container_stack.pop()
        self.container_stack.append(node)

    @staticmethod
    def is_within(node, container):
        """Return True if a node is within the line number range of a container or False otherwise."""

        if isinstance(container, ast.Module):
            return True
        return container.lineno < node.lineno <= container.end_lineno

    def is_builtin(self, node):
        """Return True if a node is a python builtin name or False otherwise."""

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
        """Return True if a node is a non-user written function or False otherwise."""

        if not isinstance(node, ast.Call):
            return False
        return not (isinstance(node.func, ast.Name) and node.func.id in self.functions.keys())

    def generic_visit(self, node):
        """Override 'generic_visit' to update the container stack with each container node encountered."""

        if has_body(node):
            self.update_container_stack(node)
        super().generic_visit(node)

    def visit_Module(self, node):
        """Override 'visit_Module' to save a reference to the root node of the ast tree."""

        self.module_node = node
        self.generic_visit(node)

    def visit_Import(self, node):
        """Override 'visit_Import' to save all imported nodes."""

        self.imports["import"].append(node)

    def visit_ImportFrom(self, node):
        """Override 'visit_ImportFrom' to save all imported nodes."""

        self.imports["from"].append(node)

    def visit_FunctionDef(self, node, initial=True):
        """Override 'visit_FunctionDef' to save information about all functions.

        If 'visit_FunctionDef' is called initially, it creates and saves a new Function object. Otherwise,
        if 'visit_FunctionDef' is called a second time, it means that the function was used in the program. In such
        case, the function is visited to collect further information.

        """

        if initial:
            function = self.Function(node, self.get_current_container(node))
            self.current_func = function
            self.functions[node.name] = function
        else:
            function = self.functions[node.name]
            if not function.was_visited:
                self.current_func = function
                self.generic_visit(node)
                self.current_func.set_visited()

    def visit_For(self, node):
        """Override 'visit_For' to call a custom visit."""

        self.visit_loop(node, ObjectType.FOR)

    def visit_While(self, node):
        """Override 'visit_While' to call a custom visit."""

        self.visit_loop(node, ObjectType.WHILE)

    def visit_Assign(self, node):
        """Override 'visit_Assign' to find out weather an assignment node contains the creation of a new
         thread.

         """

        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) \
                and node.value.func.attr == 'Thread':
            name = node.targets[0]
            if isinstance(name, ast.Name):
                self.create_thread(node, name.id)
        else:
            if not self.is_within(node, self.current_loop):
                self.parameters.update(
                    [child_node.id for child_node in node.targets if isinstance(child_node, ast.Name) and
                     child_node.id not in self.builtin_funcs]
                )
            self.generic_visit(node)

    def visit_Expr(self, node):
        """Override 'visit_Expr' to find out weather an expression node contains a start() or join() call on a
        thread.

         """

        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            function = node.value.func
            if isinstance(function.value, ast.Name):
                if function.attr == 'start' or function.attr == 'join':
                    for thread in self.threads:
                        if thread.name == function.value.id:
                            current_container = self.get_current_container(node)
                            thread.set_start_call(node, current_container) \
                                if function.attr == 'start' else thread.set_join_call(node, current_container)
        self.generic_visit(node)

    def visit_Call(self, node):
        """Override 'visit_Call' to examine all call nodes.

        If the call node encountered is user written and contained by a loop, set the loop as nested.

        """

        if not self.is_external_function(node):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if not func_name:
                return
            function = self.functions[func_name]
            function.set_used()

            if self.is_within(node, self.current_loop):
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
        """Add a custom visit functionality to For and While loops.

        If a loop is within another loop, set the outer loop as nested.
        if a loop is within a function, set the loop as a loop within a function.

        """

        if not self.is_within(node, self.current_loop):
            loop = self.Loop(node, self.get_current_container(node), loop_type)
            self.loops.append(loop)
            self.current_loop = loop
        else:
            loop = self.loops[-1]
            loop.set_nested()
            loop = self.Loop(node, self.get_current_container(node), loop_type)
            self.loops.append(loop)
            if isinstance(node.target, ast.Name):
                self.builtin_funcs.add(node.target.id)

        if self.is_within(node, self.current_func):
            loop.set_within_func(self.current_func)
        while self.hierarchy_stack and not self.is_within(node, self.hierarchy_stack[-1]):
            self.hierarchy_stack.pop()
        if self.hierarchy_stack:
            base_loop = self.hierarchy_stack[0]
            base_loop.child_stack.append(loop)
            loop.set_within_loop(base_loop)
        self.hierarchy_stack.append(loop)

        self.generic_visit(node)

    def create_thread(self, assign_node, name):
        """Create a new Thread object."""

        call_node = assign_node.value
        args = func_name = None
        for keyword in call_node.keywords:
            if keyword.arg == "target" and isinstance(keyword.value, ast.Name):
                func_name = keyword.value.id
            elif keyword.arg == "args" and isinstance(keyword.value, ast.Tuple):
                args = [name.id for name in keyword.value.elts if isinstance(name, ast.Name)
                        and name.id not in self.builtin_funcs]

        # Make sure that the thread function call is not recursive
        if not self.is_within(assign_node, self.current_loop) or func_name != self.current_func.name:
            thread = self.Thread(assign_node, self.get_current_container(assign_node), args, func_name, name)
            self.threads.append(thread)
            function = self.functions[func_name]
            self.find_nested_functions(function.node)
            function.set_used()

    def find_nested_functions(self, node):
        """Find all user written function calls within a node, and set them as nested."""

        for child_node in ast.walk(node):
            if isinstance(child_node, ast.Call) and isinstance(child_node.func, ast.Name) \
                    and not self.is_builtin(child_node):
                function = self.functions[child_node.func.id]
                function.set_used()
                self.find_nested_functions(function.node)


class ClusterModifier(ast.NodeTransformer):
    def __init__(self, visitor, max_while_tasks):
        self.visitor = visitor
        self.names = visitor.names
        self.max_while_tasks = max_while_tasks
        self.imports = [f for i in visitor.imports.values() for f in i]
        self.modified_functions = [function.node for function in visitor.functions.values() if function.is_used]
        self.original_functions = [deepcopy(function) for function in self.modified_functions]

        self.templates = dict()
        self.global_nodes = list()
        self.manually_marked_nodes = list()
        self.current_params = set()
        self.all_params = set()

        self.templates_count = 0
        self.instructions = None

        self.visit_params = False

        # Create new nodes that do not depend on variables values
        self.assign_results_node = str_to_ast_node(
            f"exec('\\n'.join([f'{{key}} = {{value}}' for key, value in {self.names['parameters']}.items()]))"
        )
        self.assign_params_node = str_to_ast_node(
            f"globals().update({self.names['parameters']})"
        )
        self.close_connection_node = str_to_ast_node(
            f"{self.names['mediator_object']}.{self.names['end_connection']}()"
        )
        self.while_condition = str_to_ast_node(
            f"{self.names['condition']} = True"
        )

        self.setup_tree()

    @staticmethod
    def exhaust_generator(item, generator):
        """Iterate through a generator object until generator is exhausted or the item is found."""

        for value in generator:
            if value is item:
                break

    @staticmethod
    def lineno_difference(node1, node2):
        """Return line number difference between 2 nodes."""

        return abs(node2.lineno - node1.end_lineno)

    @staticmethod
    def thread_to_instructions(thread):
        """Return the instructions created out of a ClusterVisitor.Thread object.

        Instructions are a collection of ast nodes, that are crucial for the creation of each template object aka
        a TaskMaker.

        """

        func_str = f"{thread.func_name}("
        for arg in thread.parameters:
            func_str += f"{arg}, "
        func_str = func_str[:-2] + ")"
        expr_node = str_to_ast_node(func_str)
        return expr_node

    @staticmethod
    def loop_to_instructions(loop_copy):
        """Yield the instructions created out of a ClusterVisitor.Loop object.

        Instructions are a collection of ast nodes, that are crucial for the creation of each template object aka
        a TaskMaker.
        A loop copy is used, since the original loop could also be modified later.

        """

        yield loop_copy.node.body

    @property
    def param_dict(self):
        """Return an ast.Dict node that maps all of the parameters to their current values."""

        if not self.current_params:
            return "dict()"
        param_dict = "{"
        for parameter in self.current_params:
            param_dict += f"'{parameter}': {parameter}, "
        param_dict = param_dict[:-2] + "}"

        return param_dict

    @property
    def get_results(self):
        """Return an ast.Assign node that will await the cluster execution results."""

        return str_to_ast_node(
            f"{self.names['parameters']} = {self.names['mediator_object']}.{self.names['await']}"
            f"({self.templates_count})"
        )

    @property
    def normal_processing_request(self):
        """Return an ast.Expr node that will send a generic processing request to the server."""

        return str_to_ast_node(
            f"{self.names['mediator_object']}.{self.names['processing_request']}"
            f"({self.templates_count}, {self.param_dict})"
        )

    @property
    def while_processing_request(self):
        """Return an ast.Expr node that will send a while processing request to the server."""

        return str_to_ast_node(
            f"for _ in range({self.max_while_tasks}):"
            f"  {self.names['mediator_object']}.{self.names['while_processing_request']}"
            f"  ({self.templates_count}, {self.param_dict})"
        )

    def assign_results_nodes(self, param_names):
        """Yield a collection of nodes that will assign back the results of a cluster execution."""

        variable_name = "param_names"
        yield str_to_ast_node(f"{variable_name} = {self.names['parameters']}.keys()")

        for name in param_names:
            yield str_to_ast_node(
                f"if '{name}' in {variable_name}:"
                f"  {name} = {self.names['parameters']}['{name}']"
            )

    def params_to_instructions(self):
        """Return the instructions created out of the current params.

        Instructions are a collection of ast nodes, that are crucial for the creation of each template object aka
        a TaskMaker.

        """

        for param in self.current_params:
            assign_node = str_to_ast_node(f"{param} = {self.names['parameters']}['{param}']")
            self.instructions.append(assign_node)

    def setup_tree(self):
        """Perform all of the necessary changes to setup the ast.Module node before being modified."""

        module_node = self.visitor.module_node
        counter = 0
        # Import the mediator class
        new_import_node = str_to_ast_node(
            f"from {self.names['mediator_class']} import {self.names['mediator_class']}"
        )
        module_node.body.insert(counter, new_import_node)

        # Create and inject the mediator object
        while isinstance(module_node.body[counter], (ast.FunctionDef, ast.Import, ast.ImportFrom)):
            counter += 1
        new_obj_node = str_to_ast_node(
            f"{self.names['mediator_object']} = {self.names['mediator_class']}({self.names['user_sock_id']})"
        )
        module_node.body.insert(counter, new_obj_node)

        # Add socket closing
        module_node.body.append(self.close_connection_node)

    def setup_loop(self, loop):
        """Collect additional information about the loop and it's contents.

        This function collects important information, such param names that should be passed into a processing
        request. The params enable a clean execution of each task individually on Node computer.

        """

        for i, node in enumerate(loop.node.body):
            if has_body(node):
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Ellipsis):
                    self.manually_marked_nodes.extend(loop.node.body[i + 1:])
                    loop.node.body = loop.node.body[:i]
                    break

        if not loop.is_within_func:
            self.all_params.update(self.visitor.parameters)
        else:
            for node in loop.function.node.body:
                if has_body(node):
                    continue
                elif isinstance(node, ast.Global):
                    self.all_params.update(node.names)
                else:
                    for child in ast.walk(node):
                        if isinstance(child, ast.Name) and not self.visitor.is_builtin(child):
                            self.all_params.add(child.id)

        if loop.is_within_func:
            function_name = loop.function.name
            function = self.visitor.functions[function_name]
            self.all_params.update(function.args)

        if loop.loop_type == ObjectType.FOR:
            self.generic_visit(loop.node)
        else:
            self.visit_While(loop.node)
            self.current_params.add(self.names['condition'])
            self.all_params.add(self.names['condition'])
        shared_params = self.all_params.intersection(self.current_params)
        self.current_params = shared_params

    def modify_loop(self, loop):
        """Perform necessary modifications on a ClusterVisitor.Loop object."""

        loop.freeze()
        self.generic_visit(loop.snapshot.node)
        self.instructions = self.loop_to_instructions(loop.snapshot)

        if loop.loop_type == ObjectType.FOR:
            if isinstance(loop.node.target, ast.Name):
                self.current_params.add(loop.node.target.id)

        if loop.loop_type == ObjectType.FOR:
            loop.node.body = [self.normal_processing_request]
            self.add_new_nodes(
                loop.container,
                add_after={loop.node: (self.get_results, self.assign_results_nodes(self.current_params))}
            )
            loop.node.body.extend(self.manually_marked_nodes)
        else:
            for_node = self.while_processing_request
            for_node.body.extend(self.manually_marked_nodes)
            loop.node.body = [for_node]
            loop.node.body.extend((self.get_results, *self.assign_results_nodes(self.current_params)))
            self.add_new_nodes(
                loop.container,
                add_before={loop.node: self.while_condition}
            )

    def setup_threads(self):
        """Return a dictionary of all ClusterVisitor.Thread objects sorted by container."""

        sorted_by_container = dict()
        for thread in self.visitor.threads:
            if thread.start_call and thread.join_call:
                if not sorted_by_container.get(thread.join_call.container):
                    sorted_by_container[thread.join_call.container] = [thread]
                else:
                    sorted_by_container[thread.join_call.container].append(thread)
        self.instructions = (self.thread_to_instructions(thread) for _, threads in sorted_by_container.items()
                             for thread in threads)

        return sorted_by_container

    def modify_threads(self, sorted_by_container):
        """Perform necessary modifications on all of the ClusterVisitor.Thread objects."""

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

    def custom_modify(self, obj):
        """Call the modify function that corresponds to the object type."""

        enum_type = ObjectType(type(obj))
        method = 'modify_' + str(enum_type)
        modifier = getattr(self, method)
        modifier(obj)

    def add_new_nodes(self, container_node, add_after=None, add_before=None):
        """Adds several (or only one) ast nodes before or after a specific ast node in a given container."""

        if add_after:
            self.generic_visit(container_node, None,
                               "is_custom_visit", "is_container_node", "add",
                               add_after=add_after)
        elif add_before:
            self.generic_visit(container_node, None,
                               "is_custom_visit", "is_container_node", "add",
                               add_before=add_before)

    def generic_visit(self, node, modified_nodes=None, *flags, **operations):
        """Return the provided node after being modified.

        Override 'generic_visit' to add node replacement, deletion & addition functionalities. Moreover, keep the
        original functionality when needed and provide thread support and handling.

        """

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
                            return self.normal_processing_request
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

        return super().generic_visit(node)

    def visit_Global(self, node):
        """Return None to delete all visited Global nodes.

        Override 'visit_Global' to save all visited ast.Global nodes and remove them from their original location.
        Mostly called on Globals within a loop object.

        """

        self.global_nodes.append(node)
        self.generic_visit(node)
        return None

    def visit_Name(self, node):
        """Return a modified ast.Name object to replace the original.

        Override 'visit_Name' to save all visited names when visit_params flag is on. When flag is off, modifies the
        Name node respectably. Mostly called on Names within a loop object.

        """

        if not (self.visitor.is_builtin(node) or node.id in self.visitor.functions.keys()):
            if self.visit_params:
                self.current_params.add(node.id)
            else:
                if self.current_params and node.id in self.current_params:
                    new_node = ast.Name(id=f"{self.names['parameters']}['{node.id}']", ctx=node.ctx)
                    ast.copy_location(new_node, node)
                    return new_node
        self.generic_visit(node)
        return node

    def visit_While(self, node):
        """Return a modified ast.While object to replace the original.

        Override 'visit_While' to modify and handle it's original test attribute. The test attribute is the condition
        that determines weather the while loop should stop.

        """

        if self.visit_params:
            new_condition = ast.Name(id=self.names['condition'], ctx=ast.Load())
            condition, node.test = node.test, new_condition
            new_assignment = ast.Assign(targets=[new_condition], value=condition)
            ast.fix_missing_locations(new_assignment)
            node.body.append(new_assignment)

        self.generic_visit(node)
        return node

    def create_template(self):
        """Create a new template object aka a TaskMaker based on all of the information that was collected since
        the last template."""

        template = ast.Module(
            body=self.imports + self.original_functions + [self.assign_params_node],
            type_ignores=[]
        )
        self.templates[self.templates_count] = TaskMaker(
            template, len(self.imports + self.original_functions) + 1, self.instructions, self.names['parameters'],
            self.templates_count
        )

        self.templates_count += 1

    def create_templates(self):
        """Handle the creation of all templates.

        Use the information that was provided in the constructor, in order to Manage the creation of all templates.
        After information is collected via modify and setup functions, creates a new template with the updated info.
        """

        def modify_loop(current_loop) -> bool:
            if not current_loop.is_nested or current_loop.is_within_loop:
                return False
            return True

        for loop in self.visitor.loops:
            if modify_loop(loop):
                self.visit_params = True
                self.setup_loop(loop)
                self.visit_params = False
                self.modify_loop(loop)
                self.create_template()
                self.clear_data()

        sorted_by_container = self.setup_threads()
        self.modify_threads(sorted_by_container)

    def clear_data(self):
        """Clear all necessary data. Called after each template creation."""

        self.current_params.clear()
        self.all_params.clear()
        self.manually_marked_nodes.clear()
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
    _condition_name = "condition"
    _parameters_name = "params"
    _user_sock_id = "user_id"

    class Actions(Enum):
        RESERVED = 0
        PROCESSING_REQUEST = 1
        WHILE_PROCESSING_REQUEST = 2
        PROCESSING_RESPONSE = 3
        GET_RESULTS_REQUEST = 4
        NODE_AVAILABLE = 5
        SEND_TASK_TO_NODE = 6
        TASK_RESPONSE = 7
        TASK_FAILED = 8
        CONNECT_AS_MEDIATOR = 9
        CONNECT_AS_NODE = 10
        CONNECT_AS_USER = 11
        USER_INPUT_FILE = 12
        FINAL_RESPONSE = 13
        UNDEFINED_CODE = 14

    class ConnectionStatus(Enum):
        USER = "User"
        NODE = "Node"
        MEDIATOR = "Mediator"
        UNDEFINED = "Undefined"
        DISCONNECTED = "Disconnected"

    class ExecutionStatus(Enum):
        COMPLETED = 0
        REJECTED = 1
        FAILED = 2

        def __str__(self):
            return self.value

    class CustomSocket:
        _ids = count()
        _all_templates = queue.Queue()

        def __init__(self, sock, max_while_tasks):
            self.id = next(self._ids)
            self.task_id = count()
            ip, port = sock.getsockname()
            self.sock = sock
            self.port = port
            self.ip = ip
            self.lock = threading.Lock()
            self.initial_connection = time()
            self.time_stamps = list()
            self.max_while_tasks = max_while_tasks
            self.connection_type = ClusterServer.ConnectionStatus.UNDEFINED

            # If connection_type is Node
            self.task_to_sock = dict()
            self.ongoing_tasks = dict()
            self.executed_tasks = dict()
            self.unhandled_requests = 0
            self.available_threads = 0

            # If connection_type is Mediator
            self.task_id_to_req_id = dict()
            self.templates = dict()
            self.results = dict()
            self.template_id = 0
            self.while_highest_handled_id = 0
            self._socket_is_handling_while = False
            self.get_results = False
            self.id_to_task = None
            self.req_id = count()
            self.task_queue = queue.Queue()
            self.restored_tasks = queue.Queue()

            # If connection_type is User
            self.user_id = 0
            self.mediator_sock = None

            logger.info(f"[CONNECTION ESTABLISHED] connection has been established from {ip}, {port}")

        def __hash__(self):
            """Return a custom identifier for each of sock."""

            return self.id

        def __eq__(self, other):
            """Return True if 2 sock objects or equal or False otherwise."""

            if isinstance(other, self.__class__):
                return self.id == other.id
            return False

        def __enter__(self):
            """Return a non-custom socket object. Called each time a context manager is entered."""

            self.lock.acquire()
            return self.sock

        def __exit__(self, exc_type, exc_val, exc_tb):
            """Called each time a context manager is exited."""

            self.lock.release()

        @property
        def currently_handling_while(self):
            """Return True if sock is currently handling a group of while requests or False otherwise."""

            return self._socket_is_handling_while

        @currently_handling_while.setter
        def currently_handling_while(self, bool_value):
            """Clear all necessary data each time request type changes and is not while."""

            if bool_value:
                self._socket_is_handling_while = True
            else:
                self.req_id = count()
                self.init_id_to_task()
                self.task_queue.queue.clear()
                self.unhandled_requests = 0
                self._socket_is_handling_while = False

        def init_id_to_task(self):
            """Initialize the list of ongoing tasks."""

            self.id_to_task = [None for _ in range(self.max_while_tasks)]

        def map_task_id_to_req_id(self, task_id):
            """Map the task id to a request id.

            When a response from a node is received on the server, it can trace back the original request id.

            """

            new_id = next(self.req_id)
            if new_id == self.max_while_tasks:
                self.init_id_to_task()
                self.task_id_to_req_id.clear()
                self.while_highest_handled_id = 0
                self.req_id = count(1)
                new_id = 0
            self.task_id_to_req_id[task_id] = new_id

        def set_connection_type(self, connection_type, user_id=0):
            """Set the connection type of a sock.

            Connection type is usually one of 3: user, mediator or node.

            """

            def clear_user_data():
                self.user_id = 0
                self.mediator_sock = None

            def clear_node_data():
                self.task_to_sock.clear()
                self.ongoing_tasks.clear()
                self.executed_tasks.clear()
                self.available_threads = 0

            def clear_mediator_data():
                self.templates.clear()
                self.results.clear()
                self.get_results = False
                self.currently_handling_while = False
                self.while_highest_handled_id = 0
                self.req_id = count()
                self.result_id = 0
                with self.task_queue.mutex:
                    self.task_queue.queue.clear()
                with self.restored_tasks.mutex:
                    self.restored_tasks.queue.clear()

            if connection_type == ClusterServer.Actions.CONNECT_AS_MEDIATOR:
                clear_node_data()
                clear_user_data()
                self.templates = self._all_templates.get()
                self.results = {i: {} for i in range(len(self.templates))}
                self.init_id_to_task()
                self.connection_type = ClusterServer.ConnectionStatus.MEDIATOR
            elif connection_type == ClusterServer.Actions.CONNECT_AS_NODE:
                clear_mediator_data()
                clear_user_data()
                self.connection_type = ClusterServer.ConnectionStatus.NODE
            elif connection_type == ClusterServer.Actions.CONNECT_AS_USER:
                clear_mediator_data()
                clear_node_data()
                self.user_id = user_id
                self.connection_type = ClusterServer.ConnectionStatus.USER
            else:
                logger.error(
                    f"[CONNECTION DENIED] connection must be of type {ClusterServer.ConnectionStatus.NODE.value}, "
                    f"{ClusterServer.ConnectionStatus.USER.value} or {ClusterServer.ConnectionStatus.MEDIATOR.value} "
                    f"when initialized")
                return

        def time_stamp_action(self, action, ip=None):
            """Timestamp each action as it happens."""

            self.time_stamps.append((time(), action, ip if ip else self.ip))

        def handle_user_input(self, file):
            """Handle user request.

            This function receives an input file, visits & modifies it and then decides weather to start it's
            execution or reject the file.

            """

            modified_file = "Modified.py"
            byproduct_file = "Created.py"
            ast_tree = ast.parse(file)

            cserver = ClusterServer
            visitor = ClusterVisitor(cserver._condition_name, cserver._parameters_name, cserver._user_sock_id)
            visitor.visit(ast_tree)
            modifier = ClusterModifier(visitor, self.max_while_tasks)
            modifier.create_templates()
            templates = modifier.templates

            if templates:
                logger.warning(f"File is breakable")
                self.templates = templates
                self._all_templates.put(templates)

                modified_code = ast.unparse(ast_tree)
                with open(modified_file, 'w') as output_file:
                    output_file.write(modified_code)
                modified_code = ast.parse(modified_code)
                ClusterServer.exec_tree(self, modified_code)
            else:
                logger.warning(f"File is unbreakable")
                ClusterServer.send_final_output(self, ClusterServer.ExecutionStatus.REJECTED)

        def add_request(self, template_id, params, while_request):
            """Add a new task to the user task queue."""

            if while_request:
                self.currently_handling_while = True
            elif self.currently_handling_while:
                self.currently_handling_while = False
            task = self.get_task(template_id, params)
            self.task_queue.put(task)

        def get_request(self):
            """Return a task. Called before sending a task to a node on the cluster."""

            if not self.restored_tasks.empty():
                return self.restored_tasks.get()
            elif self.task_queue.empty():
                return
            return self.task_queue.get()

        def add_task(self, user_sock, task):
            """Return an id that is used to retrieve a mediator socket.

            When a response from a node is received on the server, it can trace back the original mediator socket it
            came from.

            """

            new_id = next(self.task_id)
            self.ongoing_tasks[new_id] = task
            self.task_to_sock[new_id] = user_sock
            return new_id

        def get_task(self, template_id, params):
            """Return a new task."""

            template = self.templates[template_id]
            task = template.to_executable(params)
            return task

        def get_result(self):
            """Return a result to a group of task.

            Typically used when all tasks of a task group were handled. Can be called several time on each template,
            since each template contains one or more task groups.

            """

            template_id, current_result = self.template_id, self.results[self.template_id]
            self.get_results = False
            return template_id, current_result

        def update_while_result(self, task_id, original, params):
            """Update the current result. Called with each processing response received.

            'update_while_result' keeps track of the order in which requests are sent and addresses them respectably.

            """

            request_id = self.task_id_to_req_id.get(task_id)
            if request_id is None:
                raise RuntimeError
            self.id_to_task[request_id] = original, params

            while request_id < self.max_while_tasks and request_id == self.while_highest_handled_id:
                info = self.id_to_task[request_id]
                if not info:
                    return
                original, params = info
                if not params.get(ClusterServer._condition_name, True):
                    self.get_results = True
                    self.currently_handling_while = False
                    response = self.results.get(self.template_id)
                    response[ClusterServer._condition_name] = False
                    return
                self.update_result(*info)
                request_id += 1
                self.while_highest_handled_id += 1

        def update_result(self, original, params):
            """Update the current result. Called with each processing response received.

            'update_result' doesn't keep track of the order in which requests are sent.

            """

            new = {p: params[p] for p in original if p in params}

            if new:
                response = self.results.get(self.template_id)
                if not response:
                    response.update(params)
                else:
                    for key in response.keys():
                        org_value, new_value = original.get(key), new.get(key)
                        if org_value is None or new_value is None:
                            continue
                        elif not type(org_value) == type(new_value) == type(response[key]):
                            if new_value:
                                response[key] = new_value
                        else:
                            if isinstance(org_value, list):
                                org_len, new_len = len(org_value), len(new_value)
                                for i in range(min(org_len, new_len)):
                                    if org_value[i] != new_value[i]:
                                        response[key][i] = new_value[i]
                                if org_len < new_len:
                                    pass
                                elif org_len > new_len:
                                    response[key] = response[key][:new_len]
                            elif isinstance(org_value, dict):
                                key_names = org_value.keys()
                                for k in new_value.keys():
                                    if k in key_names:
                                        if org_value[k] != new_value[k]:
                                            response[key][k] = new_value[k]
                                    else:
                                        response[key][k] = new_value[k]
                            else:
                                response[key] = new_value

        def restore_tasks(self, server_user_queue):
            """Restore all tasks the were ongoing on a node while it failed."""

            for task_id, task in self.ongoing_tasks.items():
                user_sock = self.task_to_sock[task_id]
                with user_sock:
                    user_sock.restored_tasks.put(task)
                    user_sock.unhandled_requests -= 1
                server_user_queue.put(user_sock)

        def connection_failure(self, server_user_queue):
            """Handle a connection failure."""

            if self.connection_type == ClusterServer.ConnectionStatus.NODE:
                self.restore_tasks(server_user_queue)

    def __init__(self, port=55555, send_format="utf-8", buffer_size=1024, max_queue=5, max_while_tasks=15):
        self.ip = "0.0.0.0"
        self.port = port
        self.addr = (self.ip, self.port)
        self.format = send_format
        self.buffer = buffer_size
        self.max_queue = max_queue
        self.main_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.max_while_tasks = max_while_tasks

        self.user_task_queue = queue.Queue()
        self.user_result_queue = queue.Queue()
        self.node_queue = queue.Queue()
        self.user_id = count()
        self.user_dict = dict()
        self.all_socks = dict()
        self.read_socks = list()
        self.write_socks = list()

        self.task_condition = threading.Condition()
        self.result_condition = threading.Condition()
        self.lock = threading.Lock()

        self.close_connections = False
        self.init_connection()

    def init_connection(self):
        """Initialize all handler threads."""

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
        """Handle all server traffic.

        This function uses select to handle traffic in a non-blocking way. Receiving message protocol is as
        following:

      Operation code | Recipient | Explanation
        0 | Reserved |
        1 | Server   | Mediator requests the services of the cluster [generic type]
        2 | Server   | Mediator requests the services of the cluster [while type]
        3 | Mediator | The server returns a response for a group of requests
        4 | Server   | Mediator asks for an answer to a group of it's requests
        5 | Server   | A node announces that it is ready to accept tasks
        6 | Node     | The server sends a processing request to a task
        7 | Server   | The server receives a processing response back
        8 | Server   | The server receives that a processing response failed
        9 | Server   | Request to connect as a mediator
       10 | Server   | Request to connect as a node
       11 | Server   | Request to connect as a user
       12 | Server   | User sends an input file as a request
       13 | User     | The server sends a final response for a user
       14 |  * * *   | An unrecognized op code was received

        Note: Utilize the Actions Enum class that automatically converts op codes.

        """

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
                            custom_sock = self.CustomSocket(sock, self.max_while_tasks)
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
                            pass
                        elif action == actions.PROCESSING_REQUEST or action == actions.WHILE_PROCESSING_REQUEST:
                            template_id = optional
                            sock.add_request(template_id, data, action == actions.WHILE_PROCESSING_REQUEST)
                            self.user_task_queue.put(sock)
                            with self.task_condition:
                                self.task_condition.notify()
                        elif action == actions.GET_RESULTS_REQUEST:
                            template_id = reserved
                            with sock:
                                sock.get_results = True
                                sock.template_id = template_id
                        elif action == actions.NODE_AVAILABLE:
                            self.node_queue.put(sock)
                            with self.task_condition:
                                self.task_condition.notify()
                        elif action == actions.TASK_RESPONSE or action == actions.TASK_FAILED:
                            task_id, new_params = reserved, data
                            with sock:
                                original = sock.ongoing_tasks.pop(task_id)
                                sock.executed_tasks[task_id] = original
                                user_sock = sock.task_to_sock[task_id]
                            org_params = original.params
                            with user_sock:
                                if not action == actions.TASK_FAILED:
                                    if user_sock.currently_handling_while:
                                        user_sock.update_while_result(task_id, org_params, new_params)
                                    else:
                                        user_sock.update_result(org_params, new_params)

                                if user_sock.unhandled_requests > 0:
                                    user_sock.unhandled_requests -= 1

                                if user_sock.get_results and user_sock.task_queue.empty() \
                                        and user_sock.unhandled_requests == 0:
                                    self.user_result_queue.put(user_sock)
                                    with self.result_condition:
                                        self.result_condition.notify()

                        elif action == actions.CONNECT_AS_MEDIATOR or action == actions.CONNECT_AS_NODE \
                                or action == actions.CONNECT_AS_USER:
                            with sock:
                                if action == actions.CONNECT_AS_USER:
                                    user_id = next(self.user_id)
                                    sock.set_connection_type(action, user_id)
                                    self.user_dict[user_id] = sock
                                else:
                                    if action == actions.CONNECT_AS_MEDIATOR:
                                        user_id = data
                                        user_sock = self.user_dict[user_id]
                                        user_sock.mediator_sock = sock
                                    sock.set_connection_type(action)

                        elif action == actions.USER_INPUT_FILE:
                            file = data
                            with sock:
                                handler = threading.Thread(target=sock.handle_user_input,
                                                           args=(file,))
                            handler.start()

            for sock in exceptions:
                sock = self.all_socks[sock]
                self.close_sock(sock)
        for sock in self.read_socks:
            sock = self.all_socks[sock]
            self.close_sock(sock)

    def send_msg(self, sock, action, msg=None, optional=0, reserved=0):
        """Main message sending protocol over the sockets.

        Each message is sent in 5 fields: | length | op code | optional | reserved | message |
        'send_msg' utilizes the serializing functionalities of struct.pack and pickle.dumps.

        """

        pickled_msg = pickle.dumps(msg)
        op_code = action.value
        pickled_msg = pack('>4I', len(pickled_msg), op_code, optional, reserved) + pickled_msg

        with sock as socket_object:
            sock.time_stamp_action(action)
            socket_object.sendall(pickled_msg)

    def recv_msg(self, sock):
        """Main message receiving protocol over the sockets.

        Each message is received in 5 fields: | length | op code | optional | reserved | message |
        'recv_msg' utilizes the deserializing functionalities of struct.unpack and pickle.loads.

        """

        with sock as socket_object:
            raw_header = None
            try:
                raw_header = self.recv_limited_bytes(socket_object, 16)
            except Exception as e:
                self.raise_exception("while receiving a message encountered", e)
            if not raw_header:
                return

            msg_len, op_code, optional, reserved = unpack('>4I', raw_header)
            pickled_msg = self.recv_limited_bytes(socket_object, msg_len)
            msg = pickle.loads(pickled_msg)

        actions = self.Actions
        action = actions.UNDEFINED_CODE
        if 0 < op_code < len(actions) - 1:
            msg_fmt = None
            action = actions(op_code)
            if action == actions.PROCESSING_REQUEST or action == actions.WHILE_PROCESSING_REQUEST:
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
            elif action == actions.TASK_FAILED:
                task_id = optional
                logger.warning(f"[TASK FAILED] failed to execute task (id={task_id})")
            elif action == actions.CONNECT_AS_MEDIATOR:
                logger.info(f"[NEW USER CONNECTION] {sock.ip} connected as a mediator")
            elif action == actions.CONNECT_AS_USER:
                logger.info(f"[NEW USER CONNECTION] {sock.ip} connected as a user")
            elif action == actions.CONNECT_AS_NODE:
                logger.info(f"[NEW NODE CONNECTION] {sock.ip} connected as a node")
            if msg_fmt:
                logger.info(f"[DATA RECEIVED] {msg_fmt}: {msg}")

        with sock:
            sock.time_stamp_action(action)

        return action, optional, reserved, msg

    def recv_limited_bytes(self, sock, n):
        """Return a limited amount of bytes waiting on the socket.

        This is a helper function that allows the reading of limited amounts of bytes.

        """

        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def handle_tasks(self):
        """Handle all task sending."""

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
                if user_sock.connection_type == self.ConnectionStatus.DISCONNECTED:
                    self.node_queue.put(node_sock)
                    continue
                with user_sock:
                    user_sock.unhandled_requests += 1
                    task = user_sock.get_request()
                if not task:
                    with user_sock:
                        user_sock.unhandled_requests -= 1
                    self.node_queue.put(node_sock)
                    logger.warning(f"[TASK NOT FOUND] task on {user_sock.ip} might have gone unhandled")
                    continue
                task_id = node_sock.add_task(user_sock, task)
                user_sock.map_task_id_to_req_id(task_id)
                user_sock.time_stamp_action(self.Actions.SEND_TASK_TO_NODE, node_sock.ip)
                self.send_msg(node_sock, self.Actions.SEND_TASK_TO_NODE, task, task.template_id, task_id)

    def handle_results(self):
        """Handle all response sending."""

        while True:
            with self.result_condition:
                self.result_condition.wait()
            if self.user_result_queue.empty():
                continue
            user_sock = self.user_result_queue.get()
            if user_sock.connection_type == self.ConnectionStatus.DISCONNECTED:
                continue
            with user_sock:
                result_id, result = user_sock.get_result()
            self.send_msg(user_sock, self.Actions.PROCESSING_RESPONSE, result)
            logger.info(f"[RESPONSE SENT] response parameters (id={result_id}): {result}")

    def close_sock(self, sock):
        """Close a socket. Called on a socket that was either disconnected or encountered an exception."""

        if sock.connection_type == self.ConnectionStatus.DISCONNECTED:
            return

        with self.lock:
            if sock in self.all_socks:
                del self.all_socks[sock]
        with sock as socket_object:
            if socket_object in self.write_socks:
                self.write_socks.remove(socket_object)
            self.read_socks.remove(socket_object)
            socket_object.close()
            sock.connection_failure(self.user_task_queue)
            sock.connection_type = self.ConnectionStatus.DISCONNECTED
        logger.error(f"[CONNECTION CLOSED] connection from {sock.ip}, {sock.port} had been closed")

    @staticmethod
    def raise_exception(additional_text, exception):
        """Raise an error on the logger."""

        logger.error(f"[ERROR ENCOUNTERED] {additional_text}: {exception}")

    @classmethod
    def exec_tree(cls, sock, tree, file_name=''):
        """Execute the user ast tree after it was modified.

        A separate thread on the server that handles the mediator cluster execution, measures the rtt sum and provides
        the user with a response after being finished (or failing).

        """

        start_time = perf_counter()
        std_out = StringIO()
        with redirect_stdout(std_out):
            try:
                exec(compile(tree, file_name, 'exec'), {'builtins': globals()['__builtins__'],
                                                        cls._user_sock_id: sock.user_id})
            except Exception as e:
                ClusterServer.raise_exception("while executing the file encountered", e)
                cls.send_final_output(sock, cls.ExecutionStatus.FAILED)
                return

        finish_time = perf_counter()

        # Remove the last char from the output since it is always '\n'
        output = std_out.getvalue()[:-1]
        user_output = f"{'-' * 10}Final{'-' * 10}\n{output}\n{'-' * 12}+{'-' * 12}"
        runtime = finish_time - start_time
        logger.critical("[EXECUTION FINISHED] Program output:\n" + user_output
                        + f"\nfinished in {runtime} second(s)")

        node_ips = dict()
        for _, action, ip in sock.mediator_sock.time_stamps:
            if action is cls.Actions.SEND_TASK_TO_NODE:
                node_count = node_ips.get(ip, 0)
                node_ips[ip] = node_count + 1
        cls.send_final_output(sock, cls.ExecutionStatus.COMPLETED, output, runtime, node_ips)

    @classmethod
    def send_final_output(cls, sock, status, result=None, runtime=None, communication=None):
        """Secondary message sending protocol that is used exclusively for user responses.

        Each message is sent in 5 fields: | len1 | len2 | len3 | len4 | all 4 messages as one string |
        'send_msg' utilizes the serializing functionalities of struct.pack and pickle.dumps.

        """

        status, runtime = pickle.dumps(status.value), pickle.dumps(runtime)
        result, communication = pickle.dumps(result), pickle.dumps(communication)

        pickled_msg = pack('>4I', len(status), len(runtime), len(result), len(communication)) \
                      + status + runtime + result + communication

        with sock as socket_object:
            socket_object.sendall(pickled_msg)


class TaskMaker:
    def __init__(self, tree, inject_index, instructions, params_name, template_id):
        self.tree = tree
        self.index = inject_index
        self.instructions = cycle(instructions)
        self.params_name = params_name
        self.template_id = template_id

    @property
    def instruction_block(self):
        """Return the next instruction block."""

        return next(self.instructions)

    @staticmethod
    def extend_in_index(lst, iterable, index):
        """Insert an iterable into a list with a given index."""

        if is_iterable(iterable):
            for counter, item in enumerate(iterable):
                lst.insert(index + counter, item)
        else:
            lst.insert(index, iterable)

    def finalize(self):
        """Return a completed ast tree that can be executed on a Node computer."""

        tree = deepcopy(self.tree)
        block = self.instruction_block
        self.extend_in_index(tree.body, block, self.index)

        tmp_file = "Created.py"
        with open(tmp_file, 'w') as f:
            f.write(ast.unparse(tree))

        return ast.parse(ast.unparse(tree))

    def to_executable(self, params):
        """Return an ExecutableTree object which is the representation of a task."""

        return ExecutableTree(self.finalize(), self.template_id, self.params_name, params)


class ExecutableTree:
    def __init__(self, tree, template_id, params_name, params):
        self.tree = tree
        self.template_id = template_id
        self.params_name = params_name
        self.params = params

    def exec_tree(self):
        """Execute the object's ast tree."""

        exec(ast.unparse(self.tree), {self.params_name: self.params})


def str_to_ast_node(string: str):
    """Return a parsed ast out of a given code string."""

    module = ast.parse(string)
    if len(module.body) == 1:
        node = module.body[0]
        return node
    logger.error("String exceeded the allowed amount of ast nodes")
    return None


def is_iterable(obj: object):
    """Return True if an object is an iterable or False otherwise."""

    return hasattr(obj, '__iter__')


def has_body(ast_node):
    """Return True if an ast node has a body (meaning if it is a container node) or False otherwise."""

    return hasattr(ast_node, 'body') and is_iterable(ast_node.body)


if __name__ == '__main__':
    fmt = '%(name)s %(asctime)s.%(msecs)03d %(message)s', '%I:%M:%S'

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logger.level)
    stream_handler.setFormatter(CustomFormatter(*fmt))
    logger.addHandler(stream_handler)

    server = ClusterServer()
