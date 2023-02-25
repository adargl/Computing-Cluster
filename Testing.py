import ast
import astpretty


class Visitor(ast.NodeTransformer):
    def generic_visit(self, node, modify=False, is_container=False, modified_node=None, remove_node=False,
                      replacement_node=None):
        if modify:
            if node is modified_node:
                if remove_node:
                    return None
                elif replacement_node:
                    return replacement_node
            else:
                if is_container:
                    node.body = [
                        self.generic_visit(n, modify, False, modified_node, remove_node, replacement_node)
                        for n in node.body if n is not None
                    ]
            return node
        return super().generic_visit(node)

    def visit_Assign(self, node):
        return node


def string_to_ast_node(string: str):
    module = ast.parse(string)
    if len(module.body) == 1:
        return module.body[0]
    return None


def my_function(*flags):
    allowed_flags = ["is_condition1", "is_condition2"]
    is_condition1 = False
    is_condition2 = False
    variables = locals()

    # Extract the arguments that were supplied
    for flag in flags:
        if flag not in allowed_flags:
            raise ValueError(f"Invalid flag type. Expected one of: {allowed_flags}")
        else:
            variables[flag] = True

    print(is_condition1)
    print(is_condition2)


if __name__ == '__main__':
    my_function("is_condition1", "is_condition2")

# def modify_loop(self, loop):
#         loop.make_node_copy()
#
#         if loop.is_within_func:
#             func_node = loop.function.node
#             loop.function.replace_node_with_copy()
#             body = func_node.body
#             body.insert(0, self.global_nodes)
#             body.insert(0, self.global_mediator_node)
#             insertion_index = get_node_locations(func_node, loop.node)
#         else:
#             body = self.visitor.module_node.body
#             insertion_index = get_node_locations(self.visitor.module_node, loop.node)
#         body.insert(insertion_index + 1, self.change_template_node)
#
#         if loop.loop_type == self.visitor.Type.FOR:
#             body.insert(insertion_index + 1, self.update_params_node)
#             loop.node.body = [self.execute_on_cluster_node]
#         elif loop.loop_type == self.visitor.Type.WHILE:
#             loop.node.body = [self.params_assign, self.execute_on_cluster_node, self.await_response_node,
#                               self.update_params_node]
#
#         return loop

# def modify_thread(self, thread):
#         tree = self.visitor.module_node
#
#         # Replace the start() and join() calls on the thread and insert new nodes
#         if thread.start_call:
#             locations = dict()
#             if thread.join_call:
#                 get_node_locations(tree, [thread.start_call, thread.join_call], locations)
#                 index, body = locations[thread.start_call]
#                 body[index] = self.update_params_node
#                 index, body = locations[thread.join_call]
#                 body.insert(index, self.change_template_node)
#             else:
#                 get_node_locations(tree, [thread.start_call], locations)
#                 index, body = locations[thread.start_call]
#                 body[index] = self.update_params_node
#                 body.insert(index, self.change_template_node)
#
#         for arg_name in thread.args:
#             self.parameters.add(arg_name)2345