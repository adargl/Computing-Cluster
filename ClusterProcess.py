from multiprocessing import Process
from multiprocessing import Value


class CustomProcess(Process):
    def __init__(self, dictionary):
        Process.__init__(self)
        self.vars = dictionary

