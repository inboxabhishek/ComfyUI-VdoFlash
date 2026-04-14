from execution import PromptExecutor

class GraphExecutor:

    def __init__(self):
        self.executor = PromptExecutor()

    def run(self, graph):
        result = self.executor.execute(graph)
        return result
