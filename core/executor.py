import sys
import os
import uuid

# Robust import for PromptExecutor
try:
    from execution import PromptExecutor
except ImportError:
    try:
        import execution
        PromptExecutor = execution.PromptExecutor
    except ImportError:
        # Check if we are in a subfolder and need to reach up
        import comfy.execution
        PromptExecutor = comfy.execution.PromptExecutor

# Robust import for PromptServer
try:
    from server import PromptServer
except ImportError:
    import server
    PromptServer = server.PromptServer

class GraphExecutor:

    def __init__(self):
        # PromptExecutor requires the server instance
        if hasattr(PromptServer, "instance"):
            self.executor = PromptExecutor(PromptServer.instance)
        else:
            # Fallback if instance isn't ready yet, though it usually is by node load time
            self.executor = None

    def run(self, graph):
        if self.executor is None:
            if hasattr(PromptServer, "instance"):
                self.executor = PromptExecutor(PromptServer.instance)
            else:
                raise Exception("PromptServer instance not found. Executor cannot initialize.")

        # execute(self, prompt, prompt_id, extra_data={}, execute_outputs=[])
        prompt_id = str(uuid.uuid4())
        
        # We need to find the output node in the graph for execution_outputs
        # For simplicity in MVP, we might just try to execute the whole graph
        # or find nodes that have no consumers.
        output_nodes = [node_id for node_id, node_data in graph.items()] # Execute all for now
        
        # PromptExecutor.execute is a wrapper around asyncio.run(execute_async)
        # Since we are likely in a worker thread, this should be fine.
        self.executor.execute(graph, prompt_id, extra_data={}, execute_outputs=output_nodes)
        
        # After execution, we need to extract the results.
        # This is the tricky part - result extraction from PromptExecutor is complex.
        # For the MVP image graph, we want the VAEDecode output.
        # We'll need to look into self.executor.caches.outputs
        
        # For now, let's return a placeholder or try to find the actual tensor
        return self.get_output_from_caches(prompt_id, graph)

    def get_output_from_caches(self, prompt_id, graph):
        # We need to find the node that was the 'final' output (VAEDecode)
        # and get its cached result.
        # Note: self.executor.caches.outputs is a CacheSet
        for node_id, node_data in graph.items():
            if node_data.get("class_type") == "VAEDecode":
                cached = self.executor.caches.outputs.get_sync(node_id)
                if cached and cached.outputs:
                    return cached.outputs[0] # The image tensor
        
        return None
