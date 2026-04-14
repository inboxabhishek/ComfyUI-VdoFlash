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

    def __init__(self, dry_run=False):
        # PromptExecutor requires the server instance and cache_args
        # Default ram headroom to 4GB if not specified
        self.dry_run = dry_run
        default_cache_args = {"ram": 4.0}
        
        if not self.dry_run:
            if hasattr(PromptServer, "instance"):
                self.executor = PromptExecutor(PromptServer.instance, cache_args=default_cache_args)
            else:
                # Fallback if instance isn't ready yet, though it usually is by node load time
                self.executor = None
        else:
            self.executor = None

    async def run(self, graph):
        if self.dry_run:
            print("[DRY-RUN] Simulating graph execution...")
            # Return a dummy tensor [B, H, W, C]
            import torch
            return torch.zeros((1, 512, 512, 3))

        if self.executor is None:
            if hasattr(PromptServer, "instance"):
                self.executor = PromptExecutor(PromptServer.instance, cache_args={"ram": 4.0})
            else:
                raise Exception("PromptServer instance not found. Executor cannot initialize.")

        # execute_async(self, prompt, prompt_id, extra_data={}, execute_outputs=[])
        prompt_id = str(uuid.uuid4())
        
        # We need to find the output node in the graph for execution_outputs
        output_nodes = [node_id for node_id, node_data in graph.items()] 
        
        # Now we await the async version directly
        await self.executor.execute_async(graph, prompt_id, extra_data={}, execute_outputs=output_nodes)
        
        # After execution, we extract results asynchronously
        result = await self.get_output_from_caches(prompt_id, graph)
        if result is None:
            print(f"⚠️ Executor failed to find VAEDecode output for prompt {prompt_id}. Check if the sub-graph crashed.")
        return result

    async def get_output_from_caches(self, prompt_id, graph):
        # We need to find the node that was the 'final' output (VAEDecode)
        # Note: self.executor.caches.outputs is a CacheSet
        for node_id, node_data in graph.items():
            if node_data.get("class_type") == "VAEDecode":
                # Use the async get method
                cached = await self.executor.caches.outputs.get(node_id)
                if cached and cached.outputs:
                    return cached.outputs[0] # The image tensor
        
        return None
