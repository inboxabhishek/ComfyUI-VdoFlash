import sys
import os
import torch
import uuid

# Get access to ComfyUI standard nodes
def get_comfy_nodes():
    # Attempt to get the global nodes module, avoiding the local nodes.py shell
    if "nodes" in sys.modules and hasattr(sys.modules["nodes"], "NODE_CLASS_MAPPINGS"):
        return sys.modules["nodes"]
    
    # Force reach to ComfyUI root
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if base_path not in sys.path:
        sys.path.insert(0, base_path) # Insert at 0 to prioritize
        
    import nodes
    return nodes

nodes = get_comfy_nodes()

try:
    from server import PromptServer
except ImportError:
    import server
    PromptServer = server.PromptServer

class GraphExecutor:

    def __init__(self, dry_run=False):
        self.dry_run = dry_run

    async def run(self, graph_or_params):
        """
        Executes the generation chain directly using Python calls instead of the Prompt Queue.
        This prevents deadlocks during nested execution.
        """
        if self.dry_run:
            print("[DRY-RUN] Simulating direct node execution...")
            return torch.zeros((1, 512, 512, 3))

        # We anticipate graph_or_params is the dict from graph_builder for compatibility.
        # But we'll extract the raw values for a direct Python chain.
        try:
            # Extract parameters from the graph dictionary (our 'contract')
            ckpt_name = graph_or_params["1"]["inputs"]["ckpt_name"]
            pos_text = graph_or_params["2"]["inputs"]["text"]
            neg_text = graph_or_params["5"]["inputs"]["text"]
            seed = graph_or_params["3"]["inputs"]["seed"]
            width = graph_or_params["6"]["inputs"]["width"]
            height = graph_or_params["6"]["inputs"]["height"]
            
            print(f"DirectExecution: Starting chain [Model: {ckpt_name}]")

            # 1. Load Checkpoint
            loader_cls = nodes.NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]
            loader = loader_cls()
            model, clip, vae = loader.load_checkpoint(ckpt_name)

            # 2. Encode Prompts
            encoder_cls = nodes.NODE_CLASS_MAPPINGS["CLIPTextEncode"]
            encoder = encoder_cls()
            pos_cond = encoder.encode(clip, pos_text)[0]
            neg_cond = encoder.encode(clip, neg_text)[0]

            # 3. Create Empty Latent
            latent_cls = nodes.NODE_CLASS_MAPPINGS["EmptyLatentImage"]
            latent_gen = latent_cls()
            latent = latent_gen.generate(width, height, batch_size=1)[0]

            # 4. KSampler (The heavy lifting)
            # We use the standard KSampler node logic
            sampler_cls = nodes.NODE_CLASS_MAPPINGS["KSampler"]
            sampler = sampler_cls()
            
            # Direct call to KSampler.sample
            # Note: We manually report progress to the UI here if needed, 
            # but standard nodes usually handle their own progress via PromptServer.instance
            samples = sampler.sample(
                model, seed, 25, 7.0, "euler", "normal", 
                pos_cond, neg_cond, latent, denoise=1.0
            )[0]

            # 5. VAE Decode
            decoder_cls = nodes.NODE_CLASS_MAPPINGS["VAEDecode"]
            decoder = decoder_cls()
            images = decoder.decode(vae, samples)[0]

            return images

        except Exception as e:
            print(f"ERROR: DirectExecution Failure: {e}")
            import traceback
            traceback.print_exc()
            return None
