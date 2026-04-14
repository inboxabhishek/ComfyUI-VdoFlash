import torch
from .graph_builder import build_image_graph
from .executor import GraphExecutor
from .vram_manager import get_vram_profile
from .validator import validate_config, resolve_dimensions

class VideoEngine:

    def __init__(self):
        self.executor = GraphExecutor()

    def run(self, cfg, reference_image=None):
        # 1. Capture Environment
        vram = get_vram_profile()

        # 2. Defensive Validation Layer
        cfg = validate_config(cfg, vram)
        width, height = resolve_dimensions(cfg)

        scenes = self.plan(cfg)

        all_frames = []
        prev_frame = None

        print(f"🎬 Init Engine -> Resolution resolved to {width}x{height} | Models: [Img: {cfg['models']['image']}, Vid: {cfg['models']['video']}]")

        for scene in scenes:

            seed = cfg["seed"] + scene["id"]
            
            # Step 1: Base Image Graph Execution
            graph = build_image_graph(scene["prompt"], seed, width, height)
            
            # We assume image executor is stable for MVP, but you could try/except this too
            image = self.executor.run(graph)

            # Step 2: Adaptive Fallback Video Block
            try:
                if cfg["models"]["video"] != "none":
                    video = image.repeat(cfg["fps"], 1, 1, 1) # Video execution stub logic here
                else:
                    print("⚠️ Video explicitly disabled, rendering static block.")
                    video = image.repeat(cfg["fps"], 1, 1, 1)
            except Exception as e:
                print(f"⚠️ VIDEO FAILURE: {e}. Recovering with image loop!")
                video = image.repeat(cfg["fps"], 1, 1, 1)

            # Keep Continuity memory
            prev_frame = video[-1:]
            prev_frame = prev_frame[:, ::2, ::2, :] # Downscale for vram if continuity exists

            all_frames.append(video)

        # Output Stitcher
        return torch.cat(all_frames, dim=0)

    def plan(self, cfg):
        # Inject styling dynamically
        style_prompt = f"style: {cfg['style']['type']}, lighting: {cfg['style']['lighting']}, camera: {cfg['style']['camera']}"
        return [
            {"id": i, "prompt": f"{cfg['topic']} scene {i} - {style_prompt}"}
            for i in range(max(1, cfg["duration"] // 5))
        ]
