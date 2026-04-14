import time
import torch
import folder_paths
import comfy.model_management as mm

class VdoFlashDirectorNode:

    @classmethod
    def INPUT_TYPES(cls):
        # Dynamically discover local checkpoints
        checkpoints = folder_paths.get_filename_list("checkpoints")
        # For the prototype, we fall back to a reasonable list if empty
        if not checkpoints:
            checkpoints = ["sdxl.safetensors", "realvisxl.safetensors"]

        return {
            "required": {
                # --- Basic Config ---
                "topic_script": ("STRING", {"multiline": True}),
                "duration_seconds": ("INT", {"default": 10}),
                "style_type": (["cinematic", "anime", "photorealistic", "3d_render"],),
                
                # --- Advanced Config ---
                "video_resolution": (["512", "768", "1024", "1280"], {"default": "1024"}),
                "aspect_ratio": (["16:9", "9:16", "1:1"], {"default": "16:9"}),
                "fps": ("INT", {"default": 24}),
                
                # --- Expert Model Selection ---
                "image_model": (checkpoints, {"default": checkpoints[0] if checkpoints else "sdxl.safetensors"}),
                "video_model": (["svd", "none", "ltx-2", "wan2.2"], {"default": "svd"}),
                
                # --- System Context ---
                "seed": ("INT", {"default": 0}),
                "bypass_validation": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                # Collapsed or minor options depending on frontend bindings
                "lighting": (["soft", "dramatic", "neon", "daylight"], {"default": "soft"}),
                "camera_motion": (["dynamic", "pan", "zoom", "static"], {"default": "dynamic"}),
                "continuity_mode": (["last_frame", "none", "blend"], {"default": "last_frame"}),
                "continuity_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0}),
                "output_format": (["mp4", "gif", "webm"], {"default": "mp4"}),
                "reference_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("job_status",)
    FUNCTION = "execute"
    CATEGORY = "🎬 VdoFlash"

    async def execute(
        self, 
        topic_script, 
        duration_seconds, 
        style_type, 
        video_resolution, 
        aspect_ratio, 
        fps, 
        image_model, 
        video_model, 
        seed, 
        bypass_validation,
        lighting="soft", 
        camera_motion="dynamic", 
        continuity_mode="last_frame", 
        continuity_strength=0.4, 
        output_format="mp4", 
        reference_image=None
    ):
        # 1. Physical Memory Cleanup
        mm.unload_all_models()
        torch.cuda.empty_cache()

        # 2. Package configuration
        cfg = {
            "topic": topic_script,
            "duration": duration_seconds,
            "fps": fps,
            "seed": seed,
            "bypass_validation": bypass_validation,
            "style": {"type": style_type, "lighting": lighting, "camera": camera_motion},
            "video": {"resolution": int(video_resolution), "aspect_ratio": aspect_ratio},
            "models": {"image": image_model, "video": video_model},
            "continuity": {"mode": continuity_mode, "strength": continuity_strength},
            "output": {"format": output_format}
        }

        # 3. Build Orchestration Graph (Strict DAG)
        from .core.orchestrator import GraphOrchestrator
        orchestrator = GraphOrchestrator()
        graph = orchestrator.build_orchestration_graph(cfg)

        # 4. Finalize and Submit to Native Queue
        from server import PromptServer
        import uuid
        import execution
        
        try:
            server = PromptServer.instance
            prompt_id = str(uuid.uuid4())
            
            # Identify the output nodes for this orchestrated task
            outputs_to_execute = ["final_output"]
            
            # Prepare submission data (mimics server.py:955)
            # number, prompt_id, prompt, extra_data, outputs_to_execute, sensitive
            number = server.number
            server.number += 1
            
            extra_data = {
                "client_id": "VdoFlash_Internal", # Can be extracted from request if needed
                "create_time": int(time.time() * 1000)
            }
            
            # Put directly into the queue to avoid API deadlock
            server.prompt_queue.put((number, prompt_id, graph, extra_data, outputs_to_execute, {}))
            
            status_msg = f"Orchestrated script into {len(cfg['topic'])} character blocks. Rendering job {prompt_id} sent to history."
            print(f"VdoFlash: {status_msg}")
            
        except Exception as e:
            status_msg = f"Orchestration failure: {e}"
            print(f"ERROR: {status_msg}")
            import traceback
            traceback.print_exc()

        return (status_msg,)
