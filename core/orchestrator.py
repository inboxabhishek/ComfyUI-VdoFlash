import torch
import folder_paths
import comfy.model_management as mm

def is_vhs_available():
    """Detect if Video Helper Suite is installed for optional enhancement."""
    try:
        import nodes
        return "VHS_VideoCombine" in nodes.NODE_CLASS_MAPPINGS
    except:
        return False

class GraphOrchestrator:
    """
    The orchestrator builds a single, massive ComfyUI DAG (Directed Acyclic Graph)
    aligned with 'ComfyUI Native' execution standards.
    """

    def build_orchestration_graph(self, cfg):
        # Configuration parameters
        width = cfg["video"]["resolution"]
        ar_mult = 9/16 if cfg["video"]["aspect_ratio"] == "16:9" else 16/9
        height = int((width * ar_mult) // 8) * 8
        
        scenes = self.plan_scenes(cfg)
        graph = {}
        
        # 1. Global Loaders (Static across scenes)
        graph["model_loader"] = {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": cfg["models"]["image"]}
        }
        
        # 2. Scene Rendering Sequence
        prev_decode_node = None
        all_decode_nodes = []
        
        for i, scene in enumerate(scenes):
            scene_prefix = f"scene_{i}_"
            
            latent_id = f"{scene_prefix}latent"
            pos_id = f"{scene_prefix}pos"
            neg_id = f"{scene_prefix}neg"
            sampler_id = f"{scene_prefix}sampler"
            decode_id = f"{scene_prefix}decode"
            
            # Define Scene Graph nodes
            graph[latent_id] = {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": width, "height": height, "batch_size": 1}
            }
            
            graph[pos_id] = {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": f"score_9, score_8_up, cinematic, {scene['prompt']}",
                    "clip": ["model_loader", 1]
                }
            }
            
            graph[neg_id] = {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "watermark, text, blurry, low quality",
                    "clip": ["model_loader", 1]
                }
            }
            
            # KSampler with VRAM-aware steps
            vram_steps = 20 if cfg.get("vram_level") == "low" else 25
            
            graph[sampler_id] = {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["model_loader", 0],
                    "seed": cfg["seed"] + i,
                    "steps": vram_steps,
                    "cfg": 7.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "positive": [pos_id, 0],
                    "negative": [neg_id, 0],
                    "latent_image": [latent_id, 0],
                    "denoise": 1.0
                }
            }
            
            graph[decode_id] = {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": [sampler_id, 0],
                    "vae": ["model_loader", 2]
                }
            }
            
            all_decode_nodes.append(decode_id)

        # 3. Batch Assembly using standard ImageBatch nodes
        current_batch_node = all_decode_nodes[0]
        for i in range(1, len(all_decode_nodes)):
            batch_id = f"batch_join_{i}"
            graph[batch_id] = {
                "class_type": "ImageBatch",
                "inputs": {
                    "image1": [current_batch_node, 0],
                    "image2": [all_decode_nodes[i], 0]
                }
            }
            current_batch_node = batch_id

        # 4. Final Output (Portability Fix)
        # We check for VHS, but fall back to standard SaveImage
        if is_vhs_available() and cfg["output"].get("format") != "images":
            graph["final_output"] = {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "images": [current_batch_node, 0],
                    "frame_rate": cfg["fps"],
                    "loop_count": 0,
                    "format": "video/h264-mp4",
                    "save_output": True,
                    "filename_prefix": "VdoFlash"
                }
            }
        else:
            # Safe Fallback: Standard SaveImage
            graph["final_output"] = {
                "class_type": "SaveImage",
                "inputs": {
                    "images": [current_batch_node, 0],
                    "filename_prefix": "VdoFlash"
                }
            }
        
        return graph

    def plan_scenes(self, cfg):
        style_prompt = f"style: {cfg['style']['type']}, lighting: {cfg['style']['lighting']}"
        # Orchestrate 1 scene every 5 seconds of duration
        scene_count = max(1, cfg["duration"] // 5)
        return [
            {"id": i, "prompt": f"{cfg['topic']} scene {i} - {style_prompt}"}
            for i in range(scene_count)
        ]
