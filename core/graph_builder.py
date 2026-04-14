def build_image_graph(prompt, seed, width, height, image_model="sdxl"):
    # Map UI Selection to actual filename found in checkpoints folder
    model_mapping = {
        "sdxl": "RealVisXL_V5.0_fp16.safetensors",
        "realvisxl": "RealVisXL_V5.0_fp16.safetensors",
        "flux": "flux-2-klein-base-9b-fp8.safetensors", # Based on previous folder scan
        # Add more mappings as needed
    }
    
    ckpt_name = model_mapping.get(image_model.lower(), "RealVisXL_V5.0_fp16.safetensors")
    
    # The actual SDXL / LTX graphs will replace this stub.
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ckpt_name}
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": f"score_9, score_8_up, score_7_up, {prompt}", "clip": ["1", 1]} # SDXL style prompt
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "seed": seed,
                "steps": 25,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "positive": ["2", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0],
                "denoise": 1.0
            }
        },
        "4": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["1", 2]}
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "text, watermark, blurry, low quality", "clip": ["1", 1]}
        },
        "6": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1}
        }
    }
