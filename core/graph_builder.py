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
            "inputs": {"text": prompt, "clip": ["1", 1]}
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "seed": seed,
                "steps": 20
            }
        },
        "4": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["1", 2]}
        }
    }
