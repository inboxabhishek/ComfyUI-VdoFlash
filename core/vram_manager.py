import comfy.model_management as mm

def get_vram_profile():
    device = mm.get_torch_device()
    total = mm.get_total_memory(device) / (1024**3)

    if total < 8:
        return {"level": "low", "res": 512, "video": False}
    elif total < 16:
        return {"level": "medium", "res": 768, "video": True}
    else:
        return {"level": "high", "res": 1024, "video": True}
