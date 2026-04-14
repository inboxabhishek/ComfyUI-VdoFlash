def validate_config(cfg, vram):
    if cfg.get("bypass_validation", False):
        print("⚠️ Validation bypassed by user.")
        return cfg

    # Example: VRAM Safety
    vid_res = cfg["video"]["resolution"]
    if vram["level"] == "low":
        if vid_res > 512:
            print(f"⚠️ Downgrading resolution to 512 from {vid_res} due to LOW VRAM safety.")
            cfg["video"]["resolution"] = 512
            
        if cfg["models"]["video"] != "none":
            print("⚠️ Disabling video model (low VRAM) - falling back to image sequence.")
            cfg["models"]["video"] = "none"
            
    elif vram["level"] == "medium" and vid_res > 768:
        print(f"⚠️ Downgrading resolution to 768 from {vid_res} due to MEDIUM VRAM safety.")
        cfg["video"]["resolution"] = 768

    # Example: Model Compatibility requirements
    if cfg["models"]["image"] == "sdxl" and cfg["video"]["resolution"] < 768:
        print("⚠️ SDXL degrades at low resolutions, adjusting resolution to 768...")
        cfg["video"]["resolution"] = 768

    return cfg


def resolve_dimensions(cfg):
    res = cfg["video"]["resolution"]
    ar = cfg["video"]["aspect_ratio"]

    # Basic logic to convert aspect ratio and max edge length to width/height
    if ar == "16:9":
        # Snap height to nearest multiple of 8
        return res, int((res * 9 / 16) // 8 * 8)
    elif ar == "9:16":
        return int((res * 9 / 16) // 8 * 8), res
    
    # Default 1:1
    return res, res
