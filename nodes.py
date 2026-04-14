from .core.engine import VideoEngine

class VdoFlashDirectorNode:

    @classmethod
    def INPUT_TYPES(cls):
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
                "image_model": (["sdxl", "realvisxl", "flux"], {"default": "sdxl"}),
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

    RETURN_TYPES = ("IMAGE",)
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
        # Package everything into the unified schema
        cfg = {
            "topic": topic_script,
            "duration": duration_seconds,
            "fps": fps,
            "seed": seed,
            "bypass_validation": bypass_validation,
            "style": {
                "type": style_type,
                "lighting": lighting,
                "camera": camera_motion
            },
            "video": {
                "resolution": int(video_resolution),
                "aspect_ratio": aspect_ratio
            },
            "models": {
                "image": image_model,
                "video": video_model
            },
            "continuity": {
                "mode": continuity_mode,
                "strength": continuity_strength
            },
            "output": {
                "format": output_format
            }
        }

        engine = VideoEngine()
        return (await engine.run(cfg, reference_image),)
