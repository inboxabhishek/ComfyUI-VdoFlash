import sys
import unittest
from unittest.mock import MagicMock
import os
import asyncio
import torch

# 1. Mock ComfyUI environment before importing our core
mm = MagicMock()
# Explicitly set return values to ensure they are used even when called with args
mm.get_torch_device = MagicMock(return_value="cuda")
mm.get_total_memory = MagicMock(return_value=16.0 * (1024**3))

sys.modules['comfy'] = MagicMock()
sys.modules['comfy.model_management'] = mm
sys.modules['execution'] = MagicMock()
sys.modules['server'] = MagicMock()

# 2. Add current directory to path so we can import our core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.engine import VideoEngine
from core.validator import validate_config, resolve_dimensions
from core.graph_builder import build_image_graph

class TestVdoFlashEngine(unittest.TestCase):

    def setUp(self):
        # Create a mock config
        self.cfg = {
            "topic": "AI Future",
            "duration": 10,
            "fps": 24,
            "seed": 42,
            "bypass_validation": False,
            "style": {"type": "cinematic", "lighting": "soft", "camera": "dynamic"},
            "video": {"resolution": 1024, "aspect_ratio": "16:9"},
            "models": {"image": "realvisxl", "video": "none"},
            "continuity": {"mode": "last_frame", "strength": 0.4},
            "output": {"format": "mp4"}
        }

    def test_validator_vram_downgrade(self):
        print("\nTEST: Testing Validator VRAM Downgrade...")
        # Simulating Medium VRAM
        vram = {"level": "medium", "res": 1024}
        cfg = validate_config(self.cfg.copy(), vram)
        
        # Should downgrade 1024 -> 768
        self.assertEqual(cfg["video"]["resolution"], 768)
        print("OK: Validator correctly downgraded resolution to 768.")

    def test_dimension_resolution(self):
        print("\nTEST: Testing Dimension Resolution (16:9)...")
        self.cfg["video"]["resolution"] = 1024
        self.cfg["video"]["aspect_ratio"] = "16:9"
        w, h = resolve_dimensions(self.cfg)
        
        self.assertEqual(w, 1024)
        self.assertEqual(h, 576) # 1024 * 9 / 16
        print(f"OK: Resolved dimensions: {w}x{h}")

    def test_graph_builder_mapping(self):
        print("\nTEST: Testing Model Mapping (realvisxl)...")
        graph = build_image_graph("test prompt", 123, 768, 432, image_model="realvisxl")
        ckpt_name = graph["1"]["inputs"]["ckpt_name"]
        
        self.assertEqual(ckpt_name, "RealVisXL_V5.0_fp16.safetensors")
        print(f"OK: Correctly mapped 'realvisxl' to '{ckpt_name}'")

    def test_engine_dry_run(self):
        print("\nTEST: Testing Engine End-to-End (DRY-RUN)...")
        from unittest.mock import patch
        
        engine = VideoEngine(dry_run=True)
        
        # Run the async engine
        with patch('core.engine.get_vram_profile') as mock_vram:
            mock_vram.return_value = {"level": "high", "res": 1024, "video": True}
            
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(engine.run(self.cfg))
            
            # Result should be a tensor [TotalFrames, H, W, C]
            # 10 seconds / 5 sec per scene = 2 scenes. 2 scenes * 24 fps = 48 frames.
            self.assertIsInstance(result, torch.Tensor)
            self.assertEqual(result.shape[0], 48)
            print(f"OK: Engine produced tensor with shape: {result.shape}")

if __name__ == '__main__':
    unittest.main()
