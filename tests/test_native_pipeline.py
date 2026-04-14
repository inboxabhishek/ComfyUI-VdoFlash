import sys
import unittest
from unittest.mock import MagicMock, patch
import os
import torch

# 1. Mock ComfyUI environment
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.model_management'] = MagicMock()
sys.modules['folder_paths'] = MagicMock()
sys.modules['nodes'] = MagicMock()

# 2. Import our orchestrator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.orchestrator import GraphOrchestrator, is_vhs_available

class TestNativeOrchestrator(unittest.TestCase):

    def setUp(self):
        self.orchestrator = GraphOrchestrator()
        self.cfg = {
            "topic": "Cyberpunk Car",
            "duration": 10,
            "fps": 24,
            "seed": 42,
            "style": {"type": "cinematic", "lighting": "soft"},
            "video": {"resolution": 1024, "aspect_ratio": "16:9"},
            "models": {"image": "sdxl.safetensors", "video": "svd"},
            "continuity": {"mode": "last_frame", "strength": 0.4},
            "output": {"format": "mp4"}
        }

    def test_scene_planning(self):
        print("\nTEST: Testing Scene Planning logic...")
        # 10s duration / 5s per scene = 2 scenes
        scenes = self.orchestrator.plan_scenes(self.cfg)
        self.assertEqual(len(scenes), 2)
        self.assertIn("scene 0", scenes[0]["prompt"])
        self.assertIn("scene 1", scenes[1]["prompt"])
        print(f"OK: Generated {len(scenes)} scenes.")

    def test_dag_link_integrity(self):
        print("\nTEST: Testing DAG Link Integrity (No Raw Tensors)...")
        graph = self.orchestrator.build_orchestration_graph(self.cfg)
        
        # We cruise through all nodes and inputs
        for node_id, node_data in graph.items():
            for input_name, input_val in node_data.get("inputs", {}).items():
                # Any link MUST be a list [node_id, index]
                if isinstance(input_val, list):
                    self.assertEqual(len(input_val), 2, f"Invalid link in node {node_id}: {input_val}")
                    self.assertIsInstance(input_val[0], str, f"Node ID must be string in {node_id}: {input_val}")
                elif isinstance(input_val, (str, int, float, bool)):
                    # Primitive types are allowed
                    pass
                else:
                    self.fail(f"ILLEGAL INPUT TYPE in node {node_id}, input {input_name}: {type(input_val)}. Only primitives or [node_id, index] allowed.")
        
        print("OK: All DAG links are structurally sound (No raw tensors found).")

    def test_vhs_fallback_logic(self):
        print("\nTEST: Testing VHS Fallback...")
        
        # Test Case 1: VHS Missing
        with patch('core.orchestrator.is_vhs_available', return_value=False):
            graph = self.orchestrator.build_orchestration_graph(self.cfg)
            self.assertEqual(graph["final_output"]["class_type"], "SaveImage")
            print("OK: Correctly fell back to SaveImage when VHS is missing.")
            
        # Test Case 2: VHS Available
        with patch('core.orchestrator.is_vhs_available', return_value=True):
            graph = self.orchestrator.build_orchestration_graph(self.cfg)
            self.assertEqual(graph["final_output"]["class_type"], "VHS_VideoCombine")
            print("OK: Correctly used VHS_VideoCombine when VHS is available.")

if __name__ == '__main__':
    unittest.main()
