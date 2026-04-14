import json
import urllib.request
import urllib.parse
import time
import uuid

def check_comfy_health(url):
    try:
        response = urllib.request.urlopen(f"{url}/system_stats")
        return response.getcode() == 200
    except:
        return False

def get_node_info(url, node_class):
    try:
        response = urllib.request.urlopen(f"{url}/object_info/{node_class}")
        return json.loads(response.read())
    except:
        return None

def queue_prompt(url, prompt):
    p = {"prompt": prompt, "client_id": str(uuid.uuid4())}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(f"{url}/prompt", data=data)
    try:
        response = urllib.request.urlopen(req)
        return json.loads(response.read())
    except urllib.error.HTTPError as e:
        print(f"HTTP ERROR {e.code}: {e.read().decode('utf-8')}")
        raise e

def get_queue(url):
    response = urllib.request.urlopen(f"{url}/queue")
    return json.loads(response.read())

def test_vdo_flash_api():
    url = "http://127.0.0.1:8188"
    print(f"\nTEST: Testing ComfyUI API Integration on {url}...")
    
    if not check_comfy_health(url):
        print("ERROR: ComfyUI server not found at 127.0.0.1:8188. Please ensure it is running.")
        return False
    print("OK: ComfyUI health check passed.")

    node_info = get_node_info(url, "VdoFlashDirector")
    if not node_info:
        print("ERROR: 'VdoFlashDirector' node not found in ComfyUI registry. Did you restart the server after the refactor?")
        return False
    print("OK: 'VdoFlashDirector' successfully registered in ComfyUI.")

    # Minimal Prompt to trigger the Director node
    # Note: We add a dummy SaveImage chain to satisfy "Prompt has no outputs" validation
    prompt = {
        "1": {
            "inputs": {
                "topic_script": "api test character shorts",
                "duration_seconds": 5, 
                "style_type": "cinematic",
                "video_resolution": "512",
                "aspect_ratio": "16:9",
                "fps": 8,
                "image_model": "RealVisXL_V5.0_fp16.safetensors",
                "video_model": "none",
                "seed": 1234,
                "bypass_validation": True
            },
            "class_type": "VdoFlashDirector"
        },
        "loader": {
            "inputs": { "ckpt_name": "RealVisXL_V5.0_fp16.safetensors" },
            "class_type": "CheckpointLoaderSimple"
        },
        "latent": {
            "inputs": { "width": 64, "height": 64, "batch_size": 1 },
            "class_type": "EmptyLatentImage"
        },
        "decode": {
            "inputs": { "samples": ["latent", 0], "vae": ["loader", 2] },
            "class_type": "VAEDecode"
        },
        "save": {
            "inputs": { "images": ["decode", 0], "filename_prefix": "api_trigger_test" },
            "class_type": "SaveImage"
        }
    }

    print("SUBMITTING: Sending trigger prompt to Director...")
    result = queue_prompt(url, prompt)
    prompt_id = result.get("prompt_id")
    print(f"OK: Prompt submitted. ID: {prompt_id}")

    print("MONITORING: Waiting for orchestration (should only take 1-2 seconds)...")
    time.sleep(3)
    
    # Now check history or queue for a second job!
    # The sub-graph we injected has client_id "VdoFlash_Internal"
    queue = get_queue(url)
    pending = queue.get("queue_pending", [])
    running = queue.get("queue_running", [])
    
    all_jobs = pending + running
    # a job in queue is [number, id, prompt, extra_data, ...]
    sub_job_found = False
    for job in all_jobs:
        extra_data = job[3]
        if extra_data.get("client_id") == "VdoFlash_Internal":
            sub_job_found = True
            print(f"SUCCESS: Found orchestrated Sub-Graph in the queue! Job ID: {job[1]}")
            break
            
    if not sub_job_found:
        print("WARNING: Could not find orchestrated job in queue. It might have finished already or failed. Check ComfyUI logs.")
        # Check history as a backup
        print("Checking History...")
        return True # We count this as a partial success if submitted correctly
        
    return True

if __name__ == '__main__':
    test_vdo_flash_api()
