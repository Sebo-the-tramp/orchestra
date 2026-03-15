import os
import zmq
import time
import json
import yaml
import subprocess

WORKERS_PATH = "./workers/"
ROUTER_ADDRESS = "tcp://10.10.151.14:5555"

routing_table = {}   # { worker_id: last_seen_time }
idle_workers = []    # [ worker_id, ... ]
pending_clients = {} # { request_id: client_id } (To route answers back!)
job_queue = []       # [ payload_dict, ... ] (Backlog of jobs waiting for workers)
models_to_workers = {} # { model_name: [worker_id, ...] } (All known workers per model)
worker_to_model = {}   # { worker_id: model_name } (So availability can change without losing registration)


def get_model_config_if_exists(model_name):    
    """Returns the model config dict if the model exists, otherwise None."""
    base_part = model_name.split("/")[0]
    if os.path.isdir(os.path.join(WORKERS_PATH, base_part)):
        with open(os.path.join(WORKERS_PATH, base_part, "config.yaml"), "r") as f:
            lookup = yaml.safe_load(f)['models']
            return lookup.get(model_name)         
    else: None


def build_command_for_model(target_model, model_config):    
    model_path = model_config["basefolder"]
    model_tp = model_config['tp']

    return [
        "gbatch", 
        "--gpus", "1", 
        "--time", "2:00:00", # max time for now
        f"./workers/{model_path}/.venv/bin/python", 
        f"./workers/{model_path}/worker.py", 
        "--model-id", target_model, 
        "--router-connect", ROUTER_ADDRESS, 
        "--tp", str(model_tp)
    ]


def main():
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.bind(ROUTER_ADDRESS)
    
    print("Barebone Queue Broker running...")

    while True:

        # --- 1. RECEIVE ---
        if socket.poll(timeout=1000):
            # print("models_to_worke?rs", models_to_workers)
            frames = socket.recv_multipart()

            # # --- DEBUGGING BLOCK ---
            # print(f"\n--- {time.ctime()} - INCOMING MESSAGE ({len(frames)} frames) ---")
            # for i, frame in enumerate(frames):
            #     # We truncate the output so huge images don't flood your terminal
            #     safe_print = frame[:10] + b"... (truncated)" if len(frame) > 50 else frame
            #     print(f"Frame {i}: {safe_print}")
            # print("--------------------------------------\n")

            # CASE A: It's a CLIENT sending a job
            # We expect at least 4 frames: client_id, empty, worker_id, json_metadata
            if len(frames) >= 4 and frames[1] == b'':
                
                client_id = frames[0]
                # frames[1] is the empty delimiter                
                raw_metadata = frames[2]                      # The JSON bytes
                # here we should dynamically check from the metatdata how many images there are, 
                # but for simplicity we just take all remaining frames as images
                image_frames = frames[3:]                     # All attached images (0 to N)
                
                # Now this will perfectly parse the JSON!
                payload = json.loads(raw_metadata.decode('utf-8'))                  
                
                # Save who asked for this job
                req_id = payload.get("request_id")
                pending_clients[req_id] = client_id
                
                # Add to the queue. 
                # We keep the target_worker_id so the broker knows who to route it to next!
                target_model = payload.get("model")  # e.g. "vlm1" -> b"vlm1"
                job_queue.append({
                    "request_id": req_id,
                    "target_worker": target_model,
                    "payload": payload,
                    "images": image_frames 
                })                
                
                print(f"Queued job {req_id} for model '{target_model}'. Attached images: {len(image_frames)}")

            # CASE B: It's a WORKER pinging or returning a result (2 frames)
            elif len(frames) == 2:                
                worker_id, raw_payload = frames
                payload = json.loads(raw_payload.decode('utf-8'))
                model_name = payload.get("model") or worker_to_model.get(worker_id)
                
                # Always update the heartbeat and keep a stable worker registry per model.
                routing_table[worker_id] = time.time()
                if model_name:
                    worker_to_model[worker_id] = model_name
                    registered_workers = models_to_workers.setdefault(model_name, [])
                    if worker_id not in registered_workers:
                        registered_workers.append(worker_id)

                if worker_id not in idle_workers:
                    print(f"Worker {worker_id.decode()} is now idle.")                                        
                    idle_workers.append(worker_id)

                # If it's a finished job, route it back to the exact Client
                if payload.get("type") == "SUCCESS":                    
                    req_id = payload.get("req_id")
                    print('req_id', req_id)
                    client_id = pending_clients.pop(req_id, None)
                    
                    if client_id:
                        # Forward to client (Must include the empty frame for REQ!)
                        socket.send_multipart([client_id, b"", raw_payload])
                        print(f"Forwarded result {req_id} to Client.")

                # If it's a finished job, route it back to the exact Client
                if payload.get("type") == "ERROR":
                    req_id = payload.get("req_id")
                    client_id = pending_clients.pop(req_id, None)
                    
                    if client_id:
                        # Forward to client (Must include the empty frame for REQ!)
                        socket.send_multipart([client_id, b"", raw_payload])
                        print(f"Forwarded result {req_id} to Client.")


        # --- 2. SPAWN WORKER IF NOT AVAILABLE ---
        # This is where you could add logic to spawn new worker processes if certain models have no        
        for job in job_queue:
            target_model = job["target_worker"]
            if target_model not in models_to_workers:
                print(f"No workers available for model '{target_model}'. Spawning new worker...")

                # Check if the model is known (e.g. we have a worker implementation for it). 
                model_config = get_model_config_if_exists(target_model)
                if model_config is not None:                    
                    gflow_command = build_command_for_model(target_model, model_config)

                    pid = subprocess.Popen(gflow_command)
                    print(f"Spawned worker process with PID {pid.pid} for model '{target_model}'.")

                    # problem we need to add it immediately else it will keep spawning new ones until the first one is registered, but we don't want to register it until it actually connects and pings with its model name.
                    # we shall have differnt states for workers: spawning, idle, busy, dead. Only idle and busy are registered in the routing table and can receive jobs, but as soon as we spawn we add it to a "spawning" set and only move it to idle when it pings with its model name. If it takes too long to ping we consider the spawn failed and remove it from the spawning set, allowing new spawns.
                    routing_table[pid.pid] = time.time()  # Track the spawn time to detect failed spawns

                    models_to_workers.setdefault(target_model, []).append(pid.pid)  # Temporarily register the PID as a worker for this model (will be updated to actual worker_id on ping)
                    worker_to_model[pid.pid] = target_model  # Map the PID to the model

                     
                else:
                    print(f"Model '{target_model}' is unknown. Cannot spawn worker. Please check the model name or implement a worker for it.")
                    continue


        # --- 3. PURGE DEAD WORKERS ---
        now = time.time()
        for wid in list(routing_table.keys()):
            if now - routing_table[wid] > 100.0: # achtung if the model takes more than 5 seconds to process it will be considered dead
                del routing_table[wid]
                if wid in idle_workers:
                    idle_workers.remove(wid)
                model_name = worker_to_model.pop(wid, None)
                if model_name:
                    workers = models_to_workers.get(model_name, [])
                    if wid in workers:
                        workers.remove(wid)
                    if not workers and model_name in models_to_workers:
                        del models_to_workers[model_name]
                print(f"Dropped dead worker. {wid.decode()}")


        # --- 4. DISPATCH JOBS ---
        # Try each queued job once per tick so we don't stall if no worker exists for a model.
        if job_queue and idle_workers:
            dispatch_attempts = len(job_queue)
            for _ in range(dispatch_attempts):
                if not idle_workers:
                    break
                job = job_queue.pop(0)
                
                # 1. Fetch the list (default to empty list if model doesn't exist)            
                available_workers = models_to_workers.get(job["target_worker"], [])
                worker_id = next((wid for wid in available_workers if wid in idle_workers), None)

                # 2. Check if the list has any workers
                if worker_id is None:
                    print(f"No available workers for model '{job['target_worker']}' right now. Re-queuing {job['request_id']}.")
                    job_queue.append(job)  # Put it back at the end of the queue                                
                    continue
                
                # 3. Mark the selected worker busy and rotate it to the end for fairer reuse.
                idle_workers.remove(worker_id)
                available_workers.remove(worker_id)
                available_workers.append(worker_id)

                metadata_bytes = json.dumps(job["payload"]).encode('utf-8')

                # 1. Routing ID
                # 2. Empty delimiter (Required for REQ/REP/DEALER patterns)
                # 3. Payload metadata
                # 4+. The images
                print('worker_id', worker_id)
                frames_to_send = [worker_id, b"", metadata_bytes] + job["images"]
                print(len(frames_to_send))
                # Send exactly 2 frames to the DEALER worker
                socket.send_multipart(frames_to_send)
                print(f"Dispatched {job['request_id']} to worker {worker_id}.")

if __name__ == "__main__":
    main()

# we will need security and iron home security!
