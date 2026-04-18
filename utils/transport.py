import zmq
import numpy
import torch
from multiprocessing import shared_memory, resource_tracker

def connect_to_router(model_name, router_connect, worker_id):
    socket = zmq.Context.instance().socket(zmq.DEALER)
    socket.setsockopt(zmq.IDENTITY, f"{model_name}-{worker_id}".encode("utf-8"))
    socket.connect(router_connect)
    return socket


# this to send numpy arrays
def send_response_array_np(socket, response, A, flags=0, copy=True, track=False):
    socket.send_json(dict(response), flags|zmq.SNDMORE)
    send_arrays_np(socket, A, flags=flags, copy=copy, track=track)

def send_arrays_np(socket, arrays: dict[str, numpy.ndarray], flags=0, copy=True, track=False):
    values = list(arrays.values())
    for arr in values[:-1]:
        socket.send(arr, flags | zmq.SNDMORE, copy=copy, track=track)
    if values:
        socket.send(values[-1], flags, copy=copy, track=track)

def recv_response_array_np(socket, flags=0, copy=True, track=False):
    response = socket.recv_json(flags=flags)

    for array_type in response["answer"].values():
        msg = socket.recv(flags=flags, copy=copy, track=track)
        array = numpy.frombuffer(msg, dtype=array_type["dtype"])
        array = array.reshape(array_type["shape"])
        array_type["data"] = array

    return response

# this to send torch arrays
def send_response_tensor(socket, response, tensors, flags=0, copy=True, track=False):
    socket.send_json(dict(response), flags | zmq.SNDMORE)
    send_tensors(socket, tensors, flags=flags, copy=copy, track=track)

def send_tensors(socket, tensors: dict[str, torch.Tensor], flags=0, copy=True, track=False):
    values = list(tensors.values())
    for tensor in values[:-1]:
        # 1. Move to CPU. 2. Ensure memory is contiguous. 3. View as raw bytes. 4. Convert to byte buffer.
        byte_buffer = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
        socket.send(byte_buffer, flags | zmq.SNDMORE, copy=copy, track=track)
    
    if values:
        byte_buffer = values[-1].detach().cpu().contiguous().view(torch.uint8).numpy()
        socket.send(byte_buffer, flags, copy=copy, track=track)

def recv_response_tensor(socket, flags=0, copy=True, track=False):
    response = socket.recv_json(flags=flags)

    for array_type in response["answer"].values():
        msg = socket.recv(flags=flags, copy=copy, track=track)

        torch_dtype = getattr(torch, array_type["dtype"].replace("torch.", "")) 

        if copy:
            tensor = torch.frombuffer(bytearray(msg), dtype=torch_dtype)
        else:
            tensor = torch.frombuffer(msg, dtype=torch_dtype)

        tensor = tensor.reshape(array_type["shape"])
        
        array_type["data"] = tensor

    return response

def send_response_tensor_shm(socket, response, tensors):
    response = dict(response)
    response["transport"] = "shared_memory"

    for name, tensor in tensors.items():
        byte_buffer = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().reshape(-1)
        shm = shared_memory.SharedMemory(create=True, size=byte_buffer.nbytes)
        shm.buf[:byte_buffer.nbytes] = memoryview(byte_buffer).cast("B")
        name_shm = shm.name
        shm.close()
        resource_tracker.unregister(shm._name, "shared_memory")
        response["answer"][name]["shm_name"] = name_shm
        response["answer"][name]["nbytes"] = byte_buffer.nbytes

    socket.send_json(response)
