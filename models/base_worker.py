import os
import sys
import json
import time
import logging

from typing import Type
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from abc import ABC, abstractmethod

from utils.image_io import decode_images
from utils.transport import connect_to_router, send_response_tensor, send_response_tensor_shm

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent.parent))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class BaseWorker(ABC):

    # Constants
    POLL_TIMEOUT_MS = 1000
    IDLE_SHUTDOWN_SECONDS = 60
    SHUTDOWN_MESSAGE_TYPE = "SHUTDOWN"

    # THIS SHOULD NOT BE HARDCODED,
    ROUTER_CONNECT = "tcp://10.10.151.14:5555"

    logger = logging.getLogger(__name__)

    class BaseRequest(BaseModel):
        type: str
        request_id: str
        num_images: int

    class BaseResponse(BaseModel):
        type: str
        request_id: str
        answer: dict
        model_name: str

    @property
    @abstractmethod
    def request_model(self) -> Type[BaseRequest]:
        pass

    @property
    @abstractmethod
    def response_model(self) -> Type[BaseResponse]:
        pass

    @abstractmethod
    def parse_args():
        pass


    @abstractmethod
    def load_model(self, args):
        self.logger.info("Model loaded!")
        pass


    @abstractmethod
    def inference(self, request: BaseRequest, frames: list) -> BaseResponse:
        self.logger.info("Inference done!")
        pass


    def run(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        
        args = self.parse_args()
        self.load_model(args)

        socket = connect_to_router(args.model_id, args.router_connect, worker_id=args.worker_id)
        last_work_time = time.monotonic()

        while True:
            socket.send_json({"type": "HEARTBEAT", "model_name": args.model_id})

            if not socket.poll(timeout=BaseWorker.POLL_TIMEOUT_MS):
                if time.monotonic() - last_work_time >= BaseWorker.IDLE_SHUTDOWN_SECONDS:
                    self.logger.info(
                        "No work received for %s seconds, shutting down",
                        BaseWorker.IDLE_SHUTDOWN_SECONDS,
                    )
                    return
                continue

            last_work_time = time.monotonic()
            request_id = None

            try:
                frames = socket.recv_multipart()
                worker_received_time = time.time_ns()

                payload_dict = json.loads(frames[1].decode("utf-8"))
                payload = self.request_model.model_validate(payload_dict)
                if payload.type == BaseWorker.SHUTDOWN_MESSAGE_TYPE:
                    self.logger.info(
                        "Received %s message, shutting down",
                        BaseWorker.SHUTDOWN_MESSAGE_TYPE,
                    )
                    return

                request_id = payload.request_id
                started_decode_time = time.time_ns()
                pil_images = decode_images(frames[2:]) # should go up until payload.num_images
                finished_decode_time = time.time_ns()

                started_inference_time = time.time_ns()
                arrays = self.inference(payload, pil_images)
                finished_inference_time = time.time_ns()
                worker_response_started_time = time.time_ns()

                arrays_def = {
                    name: {"dtype": str(array.dtype), "shape": array.shape, "frame_index": i}
                    for i, (name, array) in enumerate(arrays.items())
                }
                profile = dict(payload_dict.get("profile", {}))
                profile |= {
                    "worker_received_time": worker_received_time,
                    "started_decode_time": started_decode_time,
                    "finished_decode_time": finished_decode_time,
                    "started_inference_time": started_inference_time,
                    "finished_inference_time": finished_inference_time,
                    "worker_response_started_time": worker_response_started_time,
                    "decode_seconds": (finished_decode_time - started_decode_time) / 1e9,
                    "inference_seconds": (finished_inference_time - started_inference_time) / 1e9,
                }

                response = {
                    "type": "SUCCESS",
                    "request_id": request_id,
                    "answer": arrays_def,
                    "model_name": args.model_id,
                    "profile": profile,
                }
                self.response_model.model_validate(response)
                if payload_dict.get("return_transport") == "shared_memory":
                    send_response_tensor_shm(socket, response, arrays)
                else:
                    send_response_tensor(socket, response, arrays)

            except Exception:
                self.logger.exception("Job processing failed")
                socket.send_json(
                    {
                        "type": "ERROR",
                        "request_id": request_id,
                        "message": "Job processing failed",
                        "model_name": args.model_id,
                    }
                )
