#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker script for multi-GPU inference.

This script is designed to be called via subprocess with CUDA_VISIBLE_DEVICES
set in the environment BEFORE any imports happen. This ensures proper GPU isolation.

Usage:
    CUDA_VISIBLE_DEVICES=0 python inference_worker.py <args_json_path> <result_json_path>
"""

import json
import logging
import os
import sys
import time
from multiprocessing import shared_memory

# Log start time before any heavy imports
start_time = time.time()

logger = logging.getLogger(__name__)


def configure_worker_logging(gpu_id: int) -> logging.Logger:
    worker_logger = logging.getLogger(f"{__name__}.gpu{gpu_id}")
    if not worker_logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[GPU %(name)s] %(levelname)s: %(message)s"))
        worker_logger.addHandler(handler)
        worker_logger.setLevel(logging.DEBUG)
    return worker_logger


def main():
    if len(sys.argv) != 3:
        logger.error("Usage: %s <args_json_path> <result_json_path>", sys.argv[0])
        sys.exit(1)

    args_path = sys.argv[1]
    result_path = sys.argv[2]

    # Load arguments from JSON
    with open(args_path) as f:
        args = json.load(f)

    gpu_id = args["gpu_id"]
    data_chunk = args["data_chunk"]
    model_path = args["model_path"]
    config = args["config"]
    target_dim = args["target_dim"]
    scale_factor = args["scale_factor"]
    nsample = args["nsample"]
    seed = args["seed"]
    deterministic = args["deterministic"]
    preprocess_model_dir = args["preprocess_model_dir"]
    use_dpm_solver = args.get("use_dpm_solver", False)  # NEW: Extract DPM parameters
    dpm_steps = args.get("dpm_steps", 20)  # NEW: Extract DPM parameters

    try:
        # NOW import torch - after CUDA_VISIBLE_DEVICES is already set
        import random

        import numpy as np
        import torch

        # Add path to find modules in the ad_diffusion directory
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        from models.main_model import TSDiffuser_Generic
        from sdk.inference_ad import (
            evaluate_ad_tesseract2,
            get_dataloader,
            get_dataloader_from_windows,
        )

        # Configure logging for this worker
        worker_logger = configure_worker_logging(gpu_id)
        worker_logger.debug("Worker STARTED at %.2f", start_time)

        # NEW: Log DPM usage
        if use_dpm_solver:
            worker_logger.debug("Using DPM-Solver with %d steps", dpm_steps)

        if deterministic:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # With CUDA_VISIBLE_DEVICES set, cuda:0 refers to our assigned GPU
        device = torch.device("cuda:0")

        # Set seed for reproducibility
        device_seed = seed + gpu_id
        random.seed(device_seed)
        np.random.seed(device_seed)
        torch.manual_seed(device_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(device_seed)

        # Load model
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model = TSDiffuser_Generic(
            config,
            device=device,
            target_dim=target_dim,
            ratio=0.7,
        )
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
        model = model.to(device).eval()

        # Create dataloader for this chunk
        if isinstance(data_chunk, dict) and data_chunk.get("window_shm"):
            shm = shared_memory.SharedMemory(name=data_chunk["shm_name"])
            # Convert JSON-serialized dtype (str) and shape (list) back to numpy types
            shm_dtype = np.dtype(data_chunk["dtype"])
            shm_shape = tuple(data_chunk["shape"])
            windows = np.ndarray(shm_shape, dtype=shm_dtype, buffer=shm.buf)
            loader1, loader2 = get_dataloader_from_windows(
                windows,
                split=data_chunk["split"],
                window_indices=data_chunk["window_indices"],
            )
            shm.close()
        else:
            loader1, loader2 = get_dataloader(
                data_chunk,
                target_dim,
                scale_factor=scale_factor,
                model_dir=preprocess_model_dir,
            )

        # NEW: Run inference with DPM parameters
        results = evaluate_ad_tesseract2(
            model,
            loader1,
            loader2,
            nsample=nsample,
            use_dpm_solver=use_dpm_solver,  # NEW: Pass DPM parameters
            dpm_steps=dpm_steps,  # NEW: Pass DPM parameters
        )

        end_time = time.time()
        worker_logger.debug("Worker FINISHED at %.2f (took %.2fs)", end_time, end_time - start_time)

        # Save results as JSON (convert numpy arrays to lists)
        serializable_results = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in results.items()}
        with open(result_path, "w") as f:
            json.dump({"gpu_id": gpu_id, "results": serializable_results}, f)

    except Exception as e:
        end_time = time.time()
        try:
            worker_logger.error("Worker FAILED at %.2f: %s", end_time, e)
        except NameError:
            logger.error("[GPU %s] Worker FAILED at %.2f: %s", gpu_id, end_time, e)

        # Save error as JSON
        with open(result_path, "w") as f:
            json.dump({"gpu_id": gpu_id, "error": str(e)}, f)
        sys.exit(1)


if __name__ == "__main__":
    main()
