import os
import random
from contextlib import contextmanager

import numpy as np
import torch


def set_seed(seed: int):
    """Set seeds for python, numpy and torch for reproducible runs.

    This also configures cuDNN to be deterministic where possible.
    On CUDA >= 10.2, the environment variable CUBLAS_WORKSPACE_CONFIG
    must be set before enabling deterministic algorithms; this function
    sets it automatically if it is not already present.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # On CUDA >= 10.2, torch.use_deterministic_algorithms(True) requires
    # CUBLAS_WORKSPACE_CONFIG to be set or a RuntimeError is raised.
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        try:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        except Exception:
            pass

    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError as e:
        print(f"⚠️  Deterministic algorithms unavailable: {e}")
        print("   Training will proceed without full determinism.")
        print(f"   To fix, set: CUBLAS_WORKSPACE_CONFIG=:4096:8")
    except Exception:
        # Older torch versions don't have this API
        pass

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    """Worker init function for DataLoader to have deterministic workers."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def derive_sample_seed(base_seed: int, epoch: int, index: int) -> int:
    """Derive a stable per-sample seed from run seed, epoch, and item index."""
    return int(base_seed + epoch * 1_000_003 + index)


@contextmanager
def temporary_seed(seed: int):
    """Temporarily seed python, numpy, and torch for deterministic augmentations."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = None
    
    # Only access CUDA in main process to avoid fork issues on Linux DataLoader workers
    if torch.cuda.is_available():
        try:
            cuda_states = torch.cuda.get_rng_state_all()
        except RuntimeError:
            # CUDA not initialized in this process (worker process)
            pass

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
        except RuntimeError:
            pass

    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            try:
                torch.cuda.set_rng_state_all(cuda_states)
            except RuntimeError:
                pass
