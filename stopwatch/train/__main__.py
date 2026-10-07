"""Entrypoint for torchrun -m stopwatch.train."""

import os
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    """Load a train recipe and execute its configured multi-node preset."""
    # Set before importing torch or anything that can initialize CUDA.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    from stopwatch.config import TrainConfig, parse_config
    from stopwatch.train.trainer import train

    config = parse_config(argv)
    if not isinstance(config, TrainConfig):
        raise TypeError("Training requires a train recipe")
    train(config)


if __name__ == "__main__":
    main()
