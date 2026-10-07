"""Render CLI. --config PATH --source ovon|mp3d|hssd --gpu-id INT [--set key=value]."""

import argparse
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    from stopwatch.config import TrainConfig, load_config
    from stopwatch.data.render import render_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", choices=("ovon", "mp3d", "hssd"), required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args(argv)
    config = load_config(args.config, args.overrides)
    if not isinstance(config, TrainConfig):
        parser.error("rendering requires a training recipe")
    render_dataset(config, args.source, gpu_id=args.gpu_id)


if __name__ == "__main__":
    main()
