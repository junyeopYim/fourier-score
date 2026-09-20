"""Evaluate held-out DSM or generated-image FID/IS with one entry point."""

import argparse
import json

from fourier_score.evaluation import evaluate_checkpoint, evaluate_images


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="metric", required=True)

    dsm = commands.add_parser("dsm", help="Evaluate EMA DSM on the held-out split")
    dsm.add_argument("-r", "--checkpoint", "--resume", required=True)
    dsm.add_argument("-o", "--output", required=True)
    dsm.add_argument("--device")
    dsm.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")

    fid = commands.add_parser("fid", help="Compute FID and IS using torch-fidelity")
    fid.add_argument("--real", required=True, help="Directory containing real images")
    fid.add_argument(
        "--generated", required=True, help="Directory containing generated images"
    )
    fid.add_argument("-o", "--output", required=True)
    fid.add_argument("--device", default="cpu", help="cpu or cuda")
    fid.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args(argv)
    if args.metric == "dsm":
        result = evaluate_checkpoint(
            args.checkpoint, args.output, args.set, args.device
        )
    else:
        result = evaluate_images(
            args.real, args.generated, args.output, args.device, args.batch_size
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
