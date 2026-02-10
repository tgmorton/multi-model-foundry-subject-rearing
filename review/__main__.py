"""Allow running as `python -m review [split]`."""

import argparse
import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "output"


def main():
    parser = argparse.ArgumentParser(
        description="Launch the annotation review server",
        usage="python -m review [split] [--port PORT]",
    )
    parser.add_argument(
        "split",
        nargs="?",
        default="test_10M",
        help="Corpus split to review (default: test_10M)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8642,
        help="Port to serve on (default: 8642)",
    )
    args = parser.parse_args()

    corpus_path = DATA_DIR / args.split / "annotated_corpus"
    if not corpus_path.exists():
        available = sorted(
            d.name for d in DATA_DIR.iterdir()
            if d.is_dir() and (d / "annotated_corpus").exists()
        )
        print(f"Error: no annotated corpus at {corpus_path}", file=sys.stderr)
        if available:
            print(f"Available splits: {', '.join(available)}", file=sys.stderr)
        sys.exit(1)

    os.environ["REVIEW_CORPUS_PATH"] = str(corpus_path)

    import uvicorn
    from .server import app

    print(f"Serving {args.split} from {corpus_path}")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
