import sys
import fire
from src.cli import RagCLI


def main() -> None:
    """Main entry point for the CLI."""
    try:
        fire.Fire(RagCLI)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.", file=sys.stderr)
        sys.exit()


if __name__ == '__main__':
    main()
