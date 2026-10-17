import sys
import fire
from src.cli import RagCLI


def main() -> None:
    """Main entry point for the CLI."""
    try:
        fire.Fire(RagCLI)
    # except Exception as e:
    #     print(f"ERROR: '{e}'")
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.", file=sys.stderr)
        sys.exit()


if __name__ == '__main__':
    main()
