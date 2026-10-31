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
        sys.exit(130)


if __name__ == '__main__':
    main()




# import sys


# def main() -> None:
#     """Main entry point for the CLI."""
#     try:
#         import fire
#         from src.cli import RagCLI

#         fire.Fire(RagCLI)

#     except KeyboardInterrupt:
#         print(
#             "\nPipeline interrupted by user.",
#             file=sys.stderr
#         )
#         raise SystemExit(130) from None

#     except Exception as error:
#         print(
#             f"ERROR: {error}",
#             file=sys.stderr
#         )
#         raise SystemExit(1) from None


# if __name__ == "__main__":
#     main()