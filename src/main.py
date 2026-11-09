import sys


def main() -> None:
    """Main entry point for the CLI."""
    try:
        import fire
        from src.cli import RagCLI

        fire.Fire(RagCLI)

    except KeyboardInterrupt:
        print(
            "\nPipeline interrupted by user.",
            file=sys.stderr
        )
        raise SystemExit(130)

    except Exception as error:
        print(
            f"ERROR: {error}",
            file=sys.stderr
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
