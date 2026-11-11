import sys
import argparse


def validate_input() -> None:
    """Validate CLI arguments before executing Fire."""

    parser = argparse.ArgumentParser(
        description="RAG Pipeline CLI"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    # index
    index_parser = subparsers.add_parser("index")
    index_parser.add_argument(
        "--max_chunk_size",
        type=int,
        default=2000
    )

    # search
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("query", type=str)
    search_parser.add_argument("--k", type=int, default=5)

    # search_dataset
    search_dataset_parser = subparsers.add_parser("search_dataset")
    search_dataset_parser.add_argument("dataset_path", type=str)
    search_dataset_parser.add_argument("save_directory", type=str)
    search_dataset_parser.add_argument("--k", type=int, default=5)

    # answer
    answer_parser = subparsers.add_parser("answer")
    answer_parser.add_argument("query", type=str)
    answer_parser.add_argument("--k", type=int, default=5)

    # answer_dataset
    answer_dataset_parser = subparsers.add_parser("answer_dataset")
    answer_dataset_parser.add_argument(
        "student_search_results_path",
        type=str
    )
    answer_dataset_parser.add_argument(
        "save_directory",
        type=str
    )

    # evaluate
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument(
        "student_search_results_path",
        type=str
    )
    evaluate_parser.add_argument("dataset_path", type=str)

    args = parser.parse_args()

    if hasattr(args, "k") and args.k <= 0:
        parser.error("--k must be a positive integer")

    if hasattr(args, "max_chunk_size"):
        if not 1 <= args.max_chunk_size <= 2000:
            parser.error(
                "--max_chunk_size must be between 1 and 2000"
            )

    if hasattr(args, "query") and not args.query.strip():
        parser.error("query cannot be empty")



def main() -> None:
    """Main entry point for the CLI."""
    try:
        import fire
        from src.cli import RagCLI

        validate_input()

        fire.Fire(RagCLI)

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.", file=sys.stderr)
        raise SystemExit(130)

    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
