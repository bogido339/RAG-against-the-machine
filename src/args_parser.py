
from src.error_classes import RagError


class Parser:
    """Validate raw command-line arguments."""

    def _check_option(
        self, args: list[str], option: str, value_type: type
    ) -> None:
        """Check whether an option exists and has a valid value."""
        if option not in args:
            raise RagError(f"Missing required option: {option}")

        index = args.index(option)

        if index + 1 >= len(args):
            raise RagError(f"Missing value after {option}")

        value = args[index + 1]

        if value_type is int:
            try:
                int(value)
            except (ValueError, TypeError):
                raise RagError(
                    f"Invalid value '{value}' for {option}. "
                    "Expected an integer."
                ) from None

        elif value_type is str:
            if not value.strip() or value.startswith("--"):
                raise RagError(
                    f"Invalid value '{value}' for {option}. "
                    "Expected a non-empty string."
                )

    def index_input(self, args: list[str]) -> None:
        """Validate index arguments."""
        if len(args) != 2:
            raise RagError(
                "Invalid index arguments. "
                "Expected: --max_chunk_size <integer>"
            )

        self._check_option(args, "--max_chunk_size", int)

    def search_input(self, args: list[str]) -> None:
        """Validate search arguments."""
        if not args:
            raise RagError("Missing query.")

        if len(args) != 3:
            raise RagError(
                "Invalid search arguments. "
                "Expected: <query> --k <integer>"
            )

        if not args[0].strip():
            raise RagError("Query cannot be empty.")

        self._check_option(args, "--k", int)

    def search_dataset_input(self, args: list[str]) -> None:
        """Validate search_dataset arguments."""
        if len(args) != 6:
            raise RagError(
                "Invalid search_dataset arguments. Expected: "
                "--dataset_path <path> "
                "--save_directory <path> --k <integer>"
            )

        self._check_option(args, "--dataset_path", str)
        self._check_option(args, "--save_directory", str)
        self._check_option(args, "--k", int)

    def answer_input(self, args: list[str]) -> None:
        """Validate answer arguments."""
        if not args:
            raise RagError("Missing query.")

        if len(args) != 3:
            raise RagError(
                "Invalid answer arguments. "
                "Expected: <query> --k <integer>"
            )

        if not args[0].strip():
            raise RagError("Query cannot be empty.")

        self._check_option(args, "--k", int)

    def answer_dataset_input(self, args: list[str]) -> None:
        """Validate answer_dataset arguments."""
        if len(args) != 4:
            raise RagError(
                "Invalid answer_dataset arguments. Expected: "
                "--student_search_results_path <path> "
                "--save_directory <path>"
            )

        self._check_option(
            args, "--student_search_results_path", str
        )
        self._check_option(
            args, "--save_directory", str
        )

    def evaluate_input(self, args: list[str]) -> None:
        """Validate evaluate arguments."""
        if len(args) != 4:
            raise RagError(
                "Invalid evaluate arguments. Expected: "
                "--student_search_results_path <path> "
                "--dataset_path <path>"
            )

        self._check_option(
            args, "--student_search_results_path", str
        )
        self._check_option(
            args, "--dataset_path", str
        )
