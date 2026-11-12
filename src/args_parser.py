from src.error_classes import RagError


class Parser:
    """Validate raw command-line arguments."""

    def _check_option(
        self, args: list[str], option: str, value_type: type
    ) -> None:
        """Check whether an option exists and has a valid value."""
        if option not in args:
            raise RagError(f"Missing argument: {option}")

        index = args.index(option)

        if index + 1 >= len(args):
            raise RagError(f"Missing value for {option}")

        value = args[index + 1]

        if value_type is int:
            try:
                int(value)
            except ValueError:
                raise RagError(
                    f"{option} must be an integer."
                ) from None

        elif value_type is str:
            if not value.strip() or value.startswith("--"):
                raise RagError(
                    f"{option} must be a valid string."
                )

    def index_input(self, args: list[str]) -> None:
        """Validate index arguments."""
        if not args:
            return

        if "--max_chunk_size" in args:
            self._check_option(args, "--max_chunk_size", int)
        elif len(args) != 2:
            raise RagError("error")

    def search_input(self, args: list[str]) -> None:
        """Validate search arguments."""
        if "--query" in args:
            self._check_option(args, "--query", str)

        if "--k" in args:
            self._check_option(args, "--k", int)

    def search_dataset_input(self, args: list[str]) -> None:
        """Validate search_dataset arguments."""
        self._check_option(args, "--dataset_path", str)
        self._check_option(args, "--save_directory", str)

        if "--k" in args:
            self._check_option(args, "--k", int)

    def answer_input(self, args: list[str]) -> None:
        """Validate answer arguments."""
        if "--query" in args:
            self._check_option(args, "--query", str)

        if "--k" in args:
            self._check_option(args, "--k", int)

    def answer_dataset_input(self, args: list[str]) -> None:
        """Validate answer_dataset arguments."""
        self._check_option(
            args, "--student_search_results_path", str
        )
        self._check_option(
            args, "--save_directory", str
        )

    def evaluate_input(self, args: list[str]) -> None:
        """Validate evaluate arguments."""
        self._check_option(
            args, "--student_search_results_path", str
        )
        self._check_option(
            args, "--dataset_path", str
        )
