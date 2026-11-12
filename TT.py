
from src.error_classes import RagError


class Parser:
    """Validate raw command-line arguments."""

    def _check_option(
        self, args: list[str], option: str, value_type: type
    ) -> None:
        """Check whether an option has a valid value."""
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

    def _validate_options(
        self,
        args: list[str],
        allowed: dict[str, type],
        required: list[str] | None = None,
        positional: int = 0
    ) -> None:
        """Validate allowed options and positional arguments."""
        if required is None:
            required = []

        found = set()
        position = 0
        i = 0

        while i < len(args):
            arg = args[i]

            if arg.startswith("--"):
                if arg not in allowed:
                    raise RagError(f"Unknown argument: {arg}")

                if arg in found:
                    raise RagError(f"Duplicate argument: {arg}")

                self._check_option(args, arg, allowed[arg])
                found.add(arg)
                i += 2

            else:
                if position >= positional:
                    raise RagError(f"Unexpected argument: {arg}")

                if not arg.strip():
                    raise RagError("Positional argument cannot be empty.")

                position += 1
                i += 1

        for option in required:
            if option not in found:
                raise RagError(f"Missing required argument: {option}")

        if position < positional:
            raise RagError("Missing required positional argument.")

    def index_input(self, args: list[str]) -> None:
        """Validate index arguments."""
        self._validate_options(
            args,
            allowed={"--max_chunk_size": int}
        )

    def search_input(self, args: list[str]) -> None:
        """Validate search arguments."""
        self._validate_options(
            args,
            allowed={"--k": int},
            positional=1
        )

    def search_dataset_input(self, args: list[str]) -> None:
        """Validate search_dataset arguments."""
        self._validate_options(
            args,
            allowed={
                "--dataset_path": str,
                "--save_directory": str,
                "--k": int
            },
            required=["--dataset_path", "--save_directory"]
        )

    def answer_input(self, args: list[str]) -> None:
        """Validate answer arguments."""
        self._validate_options(
            args,
            allowed={"--k": int},
            positional=1
        )

    def answer_dataset_input(self, args: list[str]) -> None:
        """Validate answer_dataset arguments."""
        self._validate_options(
            args,
            allowed={
                "--student_search_results_path": str,
                "--save_directory": str
            },
            required=[
                "--student_search_results_path",
                "--save_directory"
            ]
        )

    def evaluate_input(self, args: list[str]) -> None:
        """Validate evaluate arguments."""
        self._validate_options(
            args,
            allowed={
                "--student_search_results_path": str,
                "--dataset_path": str
            },
            required=[
                "--student_search_results_path",
                "--dataset_path"
            ]
        )
