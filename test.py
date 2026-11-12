
from src.error_classes import RagError


class Parser:
    """Validate raw command-line arguments for the RAG pipeline."""

    def _parse_args(
        self,
        args: list[str],
        command: str,
        required: list[str],
        optional: dict[str, str] | None = None
    ) -> dict[str, str]:
        """Validate positional arguments and optional flags."""

        if optional is None:
            optional = {}

        values: dict[str, str] = {}
        position = 0
        i = 0

        while i < len(args):
            arg = args[i]

            if arg.startswith("--"):
                if "=" in arg:
                    flag, value = arg.split("=", 1)
                else:
                    flag = arg

                    if i + 1 >= len(args):
                        raise RagError(
                            f"{command}: missing value for '{flag}'."
                        )

                    value = args[i + 1]
                    i += 1

                if flag not in optional:
                    raise RagError(
                        f"{command}: unknown argument '{flag}'."
                    )

                if flag in values:
                    raise RagError(
                        f"{command}: duplicate argument '{flag}'."
                    )

                if not value or value.startswith("--"):
                    raise RagError(
                        f"{command}: invalid value for '{flag}'."
                    )

                values[flag] = value

            else:
                if position >= len(required):
                    raise RagError(
                        f"{command}: too many positional arguments."
                    )

                if not arg.strip():
                    raise RagError(
                        f"{command}: empty positional argument."
                    )

                values[required[position]] = arg
                position += 1

            i += 1

        if position < len(required):
            missing = required[position:]
            raise RagError(
                f"{command}: missing required arguments: "
                f"{', '.join(missing)}"
            )

        return values

    def _positive_int(
        self,
        value: str,
        name: str,
        maximum: int | None = None
    ) -> int:
        """Validate a positive integer."""

        if not value.isascii() or not value.isdecimal():
            raise RagError(
                f"{name} must be a positive integer."
            )

        try:
            number = int(value)
        except ValueError:
            raise RagError(
                f"{name} must be a valid integer."
            ) from None

        if number <= 0:
            raise RagError(
                f"{name} must be greater than 0."
            )

        if maximum is not None and number > maximum:
            raise RagError(
                f"{name} must be between 1 and {maximum}."
            )

        return number

    def index_input(self, args: list[str]) -> None:
        """Validate index command arguments."""

        values = self._parse_args(
            args,
            command="index",
            required=[],
            optional={"--max_chunk_size": "2000"}
        )

        if "--max_chunk_size" in values:
            self._positive_int(
                values["--max_chunk_size"],
                "max_chunk_size",
                maximum=2000
            )

    def search_input(self, args: list[str]) -> None:
        """Validate search command arguments."""

        values = self._parse_args(
            args,
            command="search",
            required=["query"],
            optional={"--k": "5"}
        )

        if not values["query"].strip():
            raise RagError("Query cannot be empty.")

        if "--k" in values:
            self._positive_int(values["--k"], "k")

    def search_dataset_input(self, args: list[str]) -> None:
        """Validate search_dataset command arguments."""

        values = self._parse_args(
            args,
            command="search_dataset",
            required=["dataset_path", "save_directory"],
            optional={"--k": "5"}
        )

        if "--k" in values:
            self._positive_int(values["--k"], "k")

    def answer_input(self, args: list[str]) -> None:
        """Validate answer command arguments."""

        values = self._parse_args(
            args,
            command="answer",
            required=["query"],
            optional={"--k": "5"}
        )

        if not values["query"].strip():
            raise RagError("Query cannot be empty.")

        if "--k" in values:
            self._positive_int(values["--k"], "k")

    def answer_dataset_input(self, args: list[str]) -> None:
        """Validate answer_dataset command arguments."""

        self._parse_args(
            args,
            command="answer_dataset",
            required=[
                "student_search_results_path",
                "save_directory"
            ]
        )

    def evaluate_input(self, args: list[str]) -> None:
        """Validate evaluate command arguments."""

        self._parse_args(
            args,
            command="evaluate",
            required=[
                "student_search_results_path",
                "dataset_path"
            ]
        )
