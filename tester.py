
import subprocess
from dataclasses import dataclass

BASE_CMD = ["uv", "run", "python", "-m", "src"]


@dataclass
class TestCase:
    name: str
    args: list[str]
    should_pass: bool


TESTS = {
    "index": [
        TestCase("Valid size", ["--max_chunk_size", "2000"], True),
        TestCase("Minimum size", ["--max_chunk_size", "1"], True),
        TestCase("Zero size", ["--max_chunk_size", "0"], False),
        TestCase("Negative size", ["--max_chunk_size", "-1"], False),
        TestCase("Too large", ["--max_chunk_size", "2001"], False),
        TestCase("Float size", ["--max_chunk_size", "2.5"], False),
        TestCase("String size", ["--max_chunk_size", "abc"], False),
        TestCase("Boolean size", ["--max_chunk_size", "True"], False),
        TestCase("Empty size", ["--max_chunk_size", ""], False),
        TestCase("Missing value", ["--max_chunk_size"], False),
        TestCase("Missing option", [], False),
        TestCase("Wrong option", ["--size", "100"], False),
        TestCase("Extra argument", ["--max_chunk_size", "100", "extra"], False),
        TestCase("Duplicate option", [
            "--max_chunk_size", "100", "--max_chunk_size", "200"
        ], False),
    ],
    "search": [
        TestCase("Valid search", ["hello", "--k", "3"], True),
        TestCase("Single result", ["hello", "--k", "1"], True),
        TestCase("Empty query", ["", "--k", "3"], False),
        TestCase("Spaces query", ["   ", "--k", "3"], False),
        TestCase("Missing query", ["--k", "3"], False),
        TestCase("Missing k", ["hello"], False),
        TestCase("Missing k value", ["hello", "--k"], False),
        TestCase("Zero k", ["hello", "--k", "0"], False),
        TestCase("Negative k", ["hello", "--k", "-1"], False),
        TestCase("Float k", ["hello", "--k", "2.5"], False),
        TestCase("String k", ["hello", "--k", "abc"], False),
        TestCase("Boolean k", ["hello", "--k", "True"], False),
        TestCase("Dash k", ["hello", "--k", "-"], False),
        TestCase("Empty k", ["hello", "--k", ""], False),
        TestCase("Wrong option", ["hello", "--top_k", "3"], False),
        TestCase("Extra argument", ["hello", "--k", "3", "extra"], False),
        TestCase("Duplicate k", [
            "hello", "--k", "3", "--k", "5"
        ], False),
    ],
    "search_dataset": [
        TestCase("Valid arguments", [
            "--dataset_path", "data/datasets/test.json",
            "--save_directory", "data/output/test",
            "--k", "3"
        ], True),
        TestCase("Missing all arguments", [], False),
        TestCase("Missing dataset path", [
            "--save_directory", "data/output/test", "--k", "3"
        ], False),
        TestCase("Empty dataset path", [
            "--dataset_path", "",
            "--save_directory", "data/output/test", "--k", "3"
        ], False),
        TestCase("Missing save directory", [
            "--dataset_path", "test.json", "--k", "3"
        ], False),
        TestCase("Empty save directory", [
            "--dataset_path", "test.json",
            "--save_directory", "", "--k", "3"
        ], False),
        TestCase("Missing k", [
            "--dataset_path", "test.json",
            "--save_directory", "output"
        ], False),
        TestCase("Zero k", [
            "--dataset_path", "test.json",
            "--save_directory", "output", "--k", "0"
        ], False),
        TestCase("Negative k", [
            "--dataset_path", "test.json",
            "--save_directory", "output", "--k", "-1"
        ], False),
        TestCase("Invalid k", [
            "--dataset_path", "test.json",
            "--save_directory", "output", "--k", "abc"
        ], False),
        TestCase("Missing file", [
            "--dataset_path", "nonexistent.json",
            "--save_directory", "output", "--k", "3"
        ], False),
        TestCase("Duplicate option", [
            "--dataset_path", "test.json",
            "--save_directory", "output",
            "--k", "3", "--k", "4"
        ], False),
    ],
    "answer": [
        TestCase("Valid answer", ["hello", "--k", "3"], True),
        TestCase("Empty query", ["", "--k", "3"], False),
        TestCase("Spaces query", ["   ", "--k", "3"], False),
        TestCase("Missing query", ["--k", "3"], False),
        TestCase("Missing k", ["hello"], False),
        TestCase("Missing k value", ["hello", "--k"], False),
        TestCase("Zero k", ["hello", "--k", "0"], False),
        TestCase("Negative k", ["hello", "--k", "-1"], False),
        TestCase("Float k", ["hello", "--k", "1.5"], False),
        TestCase("String k", ["hello", "--k", "abc"], False),
        TestCase("Boolean k", ["hello", "--k", "True"], False),
        TestCase("Dash k", ["hello", "--k", "-"], False),
        TestCase("Extra argument", ["hello", "--k", "3", "extra"], False),
    ],
    "answer_dataset": [
        TestCase("Valid arguments", [
            "--student_search_results_path", "results.json",
            "--save_directory", "output"
        ], True),
        TestCase("Missing all arguments", [], False),
        TestCase("Missing results path", [
            "--save_directory", "output"
        ], False),
        TestCase("Empty results path", [
            "--student_search_results_path", "",
            "--save_directory", "output"
        ], False),
        TestCase("Missing save directory", [
            "--student_search_results_path", "results.json"
        ], False),
        TestCase("Empty save directory", [
            "--student_search_results_path", "results.json",
            "--save_directory", ""
        ], False),
        TestCase("Missing file", [
            "--student_search_results_path", "nonexistent.json",
            "--save_directory", "output"
        ], False),
        TestCase("Extra argument", [
            "--student_search_results_path", "results.json",
            "--save_directory", "output", "extra"
        ], False),
    ],
    "evaluate": [
        TestCase("Valid arguments", [
            "--student_search_results_path", "results.json",
            "--dataset_path", "dataset.json"
        ], True),
        TestCase("Missing all arguments", [], False),
        TestCase("Missing results path", [
            "--dataset_path", "dataset.json"
        ], False),
        TestCase("Empty results path", [
            "--student_search_results_path", "",
            "--dataset_path", "dataset.json"
        ], False),
        TestCase("Missing dataset path", [
            "--student_search_results_path", "results.json"
        ], False),
        TestCase("Empty dataset path", [
            "--student_search_results_path", "results.json",
            "--dataset_path", ""
        ], False),
        TestCase("Extra argument", [
            "--student_search_results_path", "results.json",
            "--dataset_path", "dataset.json", "extra"
        ], False),
    ],
}


def run_test(command: str, test: TestCase) -> bool:
    cmd = BASE_CMD + [command] + test.args

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

        actual_pass = result.returncode == 0
        passed = actual_pass == test.should_pass

        status = "PASS" if passed else "FAIL"

        print(f"\n[{status}] {command}: {test.name}")
        print(f"Command: {cmd}")
        print(f"Exit code: {result.returncode}")

        output = (result.stderr or result.stdout).strip()
        if output:
            print(f"Output: {output[:500]}")

        return passed

    except subprocess.TimeoutExpired:
        print(f"\n[TIMEOUT] {command}: {test.name}")
        return False

    except OSError as error:
        print(f"\n[ERROR] {command}: {test.name}")
        print(f"Reason: {error}")
        return False


def main() -> None:
    total = 0
    passed = 0

    for command, tests in TESTS.items():
        print(f"\n{'=' * 60}")
        print(f"TESTING: {command.upper()}")
        print(f"{'=' * 60}")

        for test in tests:
            total += 1
            if run_test(command, test):
                passed += 1

    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"Total tests : {total}")
    print(f"Passed      : {passed}")
    print(f"Failed      : {total - passed}")
    print(f"Success rate: {passed / total * 100:.1f}%")


if __name__ == "__main__":
    main()
