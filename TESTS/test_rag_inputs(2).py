#!/usr/bin/env python3
"""RAG search input regression tests.

Run from project root:
    uv run python test_rag_inputs.py
    uv run python test_rag_inputs.py --mode unit
    uv run python test_rag_inputs.py --mode cli --timeout 120

This script does not invoke index or dataset commands and does not write results.
"""
import argparse
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


@dataclass
class Case:
    name: str
    query: Any
    k: Any
    valid: bool


UNIT_CASES = [
    Case('empty', '', 3, False),
    Case('spaces', ' ', 3, False),
    Case('newline', '\n', 3, False),
    Case('tab', '\t', 3, False),
    Case('CRLF', '\r\n', 3, False),
    Case('mixed whitespace', '   \n\t   ', 3, False),
    Case('None query', None, 3, False),
    Case('integer query', 123, 3, False),
    Case('float query', 3.14, 3, False),
    Case('boolean query', True, 3, False),
    Case('list query', [], 3, False),
    Case('dict query', {}, 3, False),
    Case('bytes query', b'Python', 3, False),
    Case('punctuation dots', '...', 3, True),
    Case('punctuation exclamations', '!!!', 3, True),
    Case('punctuation question marks', '???', 3, True),
    Case('dashes', '---', 3, True),
    Case('underscores', '___', 3, True),
    Case('digits only', '123456', 3, True),
    Case('emojis', '😀😀😀', 3, True),
    Case('zero width space', '\u200b', 3, True),
    Case('null character', '\x00', 3, False),
    Case('very long query', 'a' * 10001, 3, False),
    Case('embedded option text', 'Python\n--k 0', 3, True),
    Case('path-looking query', '../../etc/passwd', 3, True),
    Case('flag-looking query', '--help', 3, True),
    Case('flag-looking k', '--k', 3, True),
    Case('multilingual', 'é中文العربية', 3, True),
    Case('surrogate character', '\ud800', 3, False),
    Case('Python identifier', '__init__', 3, True),
    Case('zero k', 'Python', 0, False),
    Case('negative k', 'Python', -1, False),
    Case('large negative k', 'Python', -100, False),
    Case('string k', 'Python', '3', False),
    Case('invalid string k', 'Python', 'abc', False),
    Case('float k', 'Python', 3.5, False),
    Case('true k', 'Python', True, False),
    Case('false k', 'Python', False, False),
    Case('None k', 'Python', None, False),
    Case('list k', 'Python', [], False),
    Case('dict k', 'Python', {}, False),
    Case('huge k', 'Python', 10 ** 100, True),
    Case('NaN k', 'Python', float('nan'), False),
    Case('infinite k', 'Python', float('inf'), False),
    Case('minimum k', 'Python', 1, True),
    Case('more than corpus k', 'Python', 1000000, True),
    Case('normal input', 'What is vLLM?', 3, True),
]

# These are true argv arrays (no shell expansion). Most should exit nonzero.
# Commands that could run expensive retrieval or create files are excluded.
CLI_CASES = [
    ('empty query', ['search', '', '--k', '3'], False),
    ('spaces query', ['search', ' ', '--k', '3'], False),
    ('newlines query', ['search', '\n\t', '--k', '3'], False),
    ('zero k', ['search', 'Python', '--k', '0'], False),
    ('negative k', ['search', 'Python', '--k', '-1'], False),
    ('float k', ['search', 'Python', '--k', '3.5'], False),
    ('missing query', ['search', '--k', '3'], False),
    ('missing k value', ['search', 'Python', '--k'], False),
    ('unknown option', ['search', 'Python', '--unknown', '3'], False),
    ('extra argument', ['search', 'Python', '--k', '3', 'extra'], False),
    ('missing dataset paths', ['search_dataset', '--k', '10'], False),
    # Shell `\\ ` turns into a single argv item containing a leading space:
    ('escaped-space argument', ['search_dataset', ' --k', '12'], False),
]


def unit_tests() -> tuple[int, int]:
    # Import inside the function so running CLI-only tests won't import the RAG stack.
    try:
        from src.retriever import Retriever
        from src.error_classes import RetrieverError
    except (ImportError, ModuleNotFoundError) as exc:
        print(f'UNIT SETUP ERROR: {exc}\nAdjust the import paths in unit_tests().')
        return 0, 1

    obj = Retriever.__new__(Retriever)
    method = getattr(obj, '_validate_input', None)
    if method is None:
        print('UNIT SETUP ERROR: Retriever._validate_input does not exist.')
        return 0, 1

    passed = failed = 0
    for c in UNIT_CASES:
        try:
            method(c.query, c.k)
            actual_valid = True
            reason = 'accepted'
        except RetrieverError as exc:
            actual_valid = False
            reason = f'rejected: {exc}'
        except Exception as exc:
            actual_valid = None
            reason = f'unexpected {type(exc).__name__}: {exc}'
        ok = actual_valid is c.valid
        passed += ok
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} [unit] {c.name}: {reason}")
    return passed, failed


def cli_tests(timeout: float) -> tuple[int, int]:
    passed = failed = 0
    for name, args, should_succeed in CLI_CASES:
        cmd = [sys.executable, '-m', 'src', *args]
        try:
            result = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
            succeeded = result.returncode == 0
            ok = succeeded == should_succeed
            details = (result.stderr or result.stdout).strip().splitlines()
            tail = details[-1][:180] if details else '(no message)'
            info = f'exit={result.returncode}, {tail}'
        except subprocess.TimeoutExpired:
            ok = False
            info = f'timeout > {timeout:g}s'
        except OSError as exc:
            ok = False
            info = f'process failure: {exc}'
        passed += ok
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} [cli]  {name}: {info}")
    return passed, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['unit', 'cli', 'all'], default='all')
    parser.add_argument('--timeout', type=float, default=90.0)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')

    passed = failed = 0
    if args.mode in ('unit', 'all'):
        p, f = unit_tests()
        passed += p
        failed += f
    if args.mode in ('cli', 'all'):
        p, f = cli_tests(args.timeout)
        passed += p
        failed += f
    print(f'\nTOTAL: {passed} passed, {failed} failed')
    print('Note: unit tests check input validation, not BM25 retrieval quality.')
    print('CLI tests check exit status only; inspect output for error-message quality.')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
