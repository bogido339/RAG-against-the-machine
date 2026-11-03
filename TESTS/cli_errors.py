#!/usr/bin/env python3
"""CLI error-handling tests for "RAG against the machine".

Run from the project root (where `src/` and `data/` live):

    python cli_errors.py                       # run everything
    python cli_errors.py -c index -c search    # only some commands
    python cli_errors.py -i SEARCH-019 -i INDEX-02   # by id / id prefix
    python cli_errors.py -l                    # list tests, run nothing
    python cli_errors.py -m                    # list cases not automated
    python cli_errors.py -v -x                 # verbose, stop at first fail

Pass criteria
  err : exit code != 0 and no Python traceback
  ok  : exit code == 0 and no Python traceback
  any : no Python traceback (policy is yours: skip, fail, or record)

Run `index` first: tests that need an index are skipped without one.
Tests touching data/processed or data/raw back them up and restore them.
"""
import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

BASE = "uv run python -m src"
DATA, PROC, RAW = Path("data"), Path("data/processed"), Path("data/raw")
IS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0
TMP = Path(tempfile.mkdtemp(prefix="ragtest_")).resolve()
TB = "Traceback (most recent call last)"
COMMANDS = ["index", "search", "search_dataset", "answer",
            "answer_dataset", "evaluate"]

MANUAL = """\
Not automated (need fault injection, a special OS state, or unit tests):
  *-004                    shell splits lines before Python runs
  *-007                    unexpected internal bug: raise one on purpose
  INDEX-010..015           write/replace failures, ENOSPC, EROFS, serialization
  INDEX-020, 024..028      read races, chunk/BM25/save failures
  SEARCH-011..014, 022     index/chunk sync, duplicate chunks, bad spans
  SEARCH_DATASET-016, 017, 021..024, 029..032, 036, 037, 041
  ANSWER-016..022          model config, OOM, context overflow, empty output
  ANSWER_DATASET-016, 017, 021..024, 029, 030, 032..038, 044
  EVALUATE-016, 017, 020, 021, 025 (BrokenPipe: `... | head -n1`)
  *-021 / *-022 / *-024    disk full, read-only FS, interrupted write
                           (use a small tmpfs mount or a loop device)
"""


# ---------------------------------------------------------------- fixtures
def src(path=None, a=0, b=10):
    return {"file_path": path or REAL, "first_character_index": a,
            "last_character_index": b}


def item(i="q1", srcs=None, text="What is vLLM?"):
    d = {"question_id": i, "question": text}
    if srcs is not None:
        d["retrieved_sources"] = srcs
    return d


def res(*items):
    return json.dumps({"search_results": list(items), "k": 5})


def ds(*qs):
    return json.dumps({"rag_questions": list(qs)})


def q(i, t="What is vLLM?"):
    return {"question_id": i, "question": t}


REAL = next((str(p) for p in RAW.rglob("*") if p.is_file()), "data/raw/x.py")
F = {}


def f(name, data):
    p = TMP / name
    p.write_bytes(data if isinstance(data, bytes) else data.encode())
    return str(p)


def build_fixtures():
    # Adjust the JSON shapes below if your Pydantic models differ.
    F["good"] = f("good.json", ds(q("q1")))
    F["bad_json"] = f("bad.json", "{not json")
    F["bad_utf8"] = f("utf8.json", b"\xff\xfe\x00\xff")
    F["schema"] = f("schema.json", '{"foo": 1}')
    F["wrongroot"] = f("root.json", json.dumps({"questions": [q("q1")]}))
    F["empty"] = f("empty.json", "")
    F["noq"] = f("noq.json", ds())
    F["dup"] = f("dup.json", ds(q("a"), q("a")))
    F["nofield"] = f("nofield.json", ds({"question_id": "a"}))
    F["blank"] = f("blank.json", ds(q("a", "   ")))
    F["gt"] = f("gt.json", ds({**q("q1"), "sources": [src()], "answer": "x"}))
    F["ok_res"] = f("ok_res.json", res(item("q1", [src()])))
    F["res_empty"] = f("res_empty.json", res())
    F["res_dup"] = f("res_dup.json", res(item("a", [src()]), item("a", [src()])))
    F["res_nofield"] = f("res_nofield.json", json.dumps(
        {"search_results": [{"question_id": "a"}], "k": 5}))
    F["res_blank"] = f("res_blank.json", res(item("a", [src()], "   ")))
    F["nosrc"] = f("nosrc.json", res(item("q1")))
    F["srcmissing"] = f("srcmissing.json",
                        res(item("q1", [src("data/raw/__nope__.py")])))
    F["outside"] = f("outside.json", res(item("q1", [src("/etc/passwd")])))
    F["oob"] = f("oob.json", res(item("q1", [src(a=0, b=10**9)])))
    F["span_big"] = f("span_big.json", res(item("q1", [src(a=0, b=5000)])))
    F["span_neg"] = f("span_neg.json", res(item("q1", [src(a=-5, b=10)])))
    F["mismatch"] = f("mismatch.json", res(item("zzz", [src()])))
    F["dir"] = str(TMP / "adir")
    (TMP / "adir").mkdir()
    F["file"] = f("afile.txt", "x")
    F["missing"] = str(TMP / "nope.json")
    F["out"] = str(TMP / "out")
    F["unreadable"] = f("unreadable.json", ds(q("q1")))
    F["ro_dir"] = str(TMP / "ro")
    (TMP / "ro").mkdir()
    if not IS_ROOT:
        os.chmod(F["unreadable"], 0o000)
        os.chmod(F["ro_dir"], 0o555)
    (TMP / "shadow" / "bm25s").mkdir(parents=True)
    (TMP / "shadow" / "bm25s" / "__init__.py").write_text(
        "raise ModuleNotFoundError(\"No module named 'bm25s' (simulated)\")\n")


def cleanup():
    for k in ("unreadable", "ro_dir"):
        try:
            os.chmod(F[k], 0o700)
        except OSError:
            pass
    shutil.rmtree(TMP, ignore_errors=True)


# ------------------------------------------------------------------ setups
@contextmanager
def moved(path):
    bak = path.with_name(path.name + ".ragtest_bak")
    existed = path.exists()
    if existed:
        path.rename(bak)
    try:
        yield
    finally:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()
        if existed:
            bak.rename(path)


@contextmanager
def snapshot(path):
    bak = path.with_name(path.name + ".ragtest_bak")
    shutil.copytree(path, bak)
    try:
        yield
    finally:
        shutil.rmtree(path, ignore_errors=True)
        bak.rename(path)


@contextmanager
def chmod(path, mode):
    old = path.stat().st_mode & 0o777
    path.chmod(mode)
    try:
        yield
    finally:
        path.chmod(old)


@contextmanager
def corrupt_index():
    with snapshot(PROC):
        for p in PROC.rglob("*"):
            if p.is_file() and p.name != "chunks.json":
                p.write_bytes(b"garbage")
        yield


@contextmanager
def bad_chunks():
    with snapshot(PROC):
        (PROC / "chunks.json").write_text("{bad json")
        yield


@contextmanager
def proc_is_file():
    with moved(PROC):
        PROC.write_text("x")
        yield


@contextmanager
def ro_data():
    with moved(PROC), chmod(DATA, 0o555):
        yield


@contextmanager
def empty_raw():
    with moved(RAW):
        RAW.mkdir()
        yield


@contextmanager
def bad_utf8_raw():
    bad = RAW / "__ragtest_bad.py"
    bad.write_bytes(b"\xff\xfe\x00\xff")
    try:
        yield
    finally:
        bad.unlink(missing_ok=True)


@contextmanager
def unreadable_raw():
    with chmod(RAW, 0o000):
        yield


# ------------------------------------------------------------------- cases
@dataclass
class T:
    id: str
    args: list
    expect: str = "err"
    soft: bool = False        # a traceback is tolerated (fails before CLI starts)
    setup: object = None      # callable -> context manager
    env: dict = field(default_factory=dict)
    kind: str = "run"         # run | interrupt
    needs_index: bool = False
    skip: str = ""


C = []
PERM = "running as root" if IS_ROOT else ""


def add(id_, args, expect="err", **kw):
    C.append(T(id_, args, expect, **kw))


def OFFLINE():
    return {"HF_HOME": str(TMP / "hf"), "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1"}


def index_cases():
    add("INDEX-002", ["index", "--bogus=1"])
    add("INDEX-002", ["unknown_command"])
    add("INDEX-003", ["index", "--max_chunk_size=abc"])
    add("INDEX-005", ["index"], kind="interrupt")
    add("INDEX-006", ["index"], soft=True,
        env={"PYTHONPATH": str(TMP / "shadow")})
    add("INDEX-008", ["index"], setup=ro_data, skip=PERM)
    add("INDEX-009", ["index"], setup=proc_is_file)
    add("INDEX-016", ["index"], setup=lambda: moved(RAW))
    add("INDEX-017", ["index"], setup=empty_raw)
    add("INDEX-018", ["index"], setup=unreadable_raw, skip=PERM)
    add("INDEX-019", ["index"], "any", setup=bad_utf8_raw)
    for v in ("0", "-5"):
        add("INDEX-021", ["index", f"--max_chunk_size={v}"])
    add("INDEX-022", ["index", "--max_chunk_size=2001"])
    for v in ("1.5", "True", "abc"):
        add("INDEX-023", ["index", f"--max_chunk_size={v}"])
    add("INDEX-023", ["index", "--max_chunk_size"])


def query_cases(cmd, I):
    def a(key, args, exp="err", **kw):
        if key in I:
            add(I[key], [cmd, *args], exp, **kw)

    a("usage", [])
    a("unknown", ["hello", "--bogus=1"])
    a("ktype", ["hello", "--k=abc"])
    a("empty", [""])
    a("ws", ["   "])
    a("nonstr", ["123"])
    a("k0", ["hello", "--k=0"])
    a("k0", ["hello", "--k=-1"])
    a("kbad", ["hello", "--k=1.5"])
    a("kbad", ["hello", "--k=True"])
    a("noindex", ["hello"], setup=lambda: moved(PROC))
    a("corrupt", ["hello"], setup=corrupt_index, needs_index=True)
    a("chunks", ["hello"], setup=bad_chunks, needs_index=True)
    a("nomatch", ["zzzxqwv qqqq"], "ok", needs_index=True)
    a("kbig", ["hello", "--k=99999999"], "ok", needs_index=True)
    a("intr", ["hello"], kind="interrupt", needs_index=True)
    a("offline", ["hello"], env=OFFLINE(), needs_index=True)


def items_cases(ids, items, build, **kw):
    for key, fx, exp in items:
        if key in ids:
            perm = PERM if fx == "unreadable" else ""
            add(ids[key], build(F[fx]), exp, skip=perm, **kw)


def out_cases(ids, build, **kw):
    spec = {"filedir": (F["file"], ""),
            "outparentfile": (F["file"] + "/out", ""),
            "outcreate": (F["ro_dir"] + "/out", PERM),
            "outwrite": (F["ro_dir"], PERM)}
    for key, (path, skip) in spec.items():
        if key in ids:
            add(ids[key], build(path), skip=skip, **kw)


def search_dataset_cases():
    cmd = "search_dataset"
    ids = dict(usage="SEARCH_DATASET-001", unknown="SEARCH_DATASET-002",
               ktype="SEARCH_DATASET-003", missing="SEARCH_DATASET-008",
               dir="SEARCH_DATASET-009", perm="SEARCH_DATASET-011",
               badjson="SEARCH_DATASET-012", utf8="SEARCH_DATASET-013",
               schema="SEARCH_DATASET-014", empty="SEARCH_DATASET-015",
               dup="SEARCH_DATASET-033", nofield="SEARCH_DATASET-034",
               blank="SEARCH_DATASET-035", root="SEARCH_DATASET-040")

    def mk(p, out=None, k="5"):
        return [cmd, f"--dataset_path={p}",
                f"--save_directory={out or F['out']}", f"--k={k}"]

    add(ids["usage"], [cmd])
    add(ids["unknown"], [cmd, "--bogus=1"])
    add(ids["ktype"], mk(F["good"], k="abc"))
    add("SEARCH_DATASET-005", mk(F["good"]), kind="interrupt", needs_index=True)
    add("SEARCH_DATASET-038", mk(F["good"], k="0"))
    add("SEARCH_DATASET-038", mk(F["good"], k="-1"))
    add("SEARCH_DATASET-038", mk(F["good"], k="1.5"))
    add("SEARCH_DATASET-039", mk(F["good"], k="99999999"), "ok", needs_index=True)
    add("SEARCH_DATASET-026", mk(F["good"]), setup=lambda: moved(PROC))
    add("SEARCH_DATASET-027", mk(F["good"]), setup=corrupt_index, needs_index=True)
    add("SEARCH_DATASET-028", mk(F["good"]), setup=bad_chunks, needs_index=True)
    items = [("missing", "missing", "err"), ("dir", "dir", "err"),
             ("perm", "unreadable", "err"), ("badjson", "bad_json", "err"),
             ("utf8", "bad_utf8", "err"), ("schema", "schema", "err"),
             ("empty", "empty", "err"), ("empty", "noq", "any"),
             ("dup", "dup", "any"), ("nofield", "nofield", "err"),
             ("blank", "blank", "any"), ("root", "wrongroot", "err")]
    items_cases(ids, items, mk)
    out_cases(dict(filedir="SEARCH_DATASET-010", outparentfile="SEARCH_DATASET-019",
                   outcreate="SEARCH_DATASET-018", outwrite="SEARCH_DATASET-020"),
              lambda o: mk(F["good"], o), needs_index=True)


def answer_dataset_cases():
    cmd = "answer_dataset"
    ids = dict(missing="ANSWER_DATASET-008", dir="ANSWER_DATASET-009",
               perm="ANSWER_DATASET-011", badjson="ANSWER_DATASET-012",
               utf8="ANSWER_DATASET-013", schema="ANSWER_DATASET-014",
               empty="ANSWER_DATASET-015", dup="ANSWER_DATASET-026",
               nofield="ANSWER_DATASET-027", blank="ANSWER_DATASET-028",
               wrongkind="ANSWER_DATASET-039", nosrc="ANSWER_DATASET-040",
               srcmissing="ANSWER_DATASET-041", outside="ANSWER_DATASET-042",
               oob="ANSWER_DATASET-043")

    def mk(p, out=None):
        return [cmd, f"--student_search_results_path={p}",
                f"--save_directory={out or F['out']}"]

    add("ANSWER_DATASET-001", [cmd])
    add("ANSWER_DATASET-002", [cmd, "--bogus=1"])
    add("ANSWER_DATASET-003", [cmd, "--student_search_results_path=1",
                               "--save_directory=2", "--k=abc"])
    add("ANSWER_DATASET-005", mk(F["ok_res"]), kind="interrupt")
    add("ANSWER_DATASET-031", mk(F["ok_res"]), env=OFFLINE())
    items = [("missing", "missing", "err"), ("dir", "dir", "err"),
             ("perm", "unreadable", "err"), ("badjson", "bad_json", "err"),
             ("utf8", "bad_utf8", "err"), ("schema", "schema", "err"),
             ("wrongkind", "good", "err"), ("empty", "empty", "err"),
             ("empty", "res_empty", "any"), ("dup", "res_dup", "any"),
             ("nofield", "res_nofield", "err"), ("blank", "res_blank", "any"),
             ("nosrc", "nosrc", "err"), ("srcmissing", "srcmissing", "err"),
             ("outside", "outside", "err"), ("oob", "oob", "err")]
    items_cases(ids, items, mk)
    out_cases(dict(filedir="ANSWER_DATASET-010", outparentfile="ANSWER_DATASET-019",
                   outcreate="ANSWER_DATASET-018", outwrite="ANSWER_DATASET-020"),
              lambda o: mk(F["ok_res"], o))


def evaluate_cases():
    ids = dict(missing="EVALUATE-008", dir="EVALUATE-009", perm="EVALUATE-011",
               badjson="EVALUATE-012", utf8="EVALUATE-013",
               schema="EVALUATE-014", empty="EVALUATE-015",
               mismatch="EVALUATE-019", span_big="EVALUATE-022",
               span_neg="EVALUATE-024", noq="EVALUATE-023")

    def ev(r, d):
        return ["evaluate", f"--student_search_results_path={r}",
                f"--dataset_path={d}"]

    add("EVALUATE-001", ["evaluate"])
    add("EVALUATE-002", ["evaluate", "--bogus=1"])
    add("EVALUATE-005", ev(F["ok_res"], F["gt"]), kind="interrupt")
    add("EVALUATE-018", ev(F["gt"], F["ok_res"]))                 # swapped
    items = [("missing", "missing", "err"), ("dir", "dir", "err"),
             ("perm", "unreadable", "err"), ("badjson", "bad_json", "err"),
             ("utf8", "bad_utf8", "err"), ("schema", "schema", "err"),
             ("empty", "empty", "err"), ("mismatch", "mismatch", "any"),
             ("span_big", "span_big", "err"), ("span_neg", "span_neg", "err")]
    items_cases(ids, items, lambda p: ev(p, F["gt"]))             # bad results
    items_cases(ids, items[:8], lambda p: ev(F["ok_res"], p))     # bad dataset
    add(ids["noq"], ev(F["res_empty"], F["noq"]), "any")


def build_cases():
    index_cases()
    query_cases("search", dict(
        usage="SEARCH-001", unknown="SEARCH-002", ktype="SEARCH-003",
        intr="SEARCH-005", noindex="SEARCH-008", corrupt="SEARCH-009",
        chunks="SEARCH-010", empty="SEARCH-015", ws="SEARCH-016",
        nomatch="SEARCH-017", nonstr="SEARCH-018", k0="SEARCH-019",
        kbad="SEARCH-020", kbig="SEARCH-021"))
    search_dataset_cases()
    query_cases("answer", dict(
        usage="ANSWER-001", unknown="ANSWER-002", ktype="ANSWER-003",
        intr="ANSWER-005", noindex="ANSWER-008", corrupt="ANSWER-009",
        chunks="ANSWER-010", offline="ANSWER-015", empty="ANSWER-023",
        ws="ANSWER-023", nonstr="ANSWER-023", k0="ANSWER-024",
        kbad="ANSWER-024"))
    answer_dataset_cases()
    evaluate_cases()


# ------------------------------------------------------------------ runner
def judge(t, rc, out):
    tb = TB in out
    if t.expect == "err":
        ok = rc != 0 and (t.soft or not tb)
    elif t.expect == "ok":
        ok = rc == 0 and not tb
    else:
        ok = not tb
    why = []
    if t.expect == "err" and rc == 0:
        why.append("expected non-zero exit, got 0")
    if t.expect == "ok" and rc != 0:
        why.append(f"expected exit 0, got {rc}")
    if tb and not (t.soft and t.expect == "err"):
        why.append("traceback leaked")
    return ("PASS" if ok else "FAIL"), "; ".join(why)


def execute(t, base, timeout, sigint_after):
    env = {**os.environ, **t.env}
    if "PYTHONPATH" in t.env and os.environ.get("PYTHONPATH"):
        env["PYTHONPATH"] += os.pathsep + os.environ["PYTHONPATH"]
    cmd = shlex.split(base) + t.args
    if t.kind == "interrupt":
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, text=True, errors="replace",
                             env=env, start_new_session=True)
        time.sleep(sigint_after)
        if p.poll() is not None:
            return "SKIP", "finished before SIGINT", p.stdout.read(), p.returncode
        os.killpg(p.pid, signal.SIGINT)
        try:
            out, _ = p.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            out, _ = p.communicate()
            return "FAIL", "did not exit after SIGINT", out, None
        rc = p.returncode
        bad = []
        if rc != 130:
            bad.append(f"expected exit 130, got {rc}")
        if TB in out:
            bad.append("traceback leaked")
        return ("FAIL" if bad else "PASS"), "; ".join(bad), out, rc
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                           stdin=subprocess.DEVNULL, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return "FAIL", f"timeout after {timeout}s", "", None
    out = p.stdout + p.stderr
    status, why = judge(t, p.returncode, out)
    return status, why, out, p.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-c", "--command", action="append", choices=COMMANDS)
    ap.add_argument("-i", "--id", action="append",
                    help="test id or prefix, e.g. SEARCH-019 or INDEX-02")
    ap.add_argument("-l", "--list", action="store_true")
    ap.add_argument("-m", "--manual", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-x", "--fail-fast", action="store_true")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--sigint-after", type=float, default=3.0)
    ap.add_argument("--base", default=BASE, help=f"default: {BASE!r}")
    a = ap.parse_args()

    if a.manual:
        print(MANUAL)
        return 0

    build_fixtures()
    build_cases()
    sel = C
    if a.command:
        pre = tuple(c.upper() + "-" for c in a.command)
        sel = [t for t in sel if t.id.startswith(pre)]
    if a.id:
        pre = tuple(i.upper() for i in a.id)
        sel = [t for t in sel if t.id.startswith(pre)]

    if a.list:
        for t in sel:
            print(f"{t.id:22} [{t.expect}] {' '.join(t.args)}")
        print(f"\n{len(sel)} tests")
        cleanup()
        return 0

    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    try:
        for t in sel:
            label = f"{t.id:22} {' '.join(t.args)}"
            if t.skip:
                status, why, out, rc = "SKIP", t.skip, "", None
            elif t.needs_index and not PROC.exists():
                status, why, out, rc = "SKIP", "no index (run `index` first)", "", None
            else:
                cm = t.setup() if t.setup else None
                try:
                    if cm:
                        with cm:
                            status, why, out, rc = execute(
                                t, a.base, a.timeout, a.sigint_after)
                    else:
                        status, why, out, rc = execute(
                            t, a.base, a.timeout, a.sigint_after)
                except OSError as e:
                    status, why, out, rc = "SKIP", f"setup failed: {e}", "", None
            counts[status] += 1
            print(f"{status:4} {label}  (rc={rc}) {why}")
            if (a.verbose or status == "FAIL") and out.strip():
                tail = out.strip().splitlines()[-(40 if a.verbose else 6):]
                print("     | " + "\n     | ".join(tail))
            if status == "FAIL" and a.fail_fast:
                break
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        cleanup()
    print(f"\n{counts['PASS']} passed, {counts['FAIL']} failed, "
          f"{counts['SKIP']} skipped  (see -m for non-automated cases)")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
