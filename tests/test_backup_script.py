"""deploy/backup.sh, run by the `backup` and `restore-datasets` services.

Each pass dumps the database, then copies the dataset blobs the backup lacks;
only dumps are ever pruned. A restore copies back the blobs the datasets
volume lacks. Run here with a stub pg_dump under the host's POSIX sh; the
deploy-compose CI job runs it in the real containers.
"""
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "deploy" / "backup.sh"
SCHEME = "qs-ds-sha256-v1"
DAY = 86400

pytestmark = pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX sh")


def blob_name(char: str, shard: str = "ab") -> str:
    return shard + char * 62


def put_blob(root: Path, name: str, content: bytes) -> Path:
    path = root / SCHEME / name[:2] / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    path.chmod(0o444)
    return path


def blobs(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def age(path: Path, days: float) -> None:
    then = time.time() - days * DAY
    os.utime(path, (then, then))


@pytest.fixture
def env(tmp_path):
    """BACKUP_DIR, DATASET_ROOT and a stub pg_dump: it writes DUMP, stores
    the blob named by STORE_DURING_DUMP while it runs, and exits FAIL_DUMP."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "pg_dump"
    stub.write_text('#!/bin/sh\n'
                    'printf DUMP\n'
                    'if [ -n "${STORE_DURING_DUMP:-}" ]; then\n'
                    '  mkdir -p "$(dirname "$STORE_DURING_DUMP")" && printf stored-during-dump > "$STORE_DURING_DUMP"\n'
                    'fi\n'
                    'exit "${FAIL_DUMP:-0}"\n', newline="\n")
    stub.chmod(0o755)
    (tmp_path / "backups").mkdir()
    (tmp_path / "datasets").mkdir()
    return {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "BACKUP_DIR": (tmp_path / "backups").as_posix(),
            "DATASET_ROOT": (tmp_path / "datasets").as_posix(),
            "BACKUP_INTERVAL": "0"}


def run(env, *args, **extra):
    return subprocess.run(["sh", SCRIPT.as_posix(), *args], env={**os.environ, **env, **extra},
                          capture_output=True, text=True, timeout=60)


def dumps(env) -> list[Path]:
    return sorted(Path(env["BACKUP_DIR"]).glob("quantumsentinel-*.dump"))


def test_a_pass_dumps_the_database_then_copies_every_blob(env):
    datasets, backups = Path(env["DATASET_ROOT"]), Path(env["BACKUP_DIR"])
    put_blob(datasets, blob_name("a"), b"first")
    put_blob(datasets, blob_name("c", "cd"), b"second")
    # Not blobs: the worker's unfinished write, and anything else on the volume.
    unfinished = datasets / SCHEME / "ab" / ".tmp-k3j9x0qa"
    unfinished.write_bytes(b"half")
    unfinished.chmod(0)  # the backup container could not read it
    (datasets / "notes.txt").write_bytes(b"x")
    (datasets / SCHEME / "ab" / ("cd" + "c" * 62)).write_bytes(b"wrong shard")
    (datasets / SCHEME / "ab" / ("ab" + "a" * 10)).write_bytes(b"too short")
    (datasets / SCHEME / "ab" / "nested").mkdir()
    (datasets / SCHEME / "ab" / "nested" / blob_name("e")).write_bytes(b"too deep")
    try:
        result = run(env)
    finally:
        unfinished.chmod(0o600)
    assert result.returncode == 0, result.stderr
    assert "backup ok: quantumsentinel-" in result.stderr and ", 2 new dataset blobs" in result.stderr
    [dump] = dumps(env)
    assert dump.read_bytes() == b"DUMP"
    assert not list(backups.glob("*.partial"))
    assert blobs(backups / "datasets") == {f"{SCHEME}/ab/{blob_name('a')}": b"first",
                                           f"{SCHEME}/cd/{blob_name('c', 'cd')}": b"second"}


def test_blobs_stored_while_the_dump_runs_are_copied(env):
    """Every dataset in a dump had its blob stored before it was registered,
    so copying after the dump finishes misses none of them."""
    late = Path(env["DATASET_ROOT"]) / SCHEME / "ef" / blob_name("f", "ef")
    result = run(env, STORE_DURING_DUMP=late.as_posix())
    assert result.returncode == 0, result.stderr
    assert blobs(Path(env["BACKUP_DIR"]) / "datasets") == {f"{SCHEME}/ef/{late.name}": b"stored-during-dump"}


def test_a_failed_dump_keeps_the_old_dumps_and_still_copies_the_blobs(env):
    backups = Path(env["BACKUP_DIR"])
    old = backups / "quantumsentinel-20200101000000.dump"
    old.write_bytes(b"old")
    age(old, 30)
    put_blob(Path(env["DATASET_ROOT"]), blob_name("a"), b"first")
    result = run(env, FAIL_DUMP="1")
    assert result.returncode == 1
    assert "pg_dump failed" in result.stderr and "backup failed: 1 new dataset blobs" in result.stderr
    assert dumps(env) == [old]
    assert not list(backups.glob("*.partial"))
    assert blobs(backups / "datasets") == {f"{SCHEME}/ab/{blob_name('a')}": b"first"}


def test_only_dumps_older_than_the_retention_are_pruned(env):
    backups = Path(env["BACKUP_DIR"])
    old, recent = backups / "quantumsentinel-20200101000000.dump", backups / "quantumsentinel-20200102000000.dump"
    other = backups / "operator-notes.txt"
    for path, days in ((old, 9), (recent, 1), (other, 30)):
        path.write_bytes(b"x")
        age(path, days)
    kept = put_blob(backups / "datasets", blob_name("a"), b"backed up a month ago")
    age(kept, 30)
    age(kept.parent, 30)
    result = run(env)
    assert result.returncode == 0, result.stderr
    assert old not in dumps(env) and recent in dumps(env) and len(dumps(env)) == 2
    assert other.exists()
    assert kept.read_bytes() == b"backed up a month ago"


def test_a_pass_copies_only_what_the_backup_lacks(env):
    datasets, mirror = Path(env["DATASET_ROOT"]), Path(env["BACKUP_DIR"]) / "datasets"
    present, empty, missing = blob_name("a"), blob_name("b"), blob_name("c")
    for name in (present, empty, missing):
        put_blob(datasets, name, b"content of " + name.encode())
    put_blob(mirror, present, b"already backed up")
    put_blob(mirror, empty, b"").chmod(0o644)  # a copy whose bytes never reached the disk
    leftover = mirror / SCHEME / "ab" / f".tmp-qs-copy-{missing}"
    leftover.write_bytes(b"cut short")
    result = run(env)
    assert result.returncode == 0, result.stderr
    assert ", 2 new dataset blobs" in result.stderr
    assert blobs(mirror) == {f"{SCHEME}/ab/{present}": b"already backed up",
                             f"{SCHEME}/ab/{empty}": b"content of " + empty.encode(),
                             f"{SCHEME}/ab/{missing}": b"content of " + missing.encode()}
    assert run(env).stderr.count(", 0 new dataset blobs") == 1


def test_a_copy_that_does_not_match_its_source_is_not_kept(env, tmp_path):
    """A cp that writes the wrong bytes (a failing disk) leaves no blob."""
    put_blob(Path(env["DATASET_ROOT"]), blob_name("a"), b"first")
    corrupting = tmp_path / "bin" / "cp"
    corrupting.write_text('#!/bin/sh\nprintf corrupted > "$2"\n', newline="\n")
    corrupting.chmod(0o755)
    result = run(env)
    assert result.returncode == 1
    assert "could not copy" in result.stderr and "dataset blobs incomplete (0 copied)" in result.stderr
    assert blobs(Path(env["BACKUP_DIR"]) / "datasets") == {}
    assert len(dumps(env)) == 1


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="needs POSIX permissions, enforced against a non-root user")
def test_an_unreadable_blob_fails_the_pass_and_the_rest_are_copied(env):
    datasets = Path(env["DATASET_ROOT"])
    unreadable = put_blob(datasets, blob_name("a"), b"first")
    put_blob(datasets, blob_name("b"), b"second")
    unreadable.chmod(0)
    try:
        result = run(env)
    finally:
        unreadable.chmod(0o444)
    assert result.returncode == 1
    assert f"could not copy {datasets.as_posix()}/{SCHEME}/ab/{unreadable.name}" in result.stderr
    assert blobs(Path(env["BACKUP_DIR"]) / "datasets") == {f"{SCHEME}/ab/{blob_name('b')}": b"second"}


def test_a_restore_copies_back_the_blobs_the_volume_lacks(env):
    datasets, mirror = Path(env["DATASET_ROOT"]), Path(env["BACKUP_DIR"]) / "datasets"
    for name in (blob_name("a"), blob_name("c", "cd")):
        put_blob(mirror, name, b"backed up " + name.encode())
    put_blob(datasets, blob_name("a"), b"live copy")
    in_flight = datasets / SCHEME / "ab" / ".tmp-k3j9x0qa"  # a worker's write in progress
    in_flight.write_bytes(b"half")
    result = run(env, "restore-datasets")
    assert result.returncode == 0, result.stderr
    assert "restore ok: 1 dataset blobs copied" in result.stderr
    assert blobs(datasets) == {f"{SCHEME}/ab/{blob_name('a')}": b"live copy",
                               f"{SCHEME}/ab/.tmp-k3j9x0qa": b"half",
                               f"{SCHEME}/cd/{blob_name('c', 'cd')}": b"backed up " + blob_name("c", "cd").encode()}
    assert not dumps(env)


def test_a_restore_without_a_dataset_backup_fails(env):
    result = run(env, "restore-datasets")
    assert result.returncode == 1
    assert "is not a directory" in result.stderr and "restore failed" in result.stderr


def test_an_unknown_mode_is_refused(env):
    result = run(env, "restore-everything")
    assert result.returncode == 2
    assert "usage:" in result.stderr
