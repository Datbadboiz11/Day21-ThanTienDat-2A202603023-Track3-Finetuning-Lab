"""Accept Git's Windows line endings without accepting modified evaluation data."""
import hashlib
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lab21_verify", ROOT / "scripts/verify.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def test_reference_checksum_accepts_lf_and_crlf(tmp_path):
    raw = b'{"input":"ticket","label":"correct"}\n'
    expected = hashlib.sha256(raw).hexdigest()[:16]
    path = tmp_path / "eval.jsonl"
    path.write_bytes(raw)
    assert verify._matches_checksum(path, expected)
    path.write_bytes(raw.replace(b"\n", b"\r\n"))
    assert verify._matches_checksum(path, expected)


def test_reference_checksum_rejects_label_or_whitespace_edits(tmp_path):
    raw = b'{"input":"ticket","label":"correct"}\n'
    expected = hashlib.sha256(raw).hexdigest()[:16]
    path = tmp_path / "eval.jsonl"
    path.write_bytes(raw.replace(b"correct", b"wrong").replace(b"\n", b"\r\n"))
    assert not verify._matches_checksum(path, expected)
    path.write_bytes(raw.replace(b'"input":', b'"input": '))
    assert not verify._matches_checksum(path, expected)
