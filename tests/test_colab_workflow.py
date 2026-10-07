"""Protect experiment evidence and portable Colab bundles."""
import ast
import importlib.util
import json
import pathlib
import zipfile

import pytest

from labkit import report

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LABEL = {"intent": "doi_tra", "urgency": "cao", "product": "chuột", "sentiment": "tieu_cuc"}
GOOD = json.dumps(LABEL, ensure_ascii=False)


def test_qualitative_comparison_distinguishes_loss_from_low_absolute_score():
    target = [{"input": str(i), "label": LABEL} for i in range(3)]
    baseline = [{**row, "baseline_b": pred} for row, pred in zip(target, [GOOD, "bad", "bad"])]
    rows = report.compare_predictions(target, baseline, ["bad", "bad", GOOD])
    indexed = {row["i"]: row for row in rows}
    assert indexed[0]["outcome"] == "loss"
    assert indexed[1]["outcome"] == "tie"  # FT wrong is not necessarily FT losing.
    assert indexed[2]["outcome"] == "win"
    assert indexed[0]["baseline_b_pred"] == GOOD
    assert indexed[2]["ft_pred"] == GOOD


def test_qualitative_comparison_rejects_changed_ticket_or_label():
    record = {"input": "ticket", "label": LABEL}
    with pytest.raises(ValueError, match="mismatch"):
        report.compare_predictions([record], [{**record, "input": "other", "baseline_b": GOOD}], [GOOD])
    with pytest.raises(ValueError, match="counts"):
        report.compare_predictions([record], [], [GOOD])


def test_input_archive_excludes_secrets_holdout_and_trained_state(tmp_path):
    exporter = load_script("export_colab")
    for name in exporter.DATA_FILES:
        path = tmp_path / "data" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    (tmp_path / ".env").write_text("HF_TOKEN=secret", encoding="utf-8")
    (tmp_path / "data/holdout_secret.jsonl").write_text("private", encoding="utf-8")
    (tmp_path / "results").mkdir()
    (tmp_path / "results/verdict.json").write_text("{}", encoding="utf-8")
    adapter = tmp_path / "adapters/correct"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter / "checkpoint-1").mkdir()
    (adapter / "checkpoint-1/optimizer.pt").write_bytes(b"optimizer")
    output = tmp_path / "input.zip"
    exporter.build_archive(tmp_path, output, "input")
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert not any(name.endswith("/.env") or "holdout" in name or "results/" in name
                       or "adapters/" in name for name in names)
        assert f"{exporter.ARCHIVE_ROOT}/data/train_seed.jsonl" in names
    exporter.build_archive(tmp_path, output, "progress")
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert f"{exporter.ARCHIVE_ROOT}/adapters/correct/adapter_model.safetensors" in names
        assert f"{exporter.ARCHIVE_ROOT}/results/verdict.json" in names
        assert not any("checkpoint-" in name for name in names)


def test_missing_measurements_do_not_generate_a_fake_report(tmp_path):
    builder = load_script("build_measured_report")
    with pytest.raises(FileNotFoundError, match="Run NB1-NB5"):
        builder.render(tmp_path)
    assert not (tmp_path / "submission/MEASURED_RESULTS.md").exists()


def test_measured_report_uses_real_scores_and_preserves_ties(tmp_path):
    builder = load_script("build_measured_report")
    directory = tmp_path / "results"
    directory.mkdir()
    fixtures = {
        "mask_proof.json": {"answer_is_supervised": True, "question_is_masked": True,
                            "supervised_fraction": 0.25, "supervised_preview": GOOD},
        "template_check.json": {"keeps_think": False},
        "token_stats.json": {"p95": 300, "suggested_max_length": 512},
        "baselines_frozen.json": {"tier": "T4", "model": "example/model", "n_target": 5,
                                  "n_regression": 15, "smoke_mode": False},
        "verdict.json": {"comparison": [{"run": "(c)", "target": 0.75, "regression": 0.9,
                                         "format": 1.0, "latency_ms": 123.0, "n": 5}],
                         "verdict": {"passed": True, "target_delta": 0.25, "regression_delta": 0.0}},
        "autopsy.json": [{"run": run, "target": 0.75, "format": 1.0}
                         for run in ("correct", "attn_only", "wrong_lr", "qlora")],
        "qualitative.json": [{"i": i, "ticket": "ticket | xuống\ndòng", "label": LABEL,
                              "baseline_b_pred": GOOD, "ft_pred": GOOD, "base_score": 1.0,
                              "ft_score": 1.0, "outcome": "tie"} for i in range(5)],
    }
    for name, content in fixtures.items():
        (directory / name).write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
    report.append_row({"run": "correct", "final_loss": 0.3}, results_dir=directory)
    for run in ("attn_only", "wrong_lr", "qlora"):
        report.append_row({"run": run, "final_loss": 0.4}, results_dir=directory)
    content = builder.render(tmp_path)
    assert "0.75" in content and "123.0" in content
    assert "0 ca thắng, 0 ca thua, 5 ca hoà" in content
    assert "Chỉ đo được 0 ca FT thua" in content
    assert "ticket \\| xuống<br>dòng" in content


def test_generated_colab_cells_are_valid_python_and_use_full_eval():
    builder = load_script("build_colab")
    for path in (ROOT / "colab").glob("*.ipynb"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        for cell in raw["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
                assert cell["outputs"] == [] and cell["execution_count"] is None
    setup = builder.BOOTSTRAP
    assert 'os.environ.pop(name, None)' in setup and '"EVAL_LIMIT"' in setup
    assert "VinUni-AI20k" not in setup
    assert 'SOURCE = "upload"' in setup
