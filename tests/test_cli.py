import json

import extraction.__main__ as cli
from extraction.schemas import failed_result

from .conftest import PNG_BYTES


def test_bulk_folder_outputs_one_line_per_file(tmp_path, monkeypatch, capsys):
    (tmp_path / "sub").mkdir()
    for name in ("a.png", "sub/b.jpg", "c.pdf", "notes.txt"):
        (tmp_path / name).write_bytes(PNG_BYTES)
    monkeypatch.setattr(
        cli, "extract_with_ocr", lambda p: failed_result("ocr", "paddleocr", "x", "y").model_copy(update={"errors": []})
    )
    rc = cli.main([str(tmp_path), "--branch", "ocr"])
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [r["file"].split("/")[-1] for r in rows] == ["a.png", "c.pdf", "b.jpg"]  # PDFs included
    assert "raw_text" not in rows[0]["ocr"]
    assert rc == 0


def test_single_file_pretty_json(tmp_path, monkeypatch, capsys):
    img = tmp_path / "a.png"
    img.write_bytes(PNG_BYTES)
    monkeypatch.setattr(cli, "extract_with_vlm", lambda p: failed_result("vlm", "m", "vlm_unavailable", "x"))
    rc = cli.main([str(img), "--branch", "vlm"])
    out = json.loads(capsys.readouterr().out)
    assert out["vlm"]["errors"][0]["code"] == "vlm_unavailable" and rc == 1


def test_no_files(tmp_path, capsys):
    assert cli.main([str(tmp_path)]) == 2
