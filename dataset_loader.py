"""Tiện ích tải UIT-ViCoQA từ Hugging Face.
Dataset chính thức có cơ chế gated access, vì vậy người dùng cần chấp nhận điều kiện truy cập trước.
"""
import argparse
import json
from pathlib import Path


def export_vicoqa(output_dir="data/vicoqa"):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Hãy cài dependency tùy chọn: pip install 'datasets>=3.0.0'") from exc

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    try:
        ds = load_dataset("uitnlp/vicoqa")
    except Exception as exc:
        raise SystemExit(
            "Không tải được UIT-ViCoQA. Hãy đăng nhập Hugging Face và chấp nhận điều kiện "
            "truy cập dataset uitnlp/vicoqa, sau đó chạy lại chương trình.\n" + str(exc)
        ) from exc

    for split, rows in ds.items():
        path = Path(output_dir) / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Đã lưu {split}: {len(rows)} mẫu -> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/vicoqa")
    args = parser.parse_args()
    export_vicoqa(args.output)
