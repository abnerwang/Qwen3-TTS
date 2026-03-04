# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a 3-column TSV (audio_path, text, speaker_name) to Qwen3-TTS JSONL format."
    )
    parser.add_argument("--input_tsv", type=str, required=True, help="Path to train.tsv")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Output raw jsonl file")
    parser.add_argument(
        "--speaker_ref_json",
        type=str,
        default="",
        help=(
            "Optional json file mapping speaker_name -> ref_audio_path. "
            "If empty, use the first utterance of each speaker as ref_audio."
        ),
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="",
        help="Optional base dir for resolving relative audio paths. Defaults to tsv parent dir.",
    )
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip samples whose audio paths do not exist.",
    )
    return parser.parse_args()


def _resolve_audio_path(audio_path: str, base_dir: Path) -> str:
    p = Path(audio_path)
    if not p.is_absolute():
        p = base_dir / p
    return str(p.resolve())


def main():
    args = parse_args()

    input_tsv = Path(args.input_tsv)
    output_jsonl = Path(args.output_jsonl)
    base_dir = Path(args.base_dir) if args.base_dir else input_tsv.parent

    speaker_ref = {}
    if args.speaker_ref_json:
        with open(args.speaker_ref_json, "r", encoding="utf-8") as f:
            raw_map = json.load(f)
        speaker_ref = {
            k: _resolve_audio_path(v, base_dir)
            for k, v in raw_map.items()
        }

    lines = []
    first_audio_by_speaker = {}
    with open(input_tsv, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for line_idx, row in enumerate(reader, start=1):
            if len(row) != 3:
                raise ValueError(
                    f"Line {line_idx} in {input_tsv} does not have exactly 3 columns: {row}"
                )

            audio_path, text, speaker_name = [x.strip() for x in row]
            if not audio_path or not text or not speaker_name:
                raise ValueError(f"Line {line_idx} has empty fields: {row}")

            audio_path = _resolve_audio_path(audio_path, base_dir)
            if speaker_name not in first_audio_by_speaker:
                first_audio_by_speaker[speaker_name] = audio_path

            if args.skip_missing and not Path(audio_path).exists():
                continue

            lines.append(
                {
                    "audio": audio_path,
                    "text": text,
                    "speaker_name": speaker_name,
                }
            )

    for line in lines:
        spk = line["speaker_name"]
        line["ref_audio"] = speaker_ref.get(spk, first_audio_by_speaker[spk])

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"Wrote {len(lines)} samples to {output_jsonl}")
    print(f"Detected {len(first_audio_by_speaker)} speakers")


if __name__ == "__main__":
    main()
