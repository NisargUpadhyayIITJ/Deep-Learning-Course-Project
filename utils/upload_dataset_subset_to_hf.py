#!/usr/bin/env python3
"""Upload a paired image/text dataset subset to a Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from huggingface_hub import HfApi


@dataclass(frozen=True)
class PairRecord:
    stem: str
    gt_path: Path
    prompt_path: Path
    shard: str


def fail(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload paired GT image and prompt text folders to a Hugging Face dataset repo "
            "under a subset-style path such as `Replication/`."
        )
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help=(
            "Dataset repo target. Accepts either `namespace/repo` or "
            "`namespace/repo/subset`, for example `NisargUpadhyay/ImageSuperResolution/Replication`."
        ),
    )
    parser.add_argument(
        "--subset",
        help="Top-level path inside the dataset repo, for example Replication.",
    )
    parser.add_argument("--gt-dir", required=True, help="Directory containing GT images.")
    parser.add_argument("--prompt-dir", required=True, help="Directory containing prompt text files.")
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN from the environment.",
    )
    parser.add_argument("--private", action="store_true", help="Create the dataset repo as private.")
    parser.add_argument(
        "--license",
        default="",
        help="Optional dataset card license value, for example apache-2.0 or cc-by-4.0.",
    )
    parser.add_argument(
        "--commit-message",
        default="",
        help="Ignored when using upload_large_folder, which creates multiple commits.",
    )
    parser.add_argument(
        "--skip-readme",
        action="store_true",
        help="Do not upload or overwrite the dataset README.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the upload plan without making network changes.",
    )
    parser.add_argument(
        "--staging-dir",
        default="",
        help=(
            "Optional local staging directory used to build a large-folder upload tree. "
            "Defaults to `.hf_upload_staging/<repo>/<subset>` in the current repo."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional number of workers passed to HfApi.upload_large_folder().",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=60,
        help="Seconds between upload_large_folder progress reports. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--max-files-per-dir",
        type=int,
        default=5000,
        help=(
            "Maximum files staged inside each leaf directory. "
            "Defaults to %(default)s to stay comfortably below Hugging Face's 10k folder limit."
        ),
    )
    parser.add_argument(
        "--clear-remote-subset",
        action="store_true",
        help="Delete the remote subset folder before uploading the new sharded version.",
    )
    parser.add_argument(
        "--confirm-clear-subset",
        default="",
        help=(
            "Safety check for --clear-remote-subset. "
            "Must exactly match the resolved subset name, for example `Replication`."
        ),
    )
    return parser.parse_args()


def list_files(directory: Path) -> list[Path]:
    if not directory.exists():
        fail(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        fail(f"Path is not a directory: {directory}")
    return sorted(path for path in directory.iterdir() if path.is_file())


def resolve_repo_target(raw_repo_id: str, raw_subset: str | None) -> tuple[str, str]:
    parts = [part for part in raw_repo_id.strip("/").split("/") if part]
    if len(parts) < 2:
        fail(
            "--repo-id must be either `namespace/repo` or `namespace/repo/subset`, "
            f"got: {raw_repo_id}"
        )

    repo_id = "/".join(parts[:2])
    subset_from_repo = "/".join(parts[2:]).strip("/ ")
    subset_from_arg = (raw_subset or "").strip("/ ")

    if subset_from_repo and subset_from_arg:
        fail(
            "Provide the subset either in --repo-id as `namespace/repo/subset` or via --subset, not both."
        )

    subset = subset_from_arg or subset_from_repo
    if not subset:
        fail("Missing subset. Use --subset Replication or pass --repo-id namespace/repo/Replication.")

    return repo_id, subset


def validate_pairs(gt_files: list[Path], prompt_files: list[Path]) -> list[str]:
    gt_stems = {path.stem for path in gt_files}
    prompt_stems = {path.stem for path in prompt_files}
    only_gt = sorted(gt_stems - prompt_stems)
    only_prompt = sorted(prompt_stems - gt_stems)
    if only_gt or only_prompt:
        message = []
        if only_gt:
            message.append(f"missing prompt for {len(only_gt)} GT files")
        if only_prompt:
            message.append(f"missing GT image for {len(only_prompt)} prompt files")
        fail(", ".join(message))
    return sorted(gt_stems)


def build_pair_records(gt_files: list[Path], prompt_files: list[Path], max_files_per_dir: int) -> list[PairRecord]:
    if max_files_per_dir <= 0:
        fail("--max-files-per-dir must be a positive integer")

    prompt_map = {path.stem: path for path in prompt_files}
    records: list[PairRecord] = []
    for index, gt_path in enumerate(sorted(gt_files, key=lambda path: path.stem)):
        shard = f"{index // max_files_per_dir:05d}"
        records.append(
            PairRecord(
                stem=gt_path.stem,
                gt_path=gt_path,
                prompt_path=prompt_map[gt_path.stem],
                shard=shard,
            )
        )
    return records


def build_manifest(subset: str, pair_records: list[PairRecord]) -> bytes:
    lines = []
    for record in pair_records:
        lines.append(
            json.dumps(
                {
                    "id": record.stem,
                    "image_file": f"{subset}/gt/{record.shard}/{record.gt_path.name}",
                    "prompt_file": f"{subset}/prompt/{record.shard}/{record.prompt_path.name}",
                    "shard": record.shard,
                },
                ensure_ascii=True,
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def build_readme(repo_id: str, subset: str, count: int, args: argparse.Namespace) -> bytes:
    front_matter = ["---", "pretty_name: ISR", "task_categories:", "- image-to-text", "- image-classification"]
    if args.license:
        front_matter.append(f"license: {args.license}")
    front_matter.extend(["---", ""])
    body = [
        f"# ISR Dataset",
        "",
        f"This dataset repo contains the `{subset}` subset uploaded from the local DiT4SR training data.",
        "",
        "## Included paths",
        "",
        f"- `{subset}/gt/<shard>/` contains GT images split across shard directories.",
        f"- `{subset}/prompt/<shard>/` contains paired prompt text files split across shard directories.",
        f"- `{subset}/manifest.jsonl` records the image and prompt file mapping for {count} pairs.",
        "",
        "## Notes",
        "",
        "Repo IDs on Hugging Face are only `namespace/repo`, so the subset name is represented as a folder inside the dataset repo.",
    ]
    return ("\n".join(front_matter + body) + "\n").encode("utf-8")


def default_staging_dir(base_dir: Path, repo_id: str, subset: str) -> Path:
    repo_token = repo_id.replace("/", "__")
    subset_token = subset.replace("/", "__")
    return base_dir / ".hf_upload_staging" / repo_token / subset_token


def hardlink_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError as exc:
        if exc.errno in {errno.EXDEV, errno.EPERM, errno.ENOTSUP}:
            fail(
                "Failed to create hardlinks in the staging directory. "
                "Choose a --staging-dir on the same filesystem as your dataset folders, "
                "for example somewhere under the same dataset root on /DATA2."
            )
        raise


def stage_subset(
    staging_dir: Path,
    subset: str,
    pair_records: list[PairRecord],
    repo_id: str,
    args: argparse.Namespace,
) -> Path:
    subset_root = staging_dir / subset

    if subset_root.exists():
        shutil.rmtree(subset_root)
    subset_root.mkdir(parents=True, exist_ok=True)

    for record in pair_records:
        hardlink_file(record.gt_path, subset_root / "gt" / record.shard / record.gt_path.name)
        hardlink_file(record.prompt_path, subset_root / "prompt" / record.shard / record.prompt_path.name)

    (subset_root / "manifest.jsonl").write_bytes(build_manifest(subset, pair_records))
    if not args.skip_readme:
        staging_dir.mkdir(parents=True, exist_ok=True)
        (staging_dir / "README.md").write_bytes(build_readme(repo_id, subset, len(pair_records), args))

    return staging_dir


def main() -> None:
    args = parse_args()
    if not args.dry_run and not args.token:
        fail("No Hugging Face token found. Pass --token or export HF_TOKEN.")

    repo_id, subset = resolve_repo_target(args.repo_id, args.subset)
    gt_dir = Path(args.gt_dir).expanduser()
    prompt_dir = Path(args.prompt_dir).expanduser()

    gt_files = list_files(gt_dir)
    prompt_files = list_files(prompt_dir)
    pair_ids = validate_pairs(gt_files, prompt_files)
    pair_records = build_pair_records(gt_files, prompt_files, args.max_files_per_dir)
    staging_dir = (
        Path(args.staging_dir).expanduser()
        if args.staging_dir
        else default_staging_dir(gt_dir.parent, repo_id, subset)
    )

    if args.clear_remote_subset and args.confirm_clear_subset != subset:
        fail(
            "--clear-remote-subset requires --confirm-clear-subset to exactly match the subset name "
            f"({subset})."
        )

    print("Upload plan:")
    print(f"- repo: {repo_id} (dataset)")
    print(f"- subset path: {subset}/")
    print(f"- GT images: {len(gt_files)} from {gt_dir}")
    print(f"- prompts: {len(prompt_files)} from {prompt_dir}")
    print(f"- matched pairs: {len(pair_ids)}")
    print(f"- upload destination: {subset}/gt/<shard> and {subset}/prompt/<shard>")
    print(f"- manifest: {subset}/manifest.jsonl")
    print(f"- upload mode: upload_large_folder")
    print(f"- staging dir: {staging_dir}")
    print(f"- shard size: {args.max_files_per_dir}")
    print(f"- shard count: {len({record.shard for record in pair_records})}")
    print(f"- clear remote subset first: {'yes' if args.clear_remote_subset else 'no'}")
    if args.commit_message:
        print("- note: --commit-message is ignored by upload_large_folder")

    if args.dry_run:
        print("Dry run complete. No files were uploaded.")
        return

    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError:
        fail(
            "huggingface_hub is not installed in the active Python environment. "
            "Install the environment from environment.yaml or `pip install huggingface-hub` before uploading."
        )

    staged_root = stage_subset(
        staging_dir=staging_dir,
        subset=subset,
        pair_records=pair_records,
        repo_id=repo_id,
        args=args,
    )

    api = HfApi(token=args.token)
    if args.clear_remote_subset:
        api.delete_folder(
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=subset,
            commit_message=f"Remove remote subset {subset} before re-upload",
        )
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(staged_root),
        private=args.private,
        num_workers=args.num_workers,
        print_report=True,
        print_report_every=args.report_every,
    )

    print(f"Uploaded {len(pair_ids)} pairs to {repo_id} under {subset}/")


if __name__ == "__main__":
    main()
