#!/usr/bin/env python3
"""Upload one or more local folder trees into a Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import errno
import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from huggingface_hub import HfApi


def fail(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage one or more local folders into a Hugging Face dataset repo and upload them with "
            "upload_large_folder(). Useful for evaluation/test data where the structure is not a single paired subset."
        )
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help=(
            "Dataset repo target. Accepts either `namespace/repo` or `namespace/repo/subset`, "
            "for example `NisargUpadhyay/ImageSuperResolution/Evaluation`."
        ),
    )
    parser.add_argument(
        "--subset",
        help="Optional top-level path inside the dataset repo. Can also be encoded in --repo-id.",
    )
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="LOCAL_PATH=REMOTE_PATH",
        help=(
            "Map a local folder into a relative path under the subset root. "
            "Example: `preset/datasets/test_datasets=test_datasets`. "
            "Use `.` to place the folder contents directly at the subset root."
        ),
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN from the environment.",
    )
    parser.add_argument("--private", action="store_true", help="Create the dataset repo as private.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the upload plan without making network changes.",
    )
    parser.add_argument(
        "--staging-dir",
        default="",
        help=(
            "Optional local staging directory used to build the upload tree. "
            "Defaults to `.hf_upload_staging/<repo>` next to the first mapped source."
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
        "--clear-remote-subset",
        action="store_true",
        help="Delete the remote subset folder before uploading the new staged tree.",
    )
    parser.add_argument(
        "--confirm-clear-subset",
        default="",
        help="Safety check for --clear-remote-subset. Must exactly match the subset name.",
    )
    return parser.parse_args()


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
        fail("Provide the subset either in --repo-id or via --subset, not both.")

    subset = subset_from_arg or subset_from_repo
    if not subset:
        fail("Missing subset. Use --subset Evaluation or pass --repo-id namespace/repo/Evaluation.")

    return repo_id, subset


def parse_mapping(raw_mapping: str) -> tuple[Path, Path]:
    if "=" not in raw_mapping:
        fail(f"Invalid --map value: {raw_mapping}. Expected LOCAL_PATH=REMOTE_PATH.")
    local_raw, remote_raw = raw_mapping.split("=", 1)
    local_path = Path(local_raw).expanduser()
    remote_raw = remote_raw.strip()
    remote_path = Path(".") if remote_raw == "." else Path(remote_raw.strip("/ "))
    if not local_path.exists():
        fail(f"Mapped local path does not exist: {local_path}")
    if not local_path.is_dir():
        fail(f"Mapped local path is not a directory: {local_path}")
    if str(remote_path) == "":
        fail(f"Invalid remote path in --map: {raw_mapping}")
    return local_path, remote_path


def default_staging_dir(base_dir: Path, repo_id: str) -> Path:
    return base_dir / ".hf_upload_staging" / repo_id.replace("/", "__")


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
                "Choose a --staging-dir on the same filesystem as the mapped folders."
            )
        raise


def clear_upload_large_folder_cache(staging_dir: Path, subset: str) -> None:
    upload_cache_root = staging_dir / ".cache" / "huggingface" / "upload"
    subset_cache = upload_cache_root / subset
    if subset_cache.exists():
        shutil.rmtree(subset_cache)


def stage_tree(staging_dir: Path, subset: str, mappings: list[tuple[Path, Path]]) -> tuple[int, int]:
    subset_root = staging_dir / subset
    if subset_root.exists():
        shutil.rmtree(subset_root)
    subset_root.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_dirs = 0
    for local_root, remote_root in mappings:
        for source in sorted(local_root.rglob("*")):
            relative = source.relative_to(local_root)
            destination = subset_root / remote_root / relative
            if source.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                total_dirs += 1
            elif source.is_file():
                hardlink_file(source, destination)
                total_files += 1
    return total_files, total_dirs


def main() -> None:
    args = parse_args()
    if not args.map:
        fail("At least one --map LOCAL_PATH=REMOTE_PATH is required.")
    if not args.dry_run and not args.token:
        fail("No Hugging Face token found. Pass --token or export HF_TOKEN.")

    repo_id, subset = resolve_repo_target(args.repo_id, args.subset)
    mappings = [parse_mapping(raw) for raw in args.map]
    first_source = mappings[0][0]
    staging_dir = (
        Path(args.staging_dir).expanduser()
        if args.staging_dir
        else default_staging_dir(first_source.parent, repo_id)
    )

    if args.clear_remote_subset and args.confirm_clear_subset != subset:
        fail(
            "--clear-remote-subset requires --confirm-clear-subset to exactly match the subset name "
            f"({subset})."
        )

    print("Upload plan:")
    print(f"- repo: {repo_id} (dataset)")
    print(f"- subset path: {subset}/")
    print(f"- upload mode: upload_large_folder")
    print(f"- staging dir: {staging_dir}")
    print(f"- clear remote subset first: {'yes' if args.clear_remote_subset else 'no'}")
    for local_root, remote_root in mappings:
        file_count = sum(1 for path in local_root.rglob("*") if path.is_file())
        target = f"{subset}/" if str(remote_root) == "." else f"{subset}/{remote_root}"
        print(f"- map: {local_root} -> {target} ({file_count} files)")

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

    total_files, total_dirs = stage_tree(staging_dir, subset, mappings)
    print(f"Staged {total_files} files across {total_dirs} directories.")

    api = HfApi(token=args.token)
    if args.clear_remote_subset:
        api.delete_folder(
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=subset,
            commit_message=f"Remove remote subset {subset} before re-upload",
        )
        clear_upload_large_folder_cache(staging_dir, subset)

    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(staging_dir),
        private=args.private,
        num_workers=args.num_workers,
        print_report=True,
        print_report_every=args.report_every,
    )

    print(f"Uploaded staged folder tree to {repo_id} under {subset}/")


if __name__ == "__main__":
    main()
