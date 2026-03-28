#!/usr/bin/env python3
"""Upload DiT4SR experiment checkpoints to the Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from huggingface_hub import HfApi


CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


@dataclass(frozen=True)
class UploadSelection:
    experiment_name: str
    experiment_dir: Path
    source_name: str
    source_dir: Path
    step: int | None

    @property
    def transformer_dir(self) -> Path:
        return self.source_dir / "transformer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload DiT4SR experiment checkpoints to Hugging Face. "
            "By default, the latest checkpoint from each experiment is uploaded and only the publishable "
            "`transformer/` files are included."
        )
    )
    parser.add_argument(
        "experiments",
        nargs="*",
        help="Experiment directory names under the experiments root. Defaults to every experiment directory.",
    )
    parser.add_argument(
        "--experiments-root",
        default="experiments",
        help="Path containing experiment directories. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help="Which artifact to upload: latest, final, or an exact directory name like checkpoint-150000.",
    )
    parser.add_argument(
        "--repo-mode",
        choices=("split", "single"),
        default="split",
        help=(
            "`split` creates one Hub repo per experiment. "
            "`single` uploads all selected experiments into one existing-or-new repo."
        ),
    )
    parser.add_argument(
        "--namespace",
        help="Hugging Face user or org name. Required when --repo-mode=split.",
    )
    parser.add_argument(
        "--repo-id",
        help="Target Hub repo in the form user-or-org/name. Required when --repo-mode=single.",
    )
    parser.add_argument(
        "--repo-prefix",
        default="",
        help="Optional prefix added before each experiment name when --repo-mode=split.",
    )
    parser.add_argument(
        "--repo-name",
        action="append",
        default=[],
        metavar="EXPERIMENT=REPO_NAME",
        help=(
            "Override the destination repo name for a specific experiment when --repo-mode=split. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the destination repo(s) as private.",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN from the environment.",
    )
    parser.add_argument(
        "--include-training-state",
        action="store_true",
        help="Upload the full checkpoint directory instead of only transformer weights.",
    )
    parser.add_argument(
        "--skip-readme",
        action="store_true",
        help="Do not upload or overwrite README.md in the destination repo(s).",
    )
    parser.add_argument(
        "--license",
        default="",
        help="Optional Hugging Face model card license field, for example apache-2.0 or other.",
    )
    parser.add_argument(
        "--base-model",
        default="stabilityai/stable-diffusion-3.5-medium",
        help="Optional base model name used in the generated model card.",
    )
    parser.add_argument(
        "--commit-message",
        default="",
        help="Optional commit message. A sensible default is used when omitted.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files and repos would be uploaded without making network changes.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


def checkpoint_step(dirname: str) -> int | None:
    match = CHECKPOINT_RE.match(dirname)
    if match is None:
        return None
    return int(match.group(1))


def parse_repo_name_overrides(raw_overrides: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for raw_value in raw_overrides:
        if "=" not in raw_value:
            fail(f"Invalid --repo-name value: {raw_value}. Expected EXPERIMENT=REPO_NAME.")
        experiment_name, repo_name = raw_value.split("=", 1)
        experiment_name = experiment_name.strip()
        repo_name = repo_name.strip()
        if not experiment_name or not repo_name:
            fail(f"Invalid --repo-name value: {raw_value}. Expected EXPERIMENT=REPO_NAME.")
        overrides[experiment_name] = repo_name
    return overrides


def list_experiment_dirs(experiments_root: Path, requested: Iterable[str]) -> list[Path]:
    if not experiments_root.exists():
        fail(f"Experiments root does not exist: {experiments_root}")
    if not experiments_root.is_dir():
        fail(f"Experiments root is not a directory: {experiments_root}")

    available = {path.name: path for path in experiments_root.iterdir() if path.is_dir()}
    if requested:
        missing = [name for name in requested if name not in available]
        if missing:
            fail(
                "Unknown experiment directories: "
                + ", ".join(missing)
                + f". Available: {', '.join(sorted(available))}"
            )
        return [available[name] for name in requested]

    selected = sorted(available.values(), key=lambda path: path.name)
    if not selected:
        fail(f"No experiment directories were found under {experiments_root}")
    return selected


def pick_source(experiment_dir: Path, checkpoint_arg: str) -> UploadSelection:
    final_model_dir = experiment_dir / "final_model"
    checkpoint_dirs = []
    for child in experiment_dir.iterdir():
        if not child.is_dir():
            continue
        step = checkpoint_step(child.name)
        if step is not None:
            checkpoint_dirs.append((step, child))
    checkpoint_dirs.sort(key=lambda item: item[0])

    selected_dir: Path | None = None
    selected_step: int | None = None

    if checkpoint_arg == "latest":
        if checkpoint_dirs:
            selected_step, selected_dir = checkpoint_dirs[-1]
        elif final_model_dir.is_dir():
            selected_dir = final_model_dir
    elif checkpoint_arg == "final":
        if final_model_dir.is_dir():
            selected_dir = final_model_dir
    else:
        candidate_dir = experiment_dir / checkpoint_arg
        if candidate_dir.is_dir():
            selected_dir = candidate_dir
            selected_step = checkpoint_step(candidate_dir.name)

    if selected_dir is None:
        available = [path.name for _, path in checkpoint_dirs]
        if final_model_dir.is_dir():
            available.append(final_model_dir.name)
        fail(
            f"Could not find a publishable source for {experiment_dir.name} using --checkpoint={checkpoint_arg}. "
            f"Available entries: {', '.join(available) if available else 'none'}"
        )

    transformer_dir = selected_dir / "transformer"
    if not transformer_dir.is_dir():
        fail(f"Expected transformer directory at {transformer_dir}")

    return UploadSelection(
        experiment_name=experiment_dir.name,
        experiment_dir=experiment_dir,
        source_name=selected_dir.name,
        source_dir=selected_dir,
        step=selected_step,
    )


def build_front_matter(args: argparse.Namespace) -> str:
    lines = ["---", "library_name: diffusers", "tags:", "- dit4sr", "- super-resolution", "- diffusion-transformer"]
    if args.base_model:
        lines.append(f"base_model: {args.base_model}")
    if args.license:
        lines.append(f"license: {args.license}")
    lines.extend(["---", ""])
    return "\n".join(lines)


def build_split_readme(repo_id: str, selection: UploadSelection, args: argparse.Namespace) -> str:
    front_matter = build_front_matter(args)
    title = selection.experiment_name.replace("_", " ").replace("-", " ").title()
    return (
        f"{front_matter}"
        f"# {title}\n\n"
        f"This repository contains a published DiT4SR transformer checkpoint exported from the local experiment "
        f"`{selection.experiment_name}` at `{selection.source_name}`.\n\n"
        f"## What This Repo Contains\n\n"
        f"This Hub repo stores only the DiT4SR transformer weights needed by `SD3Transformer2DModel`. "
        f"It does not contain the full SD3 base model, tokenizers, or the rest of the DiT4SR inference pipeline.\n\n"
        f"## Included files\n\n"
        f"- `transformer/` contains the publishable model weights and config.\n"
        f"- `source_checkpoint.json` records the local source path and checkpoint name used for the upload.\n\n"
        f"## Loading In DiT4SR\n\n"
        f"```python\n"
        f"from model_dit4sr.transformer_sd3 import SD3Transformer2DModel\n\n"
        f"model = SD3Transformer2DModel.from_pretrained(\"{repo_id}\", subfolder=\"transformer\")\n"
        f"```\n\n"
        f"You still need the rest of the DiT4SR codebase and the base SD3 assets described in the project README.\n\n"
        f"## Notes\n\n"
        f"- Export source: `{selection.experiment_name}`\n"
        f"- Uploaded checkpoint: `{selection.source_name}`\n"
        f"- Published artifact type: `transformer` weights only\n"
    )


def build_single_readme(repo_id: str, selections: list[UploadSelection], args: argparse.Namespace) -> str:
    front_matter = build_front_matter(args)
    lines = [
        f"{front_matter}# DiT4SR Experiment Exports",
        "",
        f"This repository aggregates selected DiT4SR transformer exports from the local `{args.experiments_root}` directory.",
        "",
        "## Included checkpoints",
        "",
    ]
    for selection in selections:
        lines.append(
            f"- `{selection.experiment_name}` from `{selection.source_name}` "
            f"-> load with `subfolder=\"{selection.experiment_name}/{selection.source_name}/transformer\"`"
        )
    lines.extend(
        [
            "",
            "## Loading",
            "",
            "```python",
            "from model_dit4sr.transformer_sd3 import SD3Transformer2DModel",
            "",
            f"model = SD3Transformer2DModel.from_pretrained(",
            f"    \"{repo_id}\",",
            f"    subfolder=\"{selections[0].experiment_name}/{selections[0].source_name}/transformer\",",
            ")",
            "```",
            "",
            "Swap the `subfolder` value for whichever exported experiment you want to load.",
            "",
            "Each exported checkpoint contains only the publishable `transformer/` weights, not the full base pipeline.",
        ]
    )
    return "\n".join(lines)


def build_metadata(selection: UploadSelection, args: argparse.Namespace) -> bytes:
    payload = {
        "experiment_name": selection.experiment_name,
        "experiment_dir": str(selection.experiment_dir.resolve()),
        "source_name": selection.source_name,
        "source_dir": str(selection.source_dir.resolve()),
        "checkpoint_step": selection.step,
        "uploaded_only_transformer": not args.include_training_state,
    }
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def upload_text_file(
    api: HfApi,
    repo_id: str,
    path_in_repo: str,
    content: bytes,
    commit_message: str,
) -> None:
    api.upload_file(
        path_or_fileobj=content,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )


def ensure_repo(api: HfApi, repo_id: str, private: bool) -> None:
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)


def upload_selection(
    api: HfApi,
    repo_id: str,
    selection: UploadSelection,
    args: argparse.Namespace,
    path_in_repo: str,
) -> None:
    allow_patterns = None if args.include_training_state else ["transformer/*"]
    commit_message = args.commit_message or f"Upload {selection.experiment_name} from {selection.source_name}"
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(selection.source_dir),
        path_in_repo=path_in_repo,
        allow_patterns=allow_patterns,
        commit_message=commit_message,
    )


def main() -> None:
    args = parse_args()

    if args.repo_mode == "split" and not args.namespace:
        fail("--namespace is required when --repo-mode=split")
    if args.repo_mode == "single" and not args.repo_id:
        fail("--repo-id is required when --repo-mode=single")
    if not args.dry_run and not args.token:
        fail("No Hugging Face token found. Pass --token or export HF_TOKEN.")

    experiments_root = Path(args.experiments_root).expanduser()
    experiment_dirs = list_experiment_dirs(experiments_root, args.experiments)
    selections = [pick_source(path, args.checkpoint) for path in experiment_dirs]
    repo_name_overrides = parse_repo_name_overrides(args.repo_name)

    print("Upload plan:")
    for selection in selections:
        publish_mode = "full checkpoint" if args.include_training_state else "transformer only"
        print(
            f"- {selection.experiment_name}: {selection.source_name} "
            f"({publish_mode}) from {selection.source_dir}"
        )

    if args.repo_mode == "split":
        for selection in selections:
            repo_name = repo_name_overrides.get(selection.experiment_name, f"{args.repo_prefix}{selection.experiment_name}")
            repo_id = f"{args.namespace}/{repo_name}"
            print(f"  -> repo {repo_id}")
    else:
        print(f"  -> repo {args.repo_id}")

    if args.dry_run:
        print("Dry run complete. No files were uploaded.")
        return

    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:
        fail(
            "huggingface_hub is not installed in the active Python environment. "
            "Install the environment from environment.yaml or `pip install huggingface-hub` before uploading."
        )

    api = HfApi(token=args.token)

    if args.repo_mode == "split":
        for selection in selections:
            repo_name = repo_name_overrides.get(selection.experiment_name, f"{args.repo_prefix}{selection.experiment_name}")
            repo_id = f"{args.namespace}/{repo_name}"
            ensure_repo(api, repo_id=repo_id, private=args.private)
            upload_selection(api, repo_id=repo_id, selection=selection, args=args, path_in_repo="")
            upload_text_file(
                api=api,
                repo_id=repo_id,
                path_in_repo="source_checkpoint.json",
                content=build_metadata(selection, args),
                commit_message=args.commit_message or f"Add metadata for {selection.experiment_name}",
            )
            if not args.skip_readme:
                upload_text_file(
                    api=api,
                    repo_id=repo_id,
                    path_in_repo="README.md",
                    content=build_split_readme(repo_id, selection, args).encode("utf-8"),
                    commit_message=args.commit_message or f"Add model card for {selection.experiment_name}",
                )
            print(f"Uploaded {selection.experiment_name} to {repo_id}")
        return

    ensure_repo(api, repo_id=args.repo_id, private=args.private)
    for selection in selections:
        repo_path = f"{selection.experiment_name}/{selection.source_name}"
        upload_selection(api, repo_id=args.repo_id, selection=selection, args=args, path_in_repo=repo_path)
        upload_text_file(
            api=api,
            repo_id=args.repo_id,
            path_in_repo=f"{repo_path}/source_checkpoint.json",
            content=build_metadata(selection, args),
            commit_message=args.commit_message or f"Add metadata for {selection.experiment_name}",
        )
        print(f"Uploaded {selection.experiment_name} to {args.repo_id}:{repo_path}")

    if not args.skip_readme:
        upload_text_file(
            api=api,
            repo_id=args.repo_id,
            path_in_repo="README.md",
            content=build_single_readme(args.repo_id, selections, args).encode("utf-8"),
            commit_message=args.commit_message or "Add model card",
        )


if __name__ == "__main__":
    main()
