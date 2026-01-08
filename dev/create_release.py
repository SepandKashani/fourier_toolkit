#!/usr/bin/env python3


import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


KEEP_FILES = [
    ".git",
    ".gitignore",
    ".pre-commit-config.yaml",
]


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    # run a command; fail upon error (unless explicitly silenced)
    print("RUN " + " ".join(cmd))
    kwargs["check"] = kwargs.get("check", True)
    return subprocess.run(args=cmd, **kwargs)


def get_version() -> str:
    ap = argparse.ArgumentParser(
        description="""
Automated release process:

- tag current HEAD as <version>.
- commit generated HTML docs to branch(gh-pages), then tag it as `<version>-doc`.
- build Python package <version>.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "version",
        help="vX[.Y.Z] release version",
        type=str,
    )
    version = ap.parse_args().version.strip()
    assert re.match(r"^v\d+(?:\.\d+){0,2}$", version), (
        f"Expected 'vX.Y.Z', got {version}"
    )
    return version


def clear_directory(folder: Path, whitelist: list[str]):
    for name in os.listdir(folder):
        if name in whitelist:
            continue

        p = folder / name
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink(missing_ok=True)


if __name__ == "__main__":
    repo_root = Path(__file__).parents[1].resolve()

    version = get_version()

    # ensure clean workdir
    status = run(
        ["git", "status", "--porcelain"],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert status.stdout.strip() == "", "Clean working directory required"

    # tag current HEAD as <version>
    status = run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        stdout=subprocess.PIPE,
        text=True,
    )
    orig_branch = status.stdout.strip()
    run(["git", "tag", version])

    # build Python package
    #
    # This is actually a dummy build since it will be deleted when changing branches below.
    # It is required however for the version number to be correctly written in HTML docs.
    run(["uv", "build"], cwd=repo_root)

    # build HTML docs (two-passes to guarantee from-scratch generation)
    for mode in ["clean", "html"]:
        run(
            ["uv", "run", "make", mode],
            cwd=repo_root / "doc",
        )

    # HTML transfer to branch(gh-pages)
    html_dir = repo_root / "doc" / "_build" / "html"
    with tempfile.TemporaryDirectory() as td:
        # keep a temporary copy outside repo
        shutil.copytree(html_dir, td, dirs_exist_ok=True)

        # replace branch content
        run(["git", "checkout", "gh-pages"])
        clear_directory(repo_root, KEEP_FILES)
        shutil.copytree(td, repo_root, dirs_exist_ok=True)

    # force correct HTML rendering on GitHub Pages
    (repo_root / ".nojekyll").touch()

    # tag current HEAD as <version>-doc
    version_doc = f"{version}-doc"
    run(["git", "add", "-A"])
    run(["git", "commit", "-m", f"HTML docs for {version}"])
    run(["git", "tag", version_doc])

    # put user back on original branch
    run(["git", "checkout", orig_branch])

    # (re-)build Python package
    run(["uv", "build"], cwd=repo_root)

    # info msg for user
    dist_dir = repo_root / "dist"
    print(
        "\n".join(
            [
                f"- created git tag {version}",
                f"- created git tag {version_doc}",
                f"- created {version} packages under {dist_dir}",
            ]
        )
    )
