#!/usr/bin/env python3
"""install.py: Applies chat template to a model directory or GGUF file.

This script updates model chat templates with the one from this repository.

- Directory mode (Hugging Face format):
  - Backs up existing chat_template.jinja (if present) and tokenizer_config.json to .bak files.
  - Copies the full repository chat_template.jinja into the model directory.
  - Generates the minified template using scripts.minify_jinja.minify_jinja.
  - Patches tokenizer_config.json with the minified chat_template key.

- GGUF mode (llama.cpp format):
  - Validates GGUF file format and header magic.
  - Prompts for confirmation unless --force is given.
  - Safely rewrites tokenizer.chat_template metadata using minified template.
  - Preserves architecture, tensors, alignment, and endianness via atomic replacement.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

from jinja2 import Environment
import gguf
from gguf import GGUFValueType, Keys, GGUFWriter
from gguf.scripts.gguf_new_metadata import (
    MetadataDetails,
    copy_with_new_metadata,
    get_field_data,
)

from scripts.minify_jinja import minify_jinja

# Ensure repository root is on sys.path for internal imports
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SOURCE_TEMPLATE_NAME = "chat_template.jinja"
SOURCE_TEMPLATE_PATH = REPO_ROOT / SOURCE_TEMPLATE_NAME

# ANSI colors for diagnostics
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Behavioral rendering probe constants
MARKER = "Never: open with preamble"
THINK_PROBE = "kept-thought-4f2a"
SYSTEM_PROBE = "Be a pirate."


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Apply the chat template to a model directory or GGUF file."
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the model directory (Hugging Face format) or .gguf file",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Bypass confirmation prompt when patching GGUF files",
    )
    return parser.parse_args(argv)


def is_gguf_file(path: Path) -> bool:
    """Check if the given path is a GGUF file."""
    if not path.is_file():
        return False
    if path.suffix.lower() == ".gguf":
        return True
    try:
        with open(path, "rb") as f:
            header = f.read(4)
            return header == b"GGUF"
    except Exception:
        return False


def backup_file(path: Path) -> Path:
    """Create a .bak backup copy of a file."""
    bak_path = path.with_name(f"{path.name}.bak")
    shutil.copy2(path, bak_path)
    return bak_path


def patch_directory(target_dir: Path, source_template_path: Path) -> None:
    """Patch the Hugging Face model directory with a new chat template."""
    tokenizer_config_path = target_dir / "tokenizer_config.json"
    if not tokenizer_config_path.is_file():
        print(
            f"Error: Missing 'tokenizer_config.json' in model directory '{target_dir}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    target_template_path = target_dir / SOURCE_TEMPLATE_NAME

    # 1. Create backups of existing files
    if target_template_path.is_file():
        bak_template = backup_file(target_template_path)
        print(f"Created backup: {bak_template}")

    bak_config = backup_file(tokenizer_config_path)
    print(f"Created backup: {bak_config}")

    # 2. Copy source chat_template.jinja into the target directory
    shutil.copy2(source_template_path, target_template_path)
    print(f"Copied '{source_template_path.name}' to '{target_template_path}'")

    # 3. Read the source template and minify
    template_content = source_template_path.read_text(encoding="utf-8")
    minified_template = minify_jinja(template_content)

    # 4. Patch tokenizer_config.json
    try:
        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
    except Exception as e:
        print(
            f"Error: Failed to parse '{tokenizer_config_path}': {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    config_data["chat_template"] = minified_template

    try:
        with open(tokenizer_config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except Exception as e:
        print(
            f"Error: Failed to write '{tokenizer_config_path}': {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Updated 'chat_template' in '{tokenizer_config_path}'")


def patch_gguf(target_path: Path, minified_template: str, force: bool = False) -> bool | None:
    """Patch GGUF file metadata with a minified chat template.

    Safely updates 'tokenizer.chat_template' using an adjacent temporary file
    and atomic replacement, preserving tensors, architecture, alignment, and endianness.
    """
    if gguf is None or getattr(gguf, "GGUFReader", None) is None:
        print("Error: gguf package is required: pip install gguf", file=sys.stderr)
        sys.exit(1)

    target_path = Path(target_path)

    # 1. Validate GGUF file existence and header magic
    if not target_path.is_file():
        print(f"Error: Target path '{target_path}' is not a regular file.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(target_path, "rb") as f:
            magic = f.read(4)
    except Exception as e:
        print(f"Error: Unable to read file '{target_path}': {e}", file=sys.stderr)
        sys.exit(1)

    if magic != b"GGUF":
        print(
            f"Error: '{target_path}' is not a valid GGUF file (missing GGUF magic header).",
            file=sys.stderr,
        )
        sys.exit(1)

    # 2. Validate GGUF file can be parsed
    try:
        reader = gguf.GGUFReader(target_path, "r")
    except Exception as e:
        print(f"Error: Failed to parse GGUF file '{target_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Confirmation prompt if not forced
    if not force:
        print(
            f"Warning: Modifying '{target_path}' in-place. No backup will be made for GGUF files.",
            file=sys.stderr,
        )
        try:
            response = input("Type 'YES' to proceed: ")
        except EOFError:
            print(
                "Error: Standard input closed without confirmation. Use --force to patch non-interactively.",
                file=sys.stderr,
            )
            del reader
            gc.collect()
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nAborted.", file=sys.stderr)
            del reader
            gc.collect()
            sys.exit(1)

        if response.strip() != "YES":
            print("Aborted: Confirmation 'YES' was not received.", file=sys.stderr)
            del reader
            gc.collect()
            return False

    # 4. Safe rewrite via an adjacent temporary file
    temp_path: Path | None = None
    try:
        temp_fd, temp_path_str = tempfile.mkstemp(
            dir=target_path.parent,
            prefix=f".{target_path.name}.",
            suffix=".tmp",
        )
        os.close(temp_fd)
        temp_path = Path(temp_path_str)

        arch = get_field_data(reader, Keys.General.ARCHITECTURE)
        if isinstance(arch, bytes):
            arch = arch.decode("utf-8")
        elif arch is None:
            arch = ""

        writer = GGUFWriter(temp_path, arch=arch, endianess=reader.endianess)

        alignment = get_field_data(reader, Keys.General.ALIGNMENT)
        if alignment is None:
            alignment = getattr(reader, "alignment", None)
        if isinstance(alignment, (int, str, bytes)):
            writer.data_alignment = int(alignment)

        new_metadata = {
            Keys.Tokenizer.CHAT_TEMPLATE: MetadataDetails(
                GGUFValueType.STRING,
                minified_template,
            )
        }

        copy_with_new_metadata(reader, writer, new_metadata, [])

        # Release reader memmap before atomic replacement
        del reader
        gc.collect()

        assert temp_path is not None
        os.replace(temp_path, target_path)
        temp_path = None
        print(f"Updated 'tokenizer.chat_template' in '{target_path}'")
        return True

    except Exception as e:
        print(f"Error while updating GGUF metadata: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def extract_template_version(content: str) -> str:
    """Extract template_version string from template content."""
    if not content:
        return "None"
    m = re.search(r'template_version\s*=\s*["\']([^"\']+)["\']', content)
    if m:
        return m.group(1)
    if "qwen" in content.lower() or "im_start" in content:
        return "Stock / Unknown Qwen Template"
    return "Unknown"


def render_template(src: str, msgs: list[dict], **kw) -> str:
    """Render a template using Jinja2 with a generation prompt enabled by default."""
    if Environment is None:
        raise RuntimeError("jinja2 package is required: pip install jinja2")
    render_kw = {"add_generation_prompt": True}
    render_kw.update(kw)
    return Environment().from_string(src).render(messages=msgs, **render_kw)


def think_kept(rendered: str) -> bool:
    """Check if the last turn's reasoning survived into this prompt."""
    return any(
        THINK_PROBE in blk
        for blk in re.findall(r"<think>(.*?)</think>", rendered, re.DOTALL)
    )


def render_probe(src: str) -> tuple[str, ...]:
    """Render probe cases across various scenarios to evaluate template behavior."""
    user = [{"role": "user", "content": "hi"}]
    cases = [
        (user, {}),
        ([{"role": "system", "content": SYSTEM_PROBE}] + user, {}),
        (user, {"enable_thinking": False}),
        (user, {"reasoning_effort": "low"}),
        (
            [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": f"<think>{THINK_PROBE}</think>A1"},
                {"role": "user", "content": "Q2"},
            ],
            {},
        ),
    ]
    out = []
    for msgs, kw in cases:
        try:
            out.append(render_template(src, msgs, **kw))
        except Exception as e:
            out.append(f"__ERROR__{type(e).__name__}: {e}")
    return tuple(out)


def verify_source(
    label: str,
    content: str,
    expected_version: str,
    expected_content: str | None = None,
) -> bool:
    """Verify a single template source for the version, exact content, and probe rendering."""
    print(f"\n  [{label}] ({len(content) if content else 0} characters)")
    if not content:
        print(f"    {RED}Error: Template content is empty.{RESET}", file=sys.stderr)
        return False

    # 1. Version check
    ver = extract_template_version(content)
    version_ok = (ver == expected_version)
    ver_color = GREEN if version_ok else RED
    print(f"    Version detected: ........ {ver_color}{ver}{RESET}")
    if not version_ok:
        print(f"      {RED}Mismatch: expected '{expected_version}'{RESET}", file=sys.stderr)

    # 2. Content check
    if expected_content is not None:
        content_ok = (content == expected_content)
        content_color = GREEN if content_ok else RED
        print(f"    Exact content match: ..... {content_color}{'yes' if content_ok else 'NO'}{RESET}")
        if not content_ok:
            print(f"      {RED}Mismatch: applied content does not match expected source content{RESET}", file=sys.stderr)
    else:
        content_ok = True

    # 3. Behavioral rendering probes
    try:
        user = [{"role": "user", "content": "hi"}]
        with_sys = [
            {"role": "system", "content": SYSTEM_PROBE},
            {"role": "user", "content": "hi"},
        ]
        multi = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": f"<think>{THINK_PROBE}</think>A1"},
            {"role": "user", "content": "Q2"},
        ]
        plain = render_template(content, user)
        sysd = render_template(content, with_sys)
        mt = render_template(content, multi)

        n_terse = plain.count(MARKER)
        terse_ok = (n_terse == 1)
        terse_color = GREEN if terse_ok else RED
        print(f"    Terseness prompt: ........ {terse_color}{'yes' if terse_ok else f'NO (found {n_terse}x)'}{RESET}")
        if not terse_ok:
            print(f"      {RED}Expected terseness marker '{MARKER}' exactly once, found {n_terse}x{RESET}", file=sys.stderr)

        sys_ok = (SYSTEM_PROBE in sysd)
        sys_color = GREEN if sys_ok else RED
        print(f"    Keeps system prompt: ..... {sys_color}{'yes' if sys_ok else 'NO'}{RESET}")
        if not sys_ok:
            print(f"      {RED}Expected custom system prompt '{SYSTEM_PROBE}' to be preserved{RESET}", file=sys.stderr)

        think_ok = think_kept(mt)
        think_color = GREEN if think_ok else RED
        print(f"    Retains thinking: ........ {think_color}{'yes' if think_ok else 'NO'}{RESET}")
        if not think_ok:
            print(f"      {RED}Expected multi-turn thinking tags to be retained{RESET}", file=sys.stderr)

        for model_name in ("Nail-35b-a3b", "Dagger-27b"):
            if model_name in plain:
                print(f"    {YELLOW}Warning: names specific model '{model_name}'.{RESET}")

        render_ok = terse_ok and sys_ok and think_ok
    except Exception as e:
        print(f"    {RED}Rendering probes failed: {e}{RESET}", file=sys.stderr)
        render_ok = False

    return version_ok and content_ok and render_ok


def verify_directory(
    target_dir: Path,
    source_template_path: Path | str = SOURCE_TEMPLATE_PATH,
) -> bool:
    """Verify that both chat_template.jinja and tokenizer_config.json were updated properly,
    match the expected version and content, pass behavioral probes, and render identically.
    """
    target_dir = Path(target_dir)
    if not target_dir.is_dir():
        print(f"Error: Target '{target_dir}' is not a directory.", file=sys.stderr)
        return False

    if isinstance(source_template_path, (str, Path)) and Path(source_template_path).is_file():
        source_template_content = Path(source_template_path).read_text(encoding="utf-8")
    elif isinstance(source_template_path, str):
        source_template_content = source_template_path
    else:
        print(f"Error: Invalid source template path: {source_template_path}", file=sys.stderr)
        return False

    expected_version = extract_template_version(source_template_content)
    expected_minified = minify_jinja(source_template_content)

    jinja_file = target_dir / SOURCE_TEMPLATE_NAME
    config_file = target_dir / "tokenizer_config.json"

    if not jinja_file.is_file():
        print(f"{RED}Error: Missing '{SOURCE_TEMPLATE_NAME}' in '{target_dir}'.{RESET}", file=sys.stderr)
        return False

    if not config_file.is_file():
        print(f"{RED}Error: Missing 'tokenizer_config.json' in '{target_dir}'.{RESET}", file=sys.stderr)
        return False

    applied_jinja = jinja_file.read_text(encoding="utf-8")
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config_data = json.load(f)
    except Exception as e:
        print(f"{RED}Error: Failed to parse '{config_file}': {e}{RESET}", file=sys.stderr)
        return False

    applied_config = config_data.get("chat_template")
    if not isinstance(applied_config, str):
        print(f"{RED}Error: Missing or invalid 'chat_template' in '{config_file}'.{RESET}", file=sys.stderr)
        return False

    print(f"\n{BOLD}{CYAN}=== Verifying Directory Chat Templates ==={RESET}")
    print(f"Target directory: {target_dir.resolve()}")

    jinja_ok = verify_source(
        SOURCE_TEMPLATE_NAME,
        applied_jinja,
        expected_version,
        expected_content=source_template_content,
    )
    config_ok = verify_source(
        "tokenizer_config.json",
        applied_config,
        expected_version,
        expected_content=expected_minified,
    )

    try:
        probe_jinja = render_probe(applied_jinja)
        probe_config = render_probe(applied_config)
        same_render = (probe_jinja == probe_config)
    except Exception as e:
        print(f"\n  {RED}Error during equivalence probe rendering: {e}{RESET}", file=sys.stderr)
        same_render = False

    print("\n" + "-" * 50)
    if same_render:
        print(f"  {GREEN}✅ Equivalence check passed: Both sources render identical prompts across all probe cases.{RESET}\n")
    else:
        print(f"  {RED}❌ EQUIVALENCE MISMATCH: chat_template.jinja and tokenizer_config.json render differently!{RESET}\n", file=sys.stderr)

    return jinja_ok and config_ok and same_render


def verify_gguf(
    target_path: Path,
    source_template_path: Path | str = SOURCE_TEMPLATE_PATH,
) -> bool:
    """Verify that GGUF metadata contains an updated chat template, matches the expected version
    and content, and passes behavioral rendering probes.
    """
    if gguf is None or getattr(gguf, "GGUFReader", None) is None:
        print("Error: gguf package is required: pip install gguf", file=sys.stderr)
        return False

    target_path = Path(target_path)
    if not target_path.is_file():
        print(f"Error: Target '{target_path}' is not a regular file.", file=sys.stderr)
        return False

    if isinstance(source_template_path, (str, Path)) and Path(source_template_path).is_file():
        source_template_content = Path(source_template_path).read_text(encoding="utf-8")
    elif isinstance(source_template_path, str):
        source_template_content = source_template_path
    else:
        print(f"Error: Invalid source template path: {source_template_path}", file=sys.stderr)
        return False

    expected_version = extract_template_version(source_template_content)
    expected_minified = minify_jinja(source_template_content)

    reader = None
    try:
        reader = gguf.GGUFReader(target_path, "r")
        field = reader.get_field("tokenizer.chat_template")
        if field is None:
            print(f"{RED}Error: Missing 'tokenizer.chat_template' in '{target_path}'.{RESET}", file=sys.stderr)
            return False
        val = field.contents()
        if isinstance(val, str):
            applied_template = val
        elif isinstance(val, (bytes, bytearray, memoryview)):
            applied_template = bytes(val).decode("utf-8")
        else:
            applied_template = str(val)
    except Exception as e:
        print(f"{RED}Error: Failed to read GGUF file '{target_path}': {e}{RESET}", file=sys.stderr)
        return False
    finally:
        if reader is not None:
            del reader
            gc.collect()

    print(f"\n{BOLD}{CYAN}=== Verifying GGUF Chat Template ==={RESET}")
    print(f"Target file: {target_path.resolve()}")

    ok = verify_source(
        "GGUF tokenizer.chat_template",
        applied_template,
        expected_version,
        expected_content=expected_minified,
    )

    print("\n" + "-" * 50)
    if ok:
        print(f"  {GREEN}✅ GGUF chat template verified successfully.{RESET}\n")
    else:
        print(f"  {RED}❌ GGUF chat template verification failed.{RESET}\n", file=sys.stderr)

    return ok


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        target_path = Path(args.model_path)

        if not SOURCE_TEMPLATE_PATH.is_file():
            print(
                f"Error: Source chat template not found at '{SOURCE_TEMPLATE_PATH}'.",
                file=sys.stderr,
            )
            return 1

        if not target_path.exists():
            print(
                f"Error: Target path does not exist: '{args.model_path}'",
                file=sys.stderr,
            )
            return 1

        if target_path.is_dir():
            patch_directory(target_path, SOURCE_TEMPLATE_PATH)
            if not verify_directory(target_path, SOURCE_TEMPLATE_PATH):
                print(
                    f"{RED}Error: Verification failed for model directory '{target_path}'.{RESET}",
                    file=sys.stderr,
                )
                return 1
            print("Successfully applied chat template to directory.")
            return 0
        elif is_gguf_file(target_path):
            template_content = SOURCE_TEMPLATE_PATH.read_text(encoding="utf-8")
            minified_template = minify_jinja(template_content)
            success = patch_gguf(target_path, minified_template, force=args.force)
            if not success:
                return 0
            if not verify_gguf(target_path, SOURCE_TEMPLATE_PATH):
                print(
                    f"{RED}Error: Verification failed for GGUF file '{target_path}'.{RESET}",
                    file=sys.stderr,
                )
                return 1
            print("Successfully applied chat template to GGUF.")
            return 0
        else:
            print(
                f"Error: Target path '{args.model_path}' is neither a directory nor a GGUF file.",
                file=sys.stderr,
            )
            return 1
    except KeyboardInterrupt:
        print("\nAborted by user.", file=sys.stderr)
        return 1
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    except Exception as e:
        print(f"Error: Unexpected failure: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
