# Developer Guide (AGENTS.md)

This document provides project-specific architecture, build, testing, and debugging guidelines for developers and autonomous agents working on `qwen3-honed-chat-templates`.

---

## 1. Build & Configuration Instructions

### Environment & Toolchain
- **Python**: Requires Python `>= 3.12` (tested with Python 3.12 and 3.14).
- **Package Manager**: [`uv`](https://github.com/astral-sh/uv) is used for project and dependency management (`uv.lock`, `pyproject.toml`).
- **Core Dependencies**:
  - `gguf>=0.19.0`: GGUF format parsing and binary metadata manipulation.
  - `jinja2>=3.1.6`: Chat template rendering, evaluation, and AST inspection.
  - `numpy>=2.5.3`: Mock tensor generation and array comparisons in test fixtures.
  - `pytest`: Automated test runner.

### Initial Setup
Sync the virtual environment using `uv`:
```bash
uv sync
```
Alternatively, with standard `venv` and `pip`:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest
```

### Key Artifacts & Project Layout
- `chat_template.jinja`: The canonical source chat template (Qwen 3.x honed template, current version `qwen3.8-honed-v22.5.1`).
- `install.py`: Main CLI tool applying and uninstalling the honed chat template for Hugging Face model directories and GGUF binary files.
- `scripts/minify_jinja.py`: Template minifier that collapses whitespace and strips comments while protecting multiline literal blocks within `{% set %}...{% endset %}`.
- `scripts/check_applied_files.py`: Diagnostic utility inspecting active templates in directory configs or GGUF metadata.
- `scripts/check_applied_chatting.py`: Behavioral verification tool executing probe prompts against installed model templates.
- `scripts/test_v22.py`: Exhaustive functional test suite (105 test cases) covering template rendering parameters, tool formatting, and reasoning tags.
- `scripts/fuzz_template.py`: Deterministic property fuzzer validating structural invariants across randomized conversations.
- `tests/test_install.py`: Comprehensive test suite verifying CLI operations, directory patching, in-place GGUF metadata updating, and uninstallation.

---

## 2. Testing Information

### Test Runners & Configuration
Tests can be executed via `pytest` or Python's standard `unittest` runner.

#### Running Unit & Integration Tests
```bash
# Run the complete install test suite via pytest
uv run pytest tests/

# Run a specific test method
uv run pytest tests/test_install.py -k test_directory_patch_success_with_existing_template

# Alternatively using unittest
uv run python -m unittest discover tests
```

> **Important (Import Resolution)**: The repository root is not packaged into `site-packages`. When creating new test files under `tests/`, either execute with `PYTHONPATH=.` or ensure `sys.path.insert(0, str(REPO_ROOT))` is executed before importing `install` or `scripts.*`.

#### Running Template Regression & Invariant Tests
```bash
# 1. Generate the minified one-line template expected by template test scripts:
uv run python scripts/minify_jinja.py chat_template.jinja chat_template_oneline.txt

# 2. Run the 105-case template rendering suite:
uv run python scripts/test_v22.py

# 3. Run the property-based invariant fuzzer:
uv run python scripts/fuzz_template.py --cases 100

# 4. Clean up generated template artifact after test run:
rm -f chat_template_oneline.txt
```

### Guidelines for Adding New Tests

1. **Framework & Structure**: Use standard `unittest.TestCase` with `tempfile.TemporaryDirectory` in `setUp()` and cleanup in `tearDown()`.
2. **Synthetic Fixtures**:
   - **GGUF Models**: Avoid committing large binary files. Construct synthetic test models in-memory using `gguf.GGUFWriter` with minimal dummy float arrays for tensor data:
     ```python
     writer = gguf.GGUFWriter(path, arch="qwen2")
     writer.add_chat_template("template content")
     arr = np.array([1.0, 2.0], dtype=np.float32)
     writer.add_tensor("model.layers.0.weight", arr)
     writer.write_header_to_file()
     writer.write_kv_data_to_file()
     writer.write_ti_data_to_file()
     writer.write_tensor_data(arr)
     writer.close()
     ```
   - **Hugging Face Model Directories**: Create a temporary directory containing `tokenizer_config.json` (with a `"chat_template"` key) and optionally `chat_template.jinja`.
3. **Negative & Failure Assertions**:
   - Use subprocess execution helpers asserting non-zero exit codes (`res.returncode != 0`) and matching error strings in `stderr`.
   - Suppress confusing output in negative tests by redirecting stdout/stderr or using helper wrappers.
4. **Behavioral Rendering Probes**:
   - Validate templates against Jinja2 rendering invariants:
     - **Terseness Marker**: `"Never: open with preamble"` must appear exactly once.
     - **Custom System Prompt**: Preserved verbatim when passed (e.g., `"Be a pirate."`).
     - **Thought Retention**: Probe string inside `<think>...</think>` tags must survive across multi-turn exchanges.

### Working Demonstration Test Example

Below is a verified minimal test demonstrating template minification and probe rendering:

```python
#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path
from jinja2 import Environment

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.minify_jinja import minify_jinja

TEMPLATE_PATH = REPO_ROOT / "chat_template.jinja"


class TestChatTemplateDemonstration(unittest.TestCase):
    def setUp(self):
        self.assertTrue(TEMPLATE_PATH.is_file(), "chat_template.jinja must exist at repository root")
        self.source_template = TEMPLATE_PATH.read_text(encoding="utf-8")

    def test_minify_jinja_preserves_terse_marker(self):
        """Verify minify_jinja preserves verbatim terseness instructions inside {% set %} blocks."""
        minified = minify_jinja(self.source_template)
        self.assertIn("Never: open with preamble", minified)
        self.assertNotIn("\t", minified)

    def test_template_probe_rendering(self):
        """Verify chat template retains system prompt and applies terseness marker."""
        env = Environment()
        template = env.from_string(self.source_template)
        messages = [
            {"role": "system", "content": "Be a pirate."},
            {"role": "user", "content": "Hello!"},
        ]
        rendered = template.render(messages=messages, add_generation_prompt=True)
        self.assertIn("Be a pirate.", rendered)
        self.assertIn("Never: open with preamble", rendered)
        self.assertIn("<|im_start|>assistant", rendered)


if __name__ == "__main__":
    unittest.main()
```

---

## 3. Additional Development & Debugging Information

### Architecture & Critical Invariants

1. **Dual-Source Template Precedence (HF Directories)**:
   - Inference runtimes disagree on precedence: Hugging Face `transformers >= 4.51` and LM Studio prioritize `chat_template.jinja`, whereas oMLX and legacy loaders read `tokenizer_config.json["chat_template"]`.
   - `install.py` writes to **both** locations simultaneously: the unminified template to `chat_template.jinja` and the minified template to `tokenizer_config.json`.
   - Both template sources are validated by `verify_directory()` to ensure they produce identical probe outputs.

2. **In-Place GGUF Binary Patching (`gguf_set_metadata`)**:
   - `install.py` does not write full GGUF copies or duplicate tensor weights.
   - It parses existing header and KV metadata, packs the updated `tokenizer.chat_template` (and creates a `tokenizer.chat_template.backup` key), adjusts padding to maintain data alignment, and shifts tensor byte offsets in 16 MB chunks using file seeks.
   - When modifying `gguf_set_metadata`, ensure tensor offsets, endianness (`<` vs `>`), and alignment padding remain strictly intact.

3. **Backup & Uninstallation Contract**:
   - **Directory Mode**:
     - Pre-existing `chat_template.jinja` -> backed up to `chat_template.jinja.bak`.
     - `tokenizer_config.json` -> backed up to `tokenizer_config.json.bak`.
     - `--uninstall` restores original files from `.bak` files and removes them; if `chat_template.jinja` was newly created by `install.py` (no `.bak`), it is deleted.
   - **GGUF Mode**:
     - Pre-existing chat template is preserved inside the GGUF file under `tokenizer.chat_template.backup`.
     - `--uninstall` reads `tokenizer.chat_template.backup`, restores it to `tokenizer.chat_template`, and deletes the backup metadata key.

4. **Jinja Minification Gotcha**:
   - Standard whitespace collapsing (`replace("\n", " ")`) merges instructions into a single run-on paragraph, corrupting system prompts.
   - `scripts/minify_jinja.py` uses a regex stash (`\{%-?\s*set\s+\w+\s*%\}.*?\{%-?\s*endset\s*-?%\}`) to protect literal multiline text inside `_terse` blocks before collapsing whitespace around them.

### Diagnostics & Inspection Utilities
When debugging template issues or user model reports, use the diagnostic scripts:
```bash
# Check detected template versions and detect source mismatches:
uv run python scripts/check_applied_files.py /path/to/model_or_file.gguf

# Render test probe prompts to verify behavior on applied model:
uv run python scripts/check_applied_chatting.py /path/to/model_or_file.gguf
```

### Code Style & Quality Standards
- **Python**: PEP 8 compliance, 4-space indentation, descriptive docstrings on test methods outlining PASS/FAIL criteria.
- **Type Annotations**: Use modern Python 3.12+ syntax (`from __future__ import annotations`, `str | None`, `list[dict]`).
- **Linter**: Qodana configuration is defined in `qodana.yaml` targeting `jetbrains/qodana-python:2026.2`.
- **Cleanup Policy**: Ensure scripts and tests remove temporary `.tmp` or test directories under all exit conditions (`try...finally`).
