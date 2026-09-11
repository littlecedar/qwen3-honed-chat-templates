#!/usr/bin/env python3
"""Tests for install.py (Step 1 functionality)."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gguf
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
APPLY_SCRIPT = REPO_ROOT / "install.py"
SOURCE_TEMPLATE = REPO_ROOT / "chat_template.jinja"

sys.path.insert(0, str(REPO_ROOT))
import install
from scripts.minify_jinja import minify_jinja


@contextlib.contextmanager
def assert_expected_failure(
    description: str,
    expected_stderr_pattern: str | None = None,
):
    """Capture stdout/stderr for negative tests to prevent misleading error outputs
    and assert that the operation under test failed as expected.
    """
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    exception_caught = None
    try:
        with (
            contextlib.redirect_stdout(captured_stdout),
            contextlib.redirect_stderr(captured_stderr),
        ):
            yield captured_stdout, captured_stderr
    except (AssertionError, unittest.TestCase.failureException):
        raise
    except (Exception, SystemExit) as exc:
        exception_caught = exc

    combined_stderr = captured_stderr.getvalue()
    if exception_caught is not None:
        combined_stderr += " " + str(exception_caught)

    if expected_stderr_pattern:
        if expected_stderr_pattern.lower() not in combined_stderr.lower():
            raise AssertionError(
                f"Expected error pattern '{expected_stderr_pattern}' not found in stderr for '{description}'. "
                f"Captured stderr:\n{combined_stderr}"
            )
    print(f"[PASS - Expected Failure] {description}")


def check_expected_verification_failure(
    test_case: unittest.TestCase,
    verify_fn,
    *args,
    description: str,
    **kwargs,
) -> None:
    """Execute a verification function expected to fail, suppress deceptive ❌ logs,
    and assert that it returned False (double-negative: failure is a pass).
    """
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        result = verify_fn(*args, **kwargs)
    test_case.assertFalse(
        result,
        msg=f"Expected failure for '{description}', but verification unexpectedly succeeded.",
    )
    print(f"[PASS - Expected Failure] {description}")


class TestApplyChatTemplate(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        self.hf_cache_dir = self.test_dir / "hf_cache"
        self.hf_cache_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_script(
        self,
        *args,
        input: str | None = None,
        env: dict[str, str] | None = None,
    ):
        cmd = [sys.executable, str(APPLY_SCRIPT)] + list(args)
        proc_env = os.environ.copy()
        if env:
            proc_env.update(env)
        return subprocess.run(
            cmd, input=input, capture_output=True, text=True, env=proc_env
        )

    def assert_subprocess_failure(
        self,
        res: subprocess.CompletedProcess,
        description: str,
        expected_stderr: str | None = None,
        expected_exit_code: int | None = None,
    ):
        """Assert that a subprocess command failed as expected and record a clear pass."""
        if expected_exit_code is not None:
            self.assertEqual(
                res.returncode,
                expected_exit_code,
                msg=f"Expected exit code {expected_exit_code} for '{description}', got {res.returncode}. stderr: {res.stderr}",
            )
        else:
            self.assertNotEqual(
                res.returncode,
                0,
                msg=f"Expected non-zero exit code for '{description}', got {res.returncode}. stdout: {res.stdout}",
            )
        if expected_stderr:
            self.assertIn(
                expected_stderr.lower(),
                res.stderr.lower(),
                msg=f"Expected pattern '{expected_stderr}' in stderr for '{description}'. Actual stderr: {res.stderr}",
            )
        print(f"[PASS - Expected Failure] {description}")

    def create_synthetic_gguf(
        self,
        filename: str = "model.gguf",
        template: str = "old jinja template",
        alignment: int = 64,
        target_dir: Path | None = None,
    ):
        base_dir = target_dir if target_dir is not None else self.test_dir
        path = base_dir / filename
        writer = gguf.GGUFWriter(path, arch="qwen2")
        writer.add_chat_template(template)
        writer.data_alignment = alignment
        writer.add_uint32(gguf.Keys.General.ALIGNMENT, alignment)
        arr = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        writer.add_tensor("model.layers.0.weight", arr)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        writer.write_tensor_data(arr)
        writer.close()
        return path, arr

    def create_synthetic_hf_repo(
        self,
        repo_id: str,
        revisions: list[dict],
        cache_dir: Path | None = None,
    ) -> Path:
        """Create a synthetic Hugging Face cache directory structure for a model repository."""
        if cache_dir is None:
            cache_dir = self.hf_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        repo_folder = "models--" + repo_id.replace("/", "--")
        repo_dir = cache_dir / repo_folder
        repo_dir.mkdir(parents=True, exist_ok=True)

        for rev in revisions:
            commit_hash = rev["commit_hash"]
            snap_dir = repo_dir / "snapshots" / commit_hash
            snap_dir.mkdir(parents=True, exist_ok=True)
            files = rev.get(
                "files",
                {
                    "tokenizer_config.json": json.dumps(
                        {"chat_template": "old_template_content"}
                    )
                },
            )
            mtime = rev.get("last_modified", 1000.0)
            if files:
                for fname, fcontent in files.items():
                    fpath = snap_dir / fname
                    fpath.parent.mkdir(parents=True, exist_ok=True)
                    if isinstance(fcontent, bytes):
                        fpath.write_bytes(fcontent)
                    else:
                        fpath.write_text(fcontent, encoding="utf-8")
                    os.utime(fpath, (mtime, mtime))
            os.utime(snap_dir, (mtime, mtime))

            refs = rev.get("refs", [])
            if refs:
                refs_dir = repo_dir / "refs"
                refs_dir.mkdir(parents=True, exist_ok=True)
                for ref_name in refs:
                    ref_file = refs_dir / ref_name
                    ref_file.write_text(commit_hash, encoding="utf-8")

        return repo_dir

    def test_cli_help(self):
        """PASS: CLI prints usage documentation when invoked with --help; FAIL: Non-zero exit code or missing required flags."""
        res = self.run_script("--help")
        self.assertEqual(
            res.returncode,
            0,
            msg=f"CLI --help exited with non-zero code {res.returncode}. stderr: {res.stderr}",
        )
        self.assertIn(
            "model_path",
            res.stdout,
            msg="CLI --help output does not describe the 'model_path' positional argument.",
        )
        self.assertIn(
            "--force",
            res.stdout,
            msg="CLI --help output does not describe the '--force' flag.",
        )
        self.assertIn(
            "--uninstall",
            res.stdout,
            msg="CLI --help output does not describe the '--uninstall' flag.",
        )
        self.assertIn(
            "--latest",
            res.stdout,
            msg="CLI --help output does not describe the '--latest' flag.",
        )

    def test_missing_model_path_argument(self):
        """PASS: CLI exits with error code when no model path argument is passed; FAIL: Accepts invocation without arguments."""
        res = self.run_script()
        self.assert_subprocess_failure(
            res,
            description="CLI invocation without model_path argument",
            expected_stderr="error",
        )

    def test_target_does_not_exist(self):
        """PASS: CLI rejects non-existent model path; FAIL: Accepts non-existent path without error."""
        non_existent = self.test_dir / "does_not_exist"
        res = self.run_script(str(non_existent))
        self.assert_subprocess_failure(
            res,
            description="Model path target does not exist",
            expected_stderr="does not exist",
        )

    def test_invalid_target_type(self):
        """PASS: CLI rejects files that are neither directories nor GGUF models; FAIL: Accepts unsupported plain file without error."""
        plain_file = self.test_dir / "sample.txt"
        plain_file.write_text("just text")
        res = self.run_script(str(plain_file))
        self.assert_subprocess_failure(
            res,
            description="Target is neither directory nor GGUF file",
            expected_stderr="neither a directory nor a GGUF file",
        )

    def test_directory_missing_tokenizer_config(self):
        """PASS: CLI rejects model directory missing tokenizer_config.json; FAIL: Accepts directory lacking required config."""
        model_dir = self.test_dir / "model_missing_cfg"
        model_dir.mkdir()
        (model_dir / "chat_template.jinja").write_text("old template")

        res = self.run_script(str(model_dir))
        self.assert_subprocess_failure(
            res,
            description="Model directory missing tokenizer_config.json",
            expected_stderr="Missing 'tokenizer_config.json'",
        )

    def test_directory_patch_success_with_existing_template(self):
        """PASS: Successfully patches directory with existing template and creates backups; FAIL: Fails to patch, update config, or create backups."""
        model_dir = self.test_dir / "model_complete"
        model_dir.mkdir()

        orig_template = "old jinja template content"
        (model_dir / "chat_template.jinja").write_text(orig_template)

        initial_config = {
            "bos_token": "<|im_start|>",
            "eos_token": "<|im_end|>",
            "chat_template": orig_template,
        }
        config_path = model_dir / "tokenizer_config.json"
        config_path.write_text(json.dumps(initial_config, indent=2) + "\n")

        res = self.run_script(str(model_dir))
        self.assertEqual(
            res.returncode,
            0,
            msg=f"Script failed with code {res.returncode}: {res.stderr}",
        )
        self.assertIn(
            "Successfully applied chat template to directory.",
            res.stdout,
            msg="Expected success message not found in stdout.",
        )

        # Check backups created
        bak_template = model_dir / "chat_template.jinja.bak"
        bak_config = model_dir / "tokenizer_config.json.bak"
        self.assertTrue(
            bak_template.is_file(), msg="chat_template.jinja.bak was not created."
        )
        self.assertTrue(
            bak_config.is_file(), msg="tokenizer_config.json.bak was not created."
        )
        self.assertEqual(
            bak_template.read_text(),
            orig_template,
            msg="chat_template.jinja.bak content mismatch with original template.",
        )
        self.assertEqual(
            json.loads(bak_config.read_text()),
            initial_config,
            msg="tokenizer_config.json.bak content mismatch with initial config.",
        )

        # Check chat_template.jinja updated
        expected_template = SOURCE_TEMPLATE.read_text(encoding="utf-8")
        self.assertEqual(
            (model_dir / "chat_template.jinja").read_text(encoding="utf-8"),
            expected_template,
            msg="chat_template.jinja was not updated to repository template.",
        )

        # Check tokenizer_config.json updated and formatted
        updated_config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(
            updated_config["bos_token"],
            "<|im_start|>",
            msg="bos_token was corrupted or missing.",
        )
        self.assertEqual(
            updated_config["eos_token"],
            "<|im_end|>",
            msg="eos_token was corrupted or missing.",
        )
        expected_minified = minify_jinja(expected_template)
        self.assertEqual(
            updated_config["chat_template"],
            expected_minified,
            msg="chat_template in tokenizer_config.json does not match minified template.",
        )

        # Ensure valid JSON formatting with clean indent
        raw_config = config_path.read_text(encoding="utf-8")
        self.assertTrue(
            raw_config.endswith("\n"),
            msg="tokenizer_config.json does not end with trailing newline.",
        )

    def test_directory_patch_success_without_prior_jinja(self):
        """PASS: Successfully patches directory lacking prior chat_template.jinja; FAIL: Fails to create jinja file or update config."""
        model_dir = self.test_dir / "model_no_jinja"
        model_dir.mkdir()

        initial_config = {"model_max_length": 8192}
        config_path = model_dir / "tokenizer_config.json"
        config_path.write_text(json.dumps(initial_config, indent=2) + "\n")

        res = self.run_script(str(model_dir))
        self.assertEqual(
            res.returncode,
            0,
            msg=f"Script failed with code {res.returncode}: {res.stderr}",
        )

        # Config backup exists, but template backup does not
        self.assertTrue(
            (model_dir / "tokenizer_config.json.bak").is_file(),
            msg="tokenizer_config.json.bak was not created.",
        )
        self.assertFalse(
            (model_dir / "chat_template.jinja.bak").exists(),
            msg="chat_template.jinja.bak created unexpectedly when no prior jinja file existed.",
        )

        # chat_template.jinja created
        expected_template = SOURCE_TEMPLATE.read_text(encoding="utf-8")
        self.assertEqual(
            (model_dir / "chat_template.jinja").read_text(encoding="utf-8"),
            expected_template,
            msg="chat_template.jinja was not created with expected repository template content.",
        )

        # tokenizer_config updated
        updated_config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(
            updated_config["model_max_length"],
            8192,
            msg="Existing config properties were not preserved.",
        )
        self.assertEqual(
            updated_config["chat_template"],
            minify_jinja(expected_template),
            msg="chat_template in tokenizer_config.json does not match minified template.",
        )

    def test_corrupted_tokenizer_config(self):
        """PASS: CLI rejects directory containing malformed JSON in tokenizer_config.json; FAIL: Accepts corrupted JSON without error."""
        model_dir = self.test_dir / "model_corrupt"
        model_dir.mkdir()
        config_path = model_dir / "tokenizer_config.json"
        config_path.write_text("{not valid json")

        res = self.run_script(str(model_dir))
        self.assert_subprocess_failure(
            res,
            description="Corrupted tokenizer_config.json rejected",
            expected_stderr="Failed to parse",
        )

    def test_gguf_invalid_magic_header(self):
        """PASS: CLI rejects file with .gguf extension but lacking GGUF magic bytes; FAIL: Accepts invalid GGUF binary."""
        # File with .gguf extension but missing GGUF magic header
        gguf_ext_file = self.test_dir / "dummy.gguf"
        gguf_ext_file.write_bytes(b"some content")
        res = self.run_script(str(gguf_ext_file))
        self.assert_subprocess_failure(
            res,
            description="Invalid magic header file with .gguf extension rejected",
            expected_stderr="not a valid GGUF file",
        )

    def test_gguf_truncated_header(self):
        """PASS: CLI rejects corrupted GGUF binary with truncated header; FAIL: Accepts truncated GGUF file without error."""
        # File with GGUF magic header without .gguf extension but corrupted/truncated data
        gguf_magic_file = self.test_dir / "dummy_bin"
        gguf_magic_file.write_bytes(b"GGUF\x03\x00\x00\x00")
        res = self.run_script(str(gguf_magic_file))
        self.assert_subprocess_failure(
            res,
            description="Truncated GGUF header file rejected",
            expected_stderr="Failed to parse GGUF file",
        )

    def test_gguf_patch_with_force(self):
        """PASS: Successfully patches GGUF chat template metadata in non-interactive mode with --force; FAIL: Metadata not updated or tensors corrupted."""
        gguf_path, orig_arr = self.create_synthetic_gguf(
            "test_model.gguf", template="initial template", alignment=64
        )

        res = self.run_script(str(gguf_path), "--force")
        self.assertEqual(
            res.returncode,
            0,
            msg=f"Script failed with code {res.returncode}: {res.stderr}",
        )
        self.assertIn(
            "Updated 'tokenizer.chat_template'",
            res.stdout,
            msg="stdout missing update message for tokenizer.chat_template.",
        )
        self.assertIn(
            "Successfully applied chat template to GGUF.",
            res.stdout,
            msg="stdout missing success message for GGUF patch.",
        )

        # Inspect updated GGUF file
        reader = gguf.GGUFReader(gguf_path, "r")
        chat_template_field = reader.get_field("tokenizer.chat_template")
        self.assertIsNotNone(
            chat_template_field,
            msg="tokenizer.chat_template field not found in patched GGUF.",
        )

        expected_template = SOURCE_TEMPLATE.read_text(encoding="utf-8")
        expected_minified = minify_jinja(expected_template)
        self.assertEqual(
            chat_template_field.contents(),
            expected_minified,
            msg="Patched GGUF template does not match minified repository template.",
        )

        # Verify architecture and tensor preservation
        arch_field = reader.get_field("general.architecture")
        self.assertIsNotNone(
            arch_field, msg="general.architecture field missing from patched GGUF."
        )
        self.assertEqual(
            arch_field.contents(),
            "qwen2",
            msg="Architecture was not preserved in patched GGUF.",
        )

        self.assertEqual(
            len(reader.tensors),
            1,
            msg=f"Expected 1 tensor in patched GGUF, found {len(reader.tensors)}.",
        )
        tensor = reader.get_tensor(0)
        self.assertEqual(
            tensor.name,
            "model.layers.0.weight",
            msg="Tensor name was altered during GGUF metadata patching.",
        )
        self.assertTrue(
            np.array_equal(tensor.data, orig_arr),
            msg="Tensor weights were altered during GGUF metadata patching.",
        )

        # Verify alignment preservation
        self.assertEqual(
            reader.alignment,
            64,
            msg=f"GGUF alignment was changed from 64 to {reader.alignment}.",
        )

        del reader
        # Verify no temporary files remain
        tmp_files = list(self.test_dir.glob("*.tmp"))
        self.assertEqual(
            tmp_files,
            [],
            msg=f"Temporary files were left behind in test directory: {tmp_files}.",
        )

    def test_gguf_interactive_confirmation_yes(self):
        """PASS: Interactive GGUF patch succeeds when user confirms with YES; FAIL: Fails to patch on confirmation or template unchanged."""
        gguf_path, _ = self.create_synthetic_gguf(
            "confirm_yes.gguf", template="before patch"
        )

        res = self.run_script(str(gguf_path), input="YES\n")
        self.assertEqual(
            res.returncode,
            0,
            msg=f"Script failed with code {res.returncode}: {res.stderr}",
        )
        self.assertIn(
            "Successfully applied chat template to GGUF.",
            res.stdout,
            msg="stdout missing GGUF patch success message.",
        )

        reader = gguf.GGUFReader(gguf_path, "r")
        expected_minified = minify_jinja(SOURCE_TEMPLATE.read_text(encoding="utf-8"))
        self.assertEqual(
            reader.get_field("tokenizer.chat_template").contents(),
            expected_minified,
            msg="GGUF chat template was not updated after interactive confirmation.",
        )
        del reader

    def test_gguf_interactive_confirmation_no(self):
        """PASS: Interactive GGUF patch aborts cleanly without altering file when user responds no; FAIL: Modifies file or omits abort notice."""
        gguf_path, _ = self.create_synthetic_gguf(
            "confirm_no.gguf", template="before patch"
        )

        res = self.run_script(str(gguf_path), input="no\n")
        self.assertEqual(
            res.returncode,
            0,
            msg=f"Expected exit code 0 when user declines confirmation, got {res.returncode}. stderr: {res.stderr}",
        )
        self.assertIn(
            "Aborted: Confirmation 'YES' was not received.",
            res.stderr,
            msg="Expected abort message in stderr when confirmation is denied.",
        )

        # Content must remain unchanged
        reader = gguf.GGUFReader(gguf_path, "r")
        self.assertEqual(
            reader.get_field("tokenizer.chat_template").contents(),
            "before patch",
            msg="GGUF template was modified despite user aborting confirmation.",
        )
        del reader
        print(
            "  [PASS - Expected Failure] Confirmation declined by user; GGUF patch aborted"
        )

    def test_gguf_non_interactive_without_force_fails(self):
        """PASS: Rejects unconfirmed GGUF patch when stdin is closed and --force is not provided; FAIL: Modifies GGUF file without confirmation."""
        gguf_path, _ = self.create_synthetic_gguf(
            "confirm_eof.gguf", template="before patch"
        )

        # Stdin closed / empty simulates non-interactive run without --force
        res = self.run_script(str(gguf_path), input="")
        self.assert_subprocess_failure(
            res,
            description="Non-interactive GGUF patch without --force flag rejected",
            expected_stderr="Standard input closed without confirmation. Use --force",
        )

        # Content must remain unchanged
        reader = gguf.GGUFReader(gguf_path, "r")
        self.assertEqual(
            reader.get_field("tokenizer.chat_template").contents(),
            "before patch",
            msg="GGUF template was modified despite missing confirmation.",
        )
        del reader

    def test_gguf_without_extension_magic_detection_and_patch(self):
        """PASS: Detects and patches GGUF file based on magic header bytes when .gguf extension is absent; FAIL: Rejects valid GGUF file lacking extension."""
        gguf_path, orig_arr = self.create_synthetic_gguf(
            "model_raw_bin", template="raw"
        )

        res = self.run_script(str(gguf_path), "--force")
        self.assertEqual(
            res.returncode, 0, msg=f"Script failed on extensionless GGUF: {res.stderr}"
        )

        reader = gguf.GGUFReader(gguf_path, "r")
        expected_minified = minify_jinja(SOURCE_TEMPLATE.read_text(encoding="utf-8"))
        self.assertEqual(
            reader.get_field("tokenizer.chat_template").contents(),
            expected_minified,
            msg="GGUF chat template was not updated on extensionless file.",
        )
        tensor = reader.get_tensor(0)
        self.assertTrue(
            np.array_equal(tensor.data, orig_arr),
            msg="Tensor data corrupted on extensionless GGUF file.",
        )
        del reader

    def test_gguf_metadata_update_error_exits(self):
        """PASS: Exits with an error when in-place GGUF metadata update fails; FAIL: Hides the update error or continues as if patching succeeded."""
        gguf_path, _ = self.create_synthetic_gguf("fail_write.gguf", template="init")

        def failing_set_metadata(*args, **kwargs):
            raise RuntimeError("Simulated failure during in-place metadata update")

        original_set_metadata = install.gguf_set_metadata
        self.addCleanup(setattr, install, "gguf_set_metadata", original_set_metadata)
        install.gguf_set_metadata = failing_set_metadata

        with assert_expected_failure(
            "GGUF metadata update failure exits cleanly",
            expected_stderr_pattern="Simulated failure during in-place metadata update",
        ):
            with self.assertRaises(SystemExit) as ctx:
                install.patch_gguf(gguf_path, "new template", force=True)
            self.assertEqual(
                ctx.exception.code,
                1,
                msg=f"Expected exit code 1 on metadata update error, got {ctx.exception.code}",
            )

        reader = gguf.GGUFReader(gguf_path, "r")
        self.assertEqual(
            reader.get_field("tokenizer.chat_template").contents(),
            "init",
            msg="GGUF template changed despite metadata update failure.",
        )
        del reader

    def test_gguf_set_metadata_inplace(self):
        """PASS: gguf_set_metadata modifies metadata in-place preserving tensor weights, alignment, and offsets; FAIL: Corrupts tensors or fails to update metadata."""
        gguf_path, orig_arr = self.create_synthetic_gguf(
            "inplace_test.gguf", template="short template", alignment=64
        )

        # 1. Header growth test (expanding metadata)
        large_template = "expanded template string " * 50
        res = install.gguf_set_metadata(
            gguf_path, "tokenizer.chat_template", large_template
        )
        self.assertTrue(res, msg="gguf_set_metadata returned False on expansion.")

        reader = gguf.GGUFReader(gguf_path, "r")
        self.assertEqual(
            reader.get_field("tokenizer.chat_template").contents(),
            large_template,
            msg="Expanded chat template was not properly saved.",
        )
        self.assertEqual(reader.alignment, 64)
        tensor = reader.get_tensor(0)
        self.assertTrue(
            np.array_equal(tensor.data, orig_arr),
            msg="Tensor weights corrupted during expansion.",
        )
        del reader

        # 2. Header shrinkage test (contracting metadata)
        short_template = "tiny"
        res = install.gguf_set_metadata(
            gguf_path, "tokenizer.chat_template", short_template
        )
        self.assertTrue(res, msg="gguf_set_metadata returned False on shrinkage.")

        reader = gguf.GGUFReader(gguf_path, "r")
        self.assertEqual(
            reader.get_field("tokenizer.chat_template").contents(),
            short_template,
            msg="Shrunk chat template was not properly saved.",
        )
        self.assertEqual(reader.alignment, 64)
        tensor = reader.get_tensor(0)
        self.assertTrue(
            np.array_equal(tensor.data, orig_arr),
            msg="Tensor weights corrupted during shrinkage.",
        )
        del reader

        # 3. Add new key and remove key test
        res = install.gguf_set_metadata(
            gguf_path,
            {"custom.test_key": "custom_val"},
            removals=["tokenizer.chat_template"],
        )
        self.assertTrue(res, msg="gguf_set_metadata returned False on add/removal.")

        reader = gguf.GGUFReader(gguf_path, "r")
        self.assertIsNone(
            reader.get_field("tokenizer.chat_template"),
            msg="Removed key was still found.",
        )
        self.assertEqual(
            reader.get_field("custom.test_key").contents(),
            "custom_val",
            msg="Newly added key was not found or has incorrect value.",
        )
        tensor = reader.get_tensor(0)
        self.assertTrue(
            np.array_equal(tensor.data, orig_arr),
            msg="Tensor weights corrupted during add/removal.",
        )
        del reader

    def test_gguf_patch_creates_backup_key(self):
        """PASS: patch_gguf backs up the original template to tokenizer.chat_template.backup; FAIL: Backup key missing or template not updated."""
        gguf_path, orig_arr = self.create_synthetic_gguf(
            "backup_test.gguf", template="original pre-patch template", alignment=32
        )

        res = install.patch_gguf(gguf_path, "new honed template", force=True)
        self.assertTrue(res, msg="patch_gguf failed with force=True.")

        reader = gguf.GGUFReader(gguf_path, "r")
        self.assertEqual(
            reader.get_field("tokenizer.chat_template").contents(),
            "new honed template",
            msg="tokenizer.chat_template was not updated to the new template.",
        )
        backup_field = reader.get_field("tokenizer.chat_template.backup")
        self.assertIsNotNone(
            backup_field, msg="tokenizer.chat_template.backup field was not created."
        )
        self.assertEqual(
            backup_field.contents(),
            "original pre-patch template",
            msg="tokenizer.chat_template.backup does not contain original pre-patch template.",
        )
        tensor = reader.get_tensor(0)
        self.assertTrue(
            np.array_equal(tensor.data, orig_arr),
            msg="Tensor weights corrupted after patch_gguf.",
        )
        del reader

    def test_extract_and_restore_gguf_backup(self):
        """PASS: extract_gguf_backup and restore_gguf_backup correctly restore template and remove backup key; FAIL: Restore fails or backup key retained."""
        gguf_path, orig_arr = self.create_synthetic_gguf(
            "restore_test.gguf", template="pristine stock template"
        )

        # Before patch, no backup exists
        self.assertIsNone(install.extract_gguf_backup(gguf_path))
        # Restore on unbacked model returns False
        self.assertFalse(install.restore_gguf_backup(gguf_path))

        # Patch model
        install.patch_gguf(gguf_path, "modified template", force=True)
        self.assertEqual(
            install.extract_gguf_backup(gguf_path), "pristine stock template"
        )
        self.assertEqual(
            install.extract_gguf_chat_template(gguf_path), "modified template"
        )

        # Restore backup
        restore_res = install.restore_gguf_backup(gguf_path)
        self.assertTrue(restore_res, msg="restore_gguf_backup returned False.")

        # Verify chat_template is restored and backup key is removed
        self.assertEqual(
            install.extract_gguf_chat_template(gguf_path), "pristine stock template"
        )
        self.assertIsNone(install.extract_gguf_backup(gguf_path))

        reader = gguf.GGUFReader(gguf_path, "r")
        self.assertIsNone(reader.get_field("tokenizer.chat_template.backup"))
        tensor = reader.get_tensor(0)
        self.assertTrue(
            np.array_equal(tensor.data, orig_arr),
            msg="Tensor weights corrupted after restore.",
        )
        del reader

    def test_directory_uninstall_with_existing_template(self):
        """PASS: --uninstall restores tokenizer_config.json and chat_template.jinja from backups and removes .bak files; FAIL: Fails to restore original files or retain backups."""
        model_dir = self.test_dir / "uninst_with_jinja"
        model_dir.mkdir()
        orig_jinja = "original stock jinja"
        orig_config = {
            "bos_token": "<|im_start|>",
            "chat_template": "original stock template",
        }
        (model_dir / "chat_template.jinja").write_text(orig_jinja)
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps(orig_config, indent=2) + "\n"
        )

        # Install
        patch_res = self.run_script(str(model_dir))
        self.assertEqual(patch_res.returncode, 0)
        self.assertTrue((model_dir / "tokenizer_config.json.bak").is_file())
        self.assertTrue((model_dir / "chat_template.jinja.bak").is_file())

        # Uninstall
        uninst_res = self.run_script(str(model_dir), "--uninstall")
        self.assertEqual(
            uninst_res.returncode, 0, msg=f"Uninstall failed: {uninst_res.stderr}"
        )
        self.assertIn(
            "Successfully uninstalled chat template from directory.", uninst_res.stdout
        )
        self.assertFalse((model_dir / "tokenizer_config.json.bak").exists())
        self.assertFalse((model_dir / "chat_template.jinja.bak").exists())
        self.assertEqual((model_dir / "chat_template.jinja").read_text(), orig_jinja)
        self.assertEqual(
            json.loads((model_dir / "tokenizer_config.json").read_text()), orig_config
        )

    def test_directory_uninstall_without_prior_jinja(self):
        """PASS: --uninstall restores tokenizer_config.json and deletes newly created chat_template.jinja; FAIL: Leaves newly created jinja file or fails to restore config."""
        model_dir = self.test_dir / "uninst_no_prior_jinja"
        model_dir.mkdir()
        orig_config = {
            "model_max_length": 8192,
            "chat_template": "stock config template",
        }
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps(orig_config, indent=2) + "\n"
        )

        # Install
        patch_res = self.run_script(str(model_dir))
        self.assertEqual(patch_res.returncode, 0)
        self.assertTrue((model_dir / "tokenizer_config.json.bak").is_file())
        self.assertTrue((model_dir / "chat_template.jinja").is_file())
        self.assertFalse((model_dir / "chat_template.jinja.bak").exists())

        # Uninstall
        uninst_res = self.run_script(str(model_dir), "--uninstall")
        self.assertEqual(
            uninst_res.returncode, 0, msg=f"Uninstall failed: {uninst_res.stderr}"
        )
        self.assertIn(
            "Successfully uninstalled chat template from directory.", uninst_res.stdout
        )
        self.assertFalse((model_dir / "tokenizer_config.json.bak").exists())
        self.assertFalse(
            (model_dir / "chat_template.jinja").exists(),
            msg="Newly created chat_template.jinja was not deleted upon uninstall.",
        )
        self.assertEqual(
            json.loads((model_dir / "tokenizer_config.json").read_text()), orig_config
        )

    def test_directory_uninstall_no_backups_fails(self):
        """PASS: CLI rejects uninstall on directory with no backups; FAIL: Exits with 0 or alters directory."""
        model_dir = self.test_dir / "uninst_no_backups"
        model_dir.mkdir()
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": "curr"})
        )

        res = self.run_script(str(model_dir), "--uninstall")
        self.assert_subprocess_failure(
            res,
            description="Directory uninstall without backup files",
            expected_stderr="No backup files found",
            expected_exit_code=1,
        )

    def test_gguf_uninstall_success(self):
        """PASS: --uninstall restores GGUF chat template from backup metadata and removes backup key; FAIL: Fails to restore template, corrupts tensors, or retains backup key."""
        gguf_path, orig_arr = self.create_synthetic_gguf(
            "uninst_test.gguf", template="stock pre-patch gguf template", alignment=32
        )

        # Patch GGUF
        patch_res = self.run_script(str(gguf_path), "--force")
        self.assertEqual(patch_res.returncode, 0)
        self.assertEqual(
            install.extract_gguf_backup(gguf_path), "stock pre-patch gguf template"
        )

        # Uninstall GGUF
        uninst_res = self.run_script(str(gguf_path), "--uninstall")
        self.assertEqual(
            uninst_res.returncode, 0, msg=f"Uninstall GGUF failed: {uninst_res.stderr}"
        )
        self.assertIn(
            "Successfully uninstalled chat template from GGUF.", uninst_res.stdout
        )

        # Verify chat_template is restored and backup key is removed
        self.assertEqual(
            install.extract_gguf_chat_template(gguf_path),
            "stock pre-patch gguf template",
        )
        self.assertIsNone(install.extract_gguf_backup(gguf_path))

        reader = gguf.GGUFReader(gguf_path, "r")
        self.assertIsNone(reader.get_field("tokenizer.chat_template.backup"))
        tensor = reader.get_tensor(0)
        self.assertTrue(
            np.array_equal(tensor.data, orig_arr),
            msg="Tensor weights corrupted after GGUF uninstall.",
        )
        del reader

    def test_gguf_uninstall_no_backup_fails(self):
        """PASS: CLI rejects GGUF uninstall when no backup metadata key exists; FAIL: Exits with 0 or alters file."""
        gguf_path, orig_arr = self.create_synthetic_gguf(
            "uninst_no_bak.gguf", template="stock template", alignment=32
        )

        res = self.run_script(str(gguf_path), "--uninstall")
        self.assert_subprocess_failure(
            res,
            description="GGUF uninstall without backup key",
            expected_stderr="No backup chat template found",
            expected_exit_code=1,
        )
        self.assertEqual(
            install.extract_gguf_chat_template(gguf_path), "stock template"
        )

    def test_hf_repo_gguf_uninstall_restores_single_cached_file(self):
        """PASS: --uninstall by HF repo restores one cached GGUF and removes its backup key; FAIL: Cached metadata remains patched."""
        commit = "c" * 40
        self.create_synthetic_hf_repo(
            "testorg/uninstall-single-gguf", [{"commit_hash": commit, "files": {}}]
        )
        snap_dir = (
            self.hf_cache_dir
            / "models--testorg--uninstall-single-gguf"
            / "snapshots"
            / commit
        )
        gguf_path, _ = self.create_synthetic_gguf(
            "model-q4.gguf", template="cached stock template", target_dir=snap_dir
        )

        patch_res = self.run_script(
            "testorg/uninstall-single-gguf",
            "--force",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assertEqual(
            patch_res.returncode, 0, msg=f"HF GGUF patch failed: {patch_res.stderr}"
        )
        self.assertEqual(
            install.extract_gguf_backup(gguf_path), "cached stock template"
        )

        uninstall_res = self.run_script(
            "testorg/uninstall-single-gguf",
            "--uninstall",
            "--force",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assertEqual(
            uninstall_res.returncode,
            0,
            msg=f"HF GGUF uninstall failed: {uninstall_res.stderr}",
        )
        self.assertIn("cached Hugging Face GGUF file", uninstall_res.stdout)
        self.assertEqual(
            install.extract_gguf_chat_template(gguf_path), "cached stock template"
        )
        self.assertIsNone(install.extract_gguf_backup(gguf_path))

    def test_hf_repo_gguf_uninstall_explicit_slash_and_colon_targets(self):
        """PASS: Explicit HF slash and colon GGUF targets uninstall only the requested cached file; FAIL: Other files are changed or target syntax fails."""
        commit = "d" * 40
        self.create_synthetic_hf_repo(
            "testorg/uninstall-explicit-gguf", [{"commit_hash": commit, "files": {}}]
        )
        snap_dir = (
            self.hf_cache_dir
            / "models--testorg--uninstall-explicit-gguf"
            / "snapshots"
            / commit
        )
        q4, _ = self.create_synthetic_gguf(
            "q4.gguf", template="q4 stock", target_dir=snap_dir
        )
        q8, _ = self.create_synthetic_gguf(
            "q8.gguf", template="q8 stock", target_dir=snap_dir
        )

        for target, selected, untouched in (
            ("testorg/uninstall-explicit-gguf/q4.gguf", q4, q8),
            ("testorg/uninstall-explicit-gguf:q8.gguf", q8, q4),
        ):
            original = f"{selected.stem} stock"
            patch_res = self.run_script(
                target, "--force", env={"HF_HUB_CACHE": str(self.hf_cache_dir)}
            )
            self.assertEqual(
                patch_res.returncode,
                0,
                msg=f"Patch failed for {target}: {patch_res.stderr}",
            )
            uninstall_res = self.run_script(
                target,
                "--uninstall",
                "--force",
                env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
            )
            self.assertEqual(
                uninstall_res.returncode,
                0,
                msg=f"Uninstall failed for {target}: {uninstall_res.stderr}",
            )
            self.assertEqual(install.extract_gguf_chat_template(selected), original)
            self.assertIsNone(install.extract_gguf_backup(selected))
            self.assertIsNone(install.extract_gguf_backup(untouched))

    def test_hf_repo_gguf_uninstall_all_selected_files(self):
        """PASS: Interactive 'all' selection uninstalls every patched cached GGUF; FAIL: Any backup key or patched template remains."""
        commit = "e" * 40
        self.create_synthetic_hf_repo(
            "testorg/uninstall-all-gguf", [{"commit_hash": commit, "files": {}}]
        )
        snap_dir = (
            self.hf_cache_dir
            / "models--testorg--uninstall-all-gguf"
            / "snapshots"
            / commit
        )
        q4, _ = self.create_synthetic_gguf(
            "q4.gguf", template="q4 stock", target_dir=snap_dir
        )
        q8, _ = self.create_synthetic_gguf(
            "q8.gguf", template="q8 stock", target_dir=snap_dir
        )

        patch_res = self.run_script(
            "testorg/uninstall-all-gguf",
            input="all\nYES\n",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assertEqual(
            patch_res.returncode, 0, msg=f"Patch all failed: {patch_res.stderr}"
        )
        self.assertEqual(install.extract_gguf_backup(q4), "q4 stock")
        self.assertEqual(install.extract_gguf_backup(q8), "q8 stock")

        uninstall_res = self.run_script(
            "testorg/uninstall-all-gguf",
            "--uninstall",
            input="all\n",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assertEqual(
            uninstall_res.returncode,
            0,
            msg=f"Uninstall all failed: {uninstall_res.stderr}",
        )
        self.assertIn("2 cached Hugging Face GGUF files", uninstall_res.stdout)
        for path, original in ((q4, "q4 stock"), (q8, "q8 stock")):
            self.assertEqual(install.extract_gguf_chat_template(path), original)
            self.assertIsNone(install.extract_gguf_backup(path))

    def test_verify_directory_success(self):
        """PASS: verify_directory returns True when directory template files match repository template; FAIL: Returns False for matching templates."""
        model_dir = self.test_dir / "verify_dir_ok"
        model_dir.mkdir()
        source_content = SOURCE_TEMPLATE.read_text(encoding="utf-8")
        (model_dir / "chat_template.jinja").write_text(source_content)
        config_path = model_dir / "tokenizer_config.json"
        config_path.write_text(
            json.dumps({"chat_template": minify_jinja(source_content)})
        )

        self.assertTrue(
            install.verify_directory(model_dir),
            msg="verify_directory returned False for a correctly patched model directory.",
        )

    def test_verify_directory_diagnostics_output(self):
        """PASS: Verification diagnostic probes output detailed status banners during directory patching; FAIL: Output missing required diagnostic probe lines."""
        model_dir = self.test_dir / "verify_diagnostics"
        model_dir.mkdir()
        (model_dir / "chat_template.jinja").write_text("dummy")
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": "dummy"})
        )

        res = self.run_script(str(model_dir))
        self.assertEqual(
            res.returncode,
            0,
            msg=f"Script failed with code {res.returncode}: {res.stderr}",
        )
        self.assertIn(
            "=== Verifying Directory Chat Templates ===",
            res.stdout,
            msg="Diagnostic output missing header.",
        )
        self.assertIn(
            "Version detected:",
            res.stdout,
            msg="Diagnostic output missing version detection report.",
        )
        self.assertIn(
            "Exact content match:",
            res.stdout,
            msg="Diagnostic output missing content match report.",
        )
        self.assertIn(
            "Terseness prompt:",
            res.stdout,
            msg="Diagnostic output missing terseness prompt probe.",
        )
        self.assertIn(
            "Keeps system prompt:",
            res.stdout,
            msg="Diagnostic output missing system prompt probe.",
        )
        self.assertIn(
            "Retains thinking:",
            res.stdout,
            msg="Diagnostic output missing thinking probe.",
        )
        self.assertIn(
            "Equivalence check passed",
            res.stdout,
            msg="Diagnostic output missing equivalence confirmation.",
        )

    def test_verify_directory_source_mismatch_fails(self):
        """PASS: verify_directory returns False when chat_template.jinja and tokenizer_config.json disagree; FAIL: Returns True despite template divergence."""
        model_dir = self.test_dir / "mismatch_dir"
        model_dir.mkdir()
        source_content = SOURCE_TEMPLATE.read_text(encoding="utf-8")
        (model_dir / "chat_template.jinja").write_text(source_content)
        # Differing template in tokenizer_config.json
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "chat_template": "{%- set template_version = 'qwen3.8-honed-v22.5.1' %}{{ messages[0].content }}"
                }
            )
        )

        check_expected_verification_failure(
            self,
            install.verify_directory,
            model_dir,
            description="Directory template source mismatch between jinja file and tokenizer_config.json",
        )

    def test_verify_directory_version_mismatch_fails(self):
        """PASS: verify_directory returns False when installed template version differs from repository version; FAIL: Accepts differing template version."""
        model_dir = self.test_dir / "ver_mismatch_dir"
        model_dir.mkdir()
        source_content = SOURCE_TEMPLATE.read_text(encoding="utf-8")
        bad_version_content = source_content.replace(
            "qwen3.8-honed-v22.5.1", "qwen3.8-honed-v99.9.9"
        )
        (model_dir / "chat_template.jinja").write_text(bad_version_content)
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": minify_jinja(bad_version_content)})
        )

        check_expected_verification_failure(
            self,
            install.verify_directory,
            model_dir,
            description="Directory template version mismatch with repository version",
        )

    def test_verify_directory_missing_terseness_marker_fails(self):
        """PASS: verify_directory returns False when template lacks required terseness system prompt marker; FAIL: Accepts template missing marker."""
        model_dir = self.test_dir / "no_terse_dir"
        model_dir.mkdir()
        source_content = SOURCE_TEMPLATE.read_text(encoding="utf-8")
        no_marker_content = source_content.replace(
            "Never: open with preamble", "Always: be verbose"
        )
        (model_dir / "chat_template.jinja").write_text(no_marker_content)
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": minify_jinja(no_marker_content)})
        )

        check_expected_verification_failure(
            self,
            install.verify_directory,
            model_dir,
            description="Directory template missing required terseness marker",
        )

    def test_verify_gguf_success(self):
        """PASS: verify_gguf returns True when GGUF chat template metadata matches minified repository template; FAIL: Returns False for valid GGUF template."""
        source_minified = minify_jinja(SOURCE_TEMPLATE.read_text(encoding="utf-8"))
        patched_gguf, _ = self.create_synthetic_gguf(
            "patched.gguf", template=source_minified
        )
        self.assertTrue(
            install.verify_gguf(patched_gguf),
            msg="verify_gguf returned False for a correctly patched GGUF file.",
        )

    def test_verify_gguf_unpatched_fails(self):
        """PASS: verify_gguf returns False when GGUF file contains unpatched stock template; FAIL: Returns True for unpatched GGUF template."""
        unpatched_gguf, _ = self.create_synthetic_gguf(
            "unpatched.gguf", template="stock template"
        )
        check_expected_verification_failure(
            self,
            install.verify_gguf,
            unpatched_gguf,
            description="Unpatched GGUF template verification failure",
        )

    def test_cli_directory_verification_failure_exits_nonzero(self):
        """PASS: CLI exits with code 1 when post-patch directory verification fails; FAIL: Exits with 0 despite verification failure."""
        model_dir = self.test_dir / "cli_verify_fail_dir"
        model_dir.mkdir()
        (model_dir / "chat_template.jinja").write_text("old")
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": "old"})
        )

        orig_verify = install.verify_directory
        self.addCleanup(setattr, install, "verify_directory", orig_verify)
        install.verify_directory = lambda *args, **kwargs: False
        with assert_expected_failure(
            "CLI directory verification failure exits with code 1"
        ):
            code = install.main([str(model_dir)])
            self.assertEqual(
                code,
                1,
                msg=f"Expected main to return exit code 1 on verification failure, got {code}",
            )

    def test_cli_gguf_verification_failure_exits_nonzero(self):
        """PASS: CLI exits with code 1 when post-patch GGUF verification fails; FAIL: Exits with 0 despite verification failure."""
        gguf_path, _ = self.create_synthetic_gguf("cli_gguf_fail.gguf", template="init")

        orig_verify = install.verify_gguf
        self.addCleanup(setattr, install, "verify_gguf", orig_verify)
        install.verify_gguf = lambda *args, **kwargs: False
        with assert_expected_failure("CLI GGUF verification failure exits with code 1"):
            code = install.main([str(gguf_path), "--force"])
            self.assertEqual(
                code,
                1,
                msg=f"Expected main to return exit code 1 on verification failure, got {code}",
            )

    def test_verify_source_without_jinja2(self):
        """PASS: verify_source returns False cleanly when jinja2 package is missing; FAIL: Raises unhandled exception or returns True."""
        orig_env = install.Environment
        self.addCleanup(setattr, install, "Environment", orig_env)
        install.Environment = None
        check_expected_verification_failure(
            self,
            install.verify_source,
            "dummy",
            "dummy content",
            "qwen3.8-honed-v22.5.1",
            description="verify_source fails cleanly when jinja2 Environment is unavailable",
        )

    def test_verify_gguf_without_gguf_module(self):
        """PASS: verify_gguf returns False cleanly when gguf package is missing; FAIL: Raises unhandled exception or returns True."""
        orig_gguf = install.gguf
        self.addCleanup(setattr, install, "gguf", orig_gguf)
        install.gguf = None
        dummy_file = self.test_dir / "dummy.gguf"
        dummy_file.write_bytes(b"GGUF")
        check_expected_verification_failure(
            self,
            install.verify_gguf,
            dummy_file,
            description="verify_gguf fails cleanly when gguf module is unavailable",
        )

    def test_is_hf_repo_id(self):
        """PASS: Correctly classifies valid Hugging Face repo IDs and rejects invalid ones; FAIL: False positive/negative identification."""
        self.assertTrue(install.is_hf_repo_id("froggeric/Qwen-Fixed-Chat-Templates"))
        self.assertTrue(install.is_hf_repo_id("testorg/testmodel"))
        self.assertTrue(install.is_hf_repo_id("bert-base-uncased"))
        self.assertTrue(install.is_hf_repo_id("Qwen/Qwen2.5-7B-Instruct"))
        self.assertTrue(install.is_hf_repo_id("org.sub/model-1_2"))

        self.assertFalse(install.is_hf_repo_id(""))
        self.assertFalse(install.is_hf_repo_id("   "))
        self.assertFalse(install.is_hf_repo_id("/testorg/testmodel"))
        self.assertFalse(install.is_hf_repo_id("testorg//testmodel"))
        self.assertFalse(install.is_hf_repo_id("testorg/testmodel/sub"))
        self.assertFalse(install.is_hf_repo_id(None))
        self.assertFalse(install.is_hf_repo_id(12345))

    def test_resolve_hf_model_path_unit(self):
        """PASS: resolve_hf_model_path resolves single/latest revisions and returns None for invalid or missing repos; FAIL: Resolves incorrectly."""
        # Non-HF identifier returns None
        self.assertIsNone(install.resolve_hf_model_path("not/a/valid/repo/path"))

        # Repo not in cache returns None
        self.assertIsNone(
            install.resolve_hf_model_path("missing/repo", cache_dir=self.hf_cache_dir)
        )

        # Single revision returns Path
        commit = "1" * 40
        self.create_synthetic_hf_repo("testorg/single", [{"commit_hash": commit}])
        resolved = install.resolve_hf_model_path(
            "testorg/single", cache_dir=self.hf_cache_dir
        )
        self.assertIsNotNone(resolved)
        self.assertTrue(resolved.is_dir())
        self.assertIn(commit, str(resolved))

        # Multi revisions with latest=True returns newest
        c_old = "a" * 40
        c_new = "b" * 40
        self.create_synthetic_hf_repo(
            "testorg/multi",
            [
                {"commit_hash": c_old, "last_modified": 100.0},
                {"commit_hash": c_new, "last_modified": 200.0},
            ],
        )
        resolved_latest = install.resolve_hf_model_path(
            "testorg/multi", latest=True, cache_dir=self.hf_cache_dir
        )
        self.assertIsNotNone(resolved_latest)
        self.assertIn(c_new, str(resolved_latest))

    def test_resolve_hf_model_path_keyboard_interrupt_aborts(self):
        """PASS: resolve_hf_model_path exits with code 1 upon KeyboardInterrupt during interactive selection; FAIL: Uncaught exception or wrong exit code."""
        c1 = "1" * 40
        c2 = "2" * 40
        self.create_synthetic_hf_repo(
            "testorg/multi-kb",
            [
                {"commit_hash": c1, "last_modified": 100.0},
                {"commit_hash": c2, "last_modified": 200.0},
            ],
        )
        with mock.patch("builtins.input", side_effect=KeyboardInterrupt):
            with assert_expected_failure(
                "resolve_hf_model_path aborts cleanly on KeyboardInterrupt",
                expected_stderr_pattern="Aborted by user",
            ):
                with self.assertRaises(SystemExit) as cm:
                    install.resolve_hf_model_path(
                        "testorg/multi-kb", latest=False, cache_dir=self.hf_cache_dir
                    )
                self.assertEqual(cm.exception.code, 1)

    def test_resolve_hf_model_path_eof_aborts(self):
        """PASS: resolve_hf_model_path exits with code 1 upon EOFError during interactive selection; FAIL: Uncaught exception or wrong exit code."""
        c1 = "1" * 40
        c2 = "2" * 40
        self.create_synthetic_hf_repo(
            "testorg/multi-eof",
            [
                {"commit_hash": c1, "last_modified": 100.0},
                {"commit_hash": c2, "last_modified": 200.0},
            ],
        )
        with mock.patch("builtins.input", side_effect=EOFError):
            with assert_expected_failure(
                "resolve_hf_model_path aborts cleanly on EOFError",
                expected_stderr_pattern="Aborted by user",
            ):
                with self.assertRaises(SystemExit) as cm:
                    install.resolve_hf_model_path(
                        "testorg/multi-eof", latest=False, cache_dir=self.hf_cache_dir
                    )
                self.assertEqual(cm.exception.code, 1)

    def test_hf_resolve_single_revision_auto_selection(self):
        """PASS: CLI auto-resolves single-revision model without prompt and applies template; FAIL: Prompts or fails to patch."""
        commit = "c" * 40
        self.create_synthetic_hf_repo("testorg/single-cli", [{"commit_hash": commit}])
        res = self.run_script(
            "testorg/single-cli", env={"HF_HUB_CACHE": str(self.hf_cache_dir)}
        )
        self.assertEqual(res.returncode, 0, msg=f"Script failed: {res.stderr}")
        self.assertIn("Successfully applied chat template to directory.", res.stdout)
        self.assertNotIn("Select revision", res.stdout)

        snap_dir = (
            self.hf_cache_dir / "models--testorg--single-cli" / "snapshots" / commit
        )
        self.assertTrue((snap_dir / "chat_template.jinja").is_file())
        config_data = json.loads(
            (snap_dir / "tokenizer_config.json").read_text(encoding="utf-8")
        )
        self.assertIn("Never: open with preamble", config_data["chat_template"])

    def test_hf_resolve_multiple_revisions_with_latest_flag(self):
        """PASS: CLI with --latest flag automatically patches newest revision; FAIL: Prompts or patches wrong revision."""
        c_old = "1" * 40
        c_new = "2" * 40
        self.create_synthetic_hf_repo(
            "testorg/multi-latest-cli",
            [
                {"commit_hash": c_old, "last_modified": 1000.0, "refs": ["v1.0"]},
                {"commit_hash": c_new, "last_modified": 2000.0, "refs": ["main"]},
            ],
        )
        res = self.run_script(
            "testorg/multi-latest-cli",
            "--latest",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assertEqual(res.returncode, 0, msg=f"Script failed: {res.stderr}")
        self.assertIn("Successfully applied chat template to directory.", res.stdout)
        self.assertNotIn("Select revision", res.stdout)

        repo_dir = self.hf_cache_dir / "models--testorg--multi-latest-cli" / "snapshots"
        snap_new = repo_dir / c_new
        snap_old = repo_dir / c_old

        # Newest revision is patched
        self.assertTrue((snap_new / "chat_template.jinja").is_file())
        new_config = json.loads(
            (snap_new / "tokenizer_config.json").read_text(encoding="utf-8")
        )
        self.assertIn("Never: open with preamble", new_config["chat_template"])

        # Older revision remains untouched
        self.assertFalse((snap_old / "chat_template.jinja").is_file())
        old_config = json.loads(
            (snap_old / "tokenizer_config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(old_config["chat_template"], "old_template_content")

    def test_hf_resolve_multiple_revisions_interactive_by_index(self):
        """PASS: Interactive prompt accepts index selection and patches chosen revision; FAIL: Prompts fail or wrong revision patched."""
        c_old = "1" * 40
        c_new = "2" * 40
        self.create_synthetic_hf_repo(
            "testorg/multi-interactive-cli",
            [
                {"commit_hash": c_old, "last_modified": 1000.0},
                {"commit_hash": c_new, "last_modified": 2000.0},
            ],
        )
        # In chronological descending order: [1] is c_new, [2] is c_old.
        # Select index '2' to patch the older revision.
        res = self.run_script(
            "testorg/multi-interactive-cli",
            input="2\n",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assertEqual(res.returncode, 0, msg=f"Script failed: {res.stderr}")
        self.assertIn("Select revision", res.stdout)
        self.assertIn("Successfully applied chat template to directory.", res.stdout)

        repo_dir = (
            self.hf_cache_dir / "models--testorg--multi-interactive-cli" / "snapshots"
        )
        snap_old = repo_dir / c_old
        snap_new = repo_dir / c_new

        # Selected revision 2 (c_old) is patched
        self.assertTrue((snap_old / "chat_template.jinja").is_file())
        old_config = json.loads(
            (snap_old / "tokenizer_config.json").read_text(encoding="utf-8")
        )
        self.assertIn("Never: open with preamble", old_config["chat_template"])

        # Revision 1 (c_new) remains untouched
        self.assertFalse((snap_new / "chat_template.jinja").is_file())

    def test_hf_resolve_multiple_revisions_interactive_by_commit_hash(self):
        """PASS: Interactive prompt accepts commit hash prefix and patches matched revision; FAIL: Commit hash prefix rejected."""
        c1 = "a" * 40
        c2 = "b" * 40
        self.create_synthetic_hf_repo(
            "testorg/multi-commit-prefix",
            [
                {"commit_hash": c1, "last_modified": 1000.0},
                {"commit_hash": c2, "last_modified": 2000.0},
            ],
        )
        res = self.run_script(
            "testorg/multi-commit-prefix",
            input=f"{c1[:8]}\n",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assertEqual(res.returncode, 0, msg=f"Script failed: {res.stderr}")
        self.assertIn("Successfully applied chat template to directory.", res.stdout)

        snap_c1 = (
            self.hf_cache_dir
            / "models--testorg--multi-commit-prefix"
            / "snapshots"
            / c1
        )
        self.assertTrue((snap_c1 / "chat_template.jinja").is_file())

    def test_hf_resolve_multiple_revisions_interactive_by_ref(self):
        """PASS: Interactive prompt accepts ref name and patches matched revision; FAIL: Ref name rejected."""
        c1 = "1" * 40
        c2 = "2" * 40
        self.create_synthetic_hf_repo(
            "testorg/multi-ref",
            [
                {"commit_hash": c1, "last_modified": 1000.0, "refs": ["v1.0"]},
                {"commit_hash": c2, "last_modified": 2000.0, "refs": ["main"]},
            ],
        )
        res = self.run_script(
            "testorg/multi-ref",
            input="v1.0\n",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assertEqual(res.returncode, 0, msg=f"Script failed: {res.stderr}")
        self.assertIn("Successfully applied chat template to directory.", res.stdout)

        snap_c1 = self.hf_cache_dir / "models--testorg--multi-ref" / "snapshots" / c1
        self.assertTrue((snap_c1 / "chat_template.jinja").is_file())

    def test_hf_resolve_multiple_revisions_interactive_default_selection(self):
        """PASS: Interactive prompt defaults to newest revision when user presses Enter; FAIL: Fails to use default revision."""
        c_old = "1" * 40
        c_new = "2" * 40
        self.create_synthetic_hf_repo(
            "testorg/multi-default",
            [
                {"commit_hash": c_old, "last_modified": 1000.0},
                {"commit_hash": c_new, "last_modified": 2000.0},
            ],
        )
        res = self.run_script(
            "testorg/multi-default",
            input="\n",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assertEqual(res.returncode, 0, msg=f"Script failed: {res.stderr}")
        self.assertIn("Successfully applied chat template to directory.", res.stdout)

        snap_new = (
            self.hf_cache_dir / "models--testorg--multi-default" / "snapshots" / c_new
        )
        self.assertTrue((snap_new / "chat_template.jinja").is_file())

    def test_hf_resolve_multiple_revisions_interactive_retry_invalid_input(self):
        """PASS: Interactive prompt reprompts on invalid selection and succeeds on subsequent valid entry; FAIL: Aborts on invalid input."""
        c1 = "1" * 40
        c2 = "2" * 40
        self.create_synthetic_hf_repo(
            "testorg/multi-retry",
            [
                {"commit_hash": c1, "last_modified": 1000.0},
                {"commit_hash": c2, "last_modified": 2000.0},
            ],
        )
        res = self.run_script(
            "testorg/multi-retry",
            input="99\n1\n",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assertEqual(res.returncode, 0, msg=f"Script failed: {res.stderr}")
        self.assertIn("Invalid selection '99'", res.stderr)
        self.assertIn("Successfully applied chat template to directory.", res.stdout)

    def test_hf_resolve_multiple_revisions_eof_aborts_cleanly(self):
        """PASS: Closed standard input during interactive selection exits with code 1; FAIL: Crashes or exits with 0."""
        c1 = "1" * 40
        c2 = "2" * 40
        self.create_synthetic_hf_repo(
            "testorg/multi-eof-cli",
            [
                {"commit_hash": c1, "last_modified": 1000.0},
                {"commit_hash": c2, "last_modified": 2000.0},
            ],
        )
        res = self.run_script(
            "testorg/multi-eof-cli",
            input="",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assert_subprocess_failure(
            res,
            description="Interactive selection EOF",
            expected_stderr="Aborted by user.",
            expected_exit_code=1,
        )

    def test_hf_repo_not_found_in_cache(self):
        """PASS: Non-existent HF repo ID reports descriptive error to stderr and exits with code 1; FAIL: Misleading error or 0 exit code."""
        res = self.run_script(
            "nonexistent-org/nonexistent-model",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assert_subprocess_failure(
            res,
            description="HF repo not in cache",
            expected_stderr="Hugging Face model repository 'nonexistent-org/nonexistent-model' not found in local cache.",
            expected_exit_code=1,
        )

    def test_hf_repo_no_valid_snapshots_in_cache(self):
        """PASS: Cached HF repo with no valid snapshots reports descriptive error to stderr and exits with code 1; FAIL: Accepts empty repo."""
        orig_find = install.find_cached_repo
        orig_resolve = install.resolve_hf_model_path
        self.addCleanup(setattr, install, "find_cached_repo", orig_find)
        self.addCleanup(setattr, install, "resolve_hf_model_path", orig_resolve)
        install.find_cached_repo = lambda *args, **kwargs: object()
        install.resolve_hf_model_path = lambda *args, **kwargs: None

        with assert_expected_failure(
            "HF model repo with no snapshots in cache",
            expected_stderr_pattern="No valid snapshot directory found in cache for Hugging Face repository",
        ):
            code = install.main(["testorg/empty-snapshots"])
            self.assertEqual(code, 1)

    def test_hf_model_uninstall_success(self):
        """PASS: --uninstall with HF model repo ID restores original snapshot files and removes backups; FAIL: Fails to restore snapshot."""
        commit = "u" * 40
        self.create_synthetic_hf_repo("testorg/uninstall-hf", [{"commit_hash": commit}])

        # 1. Apply patch
        res_apply = self.run_script(
            "testorg/uninstall-hf",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assertEqual(
            res_apply.returncode, 0, msg=f"Apply failed: {res_apply.stderr}"
        )

        snap_dir = (
            self.hf_cache_dir / "models--testorg--uninstall-hf" / "snapshots" / commit
        )
        self.assertTrue((snap_dir / "chat_template.jinja").is_file())
        self.assertTrue((snap_dir / "tokenizer_config.json.bak").is_file())

        # 2. Uninstall
        res_uninst = self.run_script(
            "testorg/uninstall-hf",
            "--uninstall",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assertEqual(
            res_uninst.returncode, 0, msg=f"Uninstall failed: {res_uninst.stderr}"
        )
        self.assertIn(
            "Successfully uninstalled chat template from directory.", res_uninst.stdout
        )

        # 3. Verify clean uninstallation
        self.assertFalse((snap_dir / "chat_template.jinja").is_file())
        self.assertFalse((snap_dir / "tokenizer_config.json.bak").is_file())
        restored_config = json.loads(
            (snap_dir / "tokenizer_config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(restored_config["chat_template"], "old_template_content")

    def test_regression_non_hf_missing_path_message(self):
        """PASS: Non-HF missing paths preserve exact backwards-compatible 'does not exist' error; FAIL: Phrasing altered."""
        res_dir = self.run_script("./non_existent_relative_dir")
        self.assert_subprocess_failure(
            res_dir,
            description="Non-HF relative path does not exist",
            expected_stderr="Error: Target path does not exist: './non_existent_relative_dir'",
            expected_exit_code=1,
        )

        res_gguf = self.run_script("missing_model.gguf")
        self.assert_subprocess_failure(
            res_gguf,
            description="Missing .gguf file does not exist",
            expected_stderr="Error: Target path does not exist: 'missing_model.gguf'",
            expected_exit_code=1,
        )

    def test_parse_hf_target(self):
        """PASS: parse_hf_target correctly parses repo IDs and explicit GGUF filenames; FAIL: Misparses or returns invalid."""
        # Plain repo IDs
        self.assertEqual(
            install.parse_hf_target("Qwen/Qwen2.5-7B-Instruct-GGUF"),
            ("Qwen/Qwen2.5-7B-Instruct-GGUF", None),
        )
        self.assertEqual(
            install.parse_hf_target("testorg/model"),
            ("testorg/model", None),
        )

        # Explicit GGUF subpaths with slash
        self.assertEqual(
            install.parse_hf_target(
                "Qwen/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q4_k_m.gguf"
            ),
            ("Qwen/Qwen2.5-7B-Instruct-GGUF", "qwen2.5-7b-instruct-q4_k_m.gguf"),
        )
        self.assertEqual(
            install.parse_hf_target("testorg/model/subfolder/q4.gguf"),
            ("testorg/model", "subfolder/q4.gguf"),
        )

        # Explicit GGUF subpaths with colon
        self.assertEqual(
            install.parse_hf_target(
                "Qwen/Qwen2.5-7B-Instruct-GGUF:qwen2.5-7b-instruct-q4_k_m.gguf"
            ),
            ("Qwen/Qwen2.5-7B-Instruct-GGUF", "qwen2.5-7b-instruct-q4_k_m.gguf"),
        )
        self.assertEqual(
            install.parse_hf_target("gpt2:model.gguf"),
            ("gpt2", "model.gguf"),
        )

        # Local filesystem paths / non-HF targets rejected
        self.assertIsNone(install.parse_hf_target("missing_model.gguf"))
        self.assertIsNone(install.parse_hf_target("./relative_dir/model.gguf"))
        self.assertIsNone(install.parse_hf_target("../parent/model.gguf"))
        self.assertIsNone(install.parse_hf_target("/absolute/path/model.gguf"))
        self.assertIsNone(install.parse_hf_target("local_folder/missing.gguf"))
        self.assertIsNone(install.parse_hf_target(""))
        self.assertIsNone(install.parse_hf_target(None))

    def test_find_snapshot_gguf_files(self):
        """PASS: find_snapshot_gguf_files returns all valid GGUF files in snapshot, including nested and symlinked; FAIL: Misses files or crashes."""
        snap_dir = self.test_dir / "snap_discovery"
        snap_dir.mkdir(parents=True, exist_ok=True)

        # 1. Empty directory
        self.assertEqual(install.find_snapshot_gguf_files(snap_dir), [])
        self.assertEqual(
            install.find_snapshot_gguf_files(self.test_dir / "nonexistent"), []
        )

        # 2. Text files only
        (snap_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (snap_dir / "README.md").write_text("# Test", encoding="utf-8")
        self.assertEqual(install.find_snapshot_gguf_files(snap_dir), [])

        # 3. Direct GGUF file
        g1, _ = self.create_synthetic_gguf("model-q4.gguf", target_dir=snap_dir)
        found = install.find_snapshot_gguf_files(snap_dir)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].name, "model-q4.gguf")

        # 4. Nested GGUF file in subfolder
        sub = snap_dir / "nested"
        sub.mkdir(parents=True, exist_ok=True)
        g2, _ = self.create_synthetic_gguf("model-q8.gguf", target_dir=sub)
        found = install.find_snapshot_gguf_files(snap_dir)
        self.assertEqual(len(found), 2)
        names = [f.name for f in found]
        self.assertIn("model-q4.gguf", names)
        self.assertIn("model-q8.gguf", names)

        # 5. Symlinked GGUF to blob
        blobs_dir = self.test_dir / "blobs_test"
        blobs_dir.mkdir(parents=True, exist_ok=True)
        blob_path, _ = self.create_synthetic_gguf(
            "blob_hash_1234", target_dir=blobs_dir
        )
        symlink_path = snap_dir / "model-symlink.gguf"
        symlink_path.symlink_to(blob_path)
        found = install.find_snapshot_gguf_files(snap_dir)
        self.assertEqual(len(found), 3)
        names = [f.name for f in found]
        self.assertIn("model-symlink.gguf", names)
        # Verify relative_to works
        for f in found:
            self.assertTrue(str(f.relative_to(snap_dir)))

    def test_hf_repo_single_gguf_auto_selected(self):
        """PASS: HF repo snapshot with a single GGUF auto-selects and patches the GGUF with --force; FAIL: Prompts or crashes."""
        commit = "1" * 40
        self.create_synthetic_hf_repo(
            "testorg/single-gguf", [{"commit_hash": commit, "files": {}}]
        )
        snap_dir = (
            self.hf_cache_dir / "models--testorg--single-gguf" / "snapshots" / commit
        )
        self.create_synthetic_gguf("model-q4_k_m.gguf", target_dir=snap_dir)

        res = self.run_script(
            "testorg/single-gguf",
            "--force",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assertEqual(res.returncode, 0, msg=f"Script failed: {res.stderr}")
        self.assertIn("Successfully applied chat template to GGUF.", res.stdout)
        self.assertTrue(
            install.verify_gguf(snap_dir / "model-q4_k_m.gguf", SOURCE_TEMPLATE)
        )

    def test_hf_repo_explicit_target_slash_and_colon(self):
        """PASS: Explicit repo_id/file.gguf and repo_id:file.gguf patch only the specified GGUF; FAIL: Disambiguation prompt or wrong file."""
        commit = "2" * 40
        self.create_synthetic_hf_repo(
            "testorg/explicit-target", [{"commit_hash": commit, "files": {}}]
        )
        snap_dir = (
            self.hf_cache_dir
            / "models--testorg--explicit-target"
            / "snapshots"
            / commit
        )
        self.create_synthetic_gguf("q4.gguf", target_dir=snap_dir)
        self.create_synthetic_gguf("q8.gguf", target_dir=snap_dir)

        # 1. Target with slash: repo_id/q4.gguf
        res_slash = self.run_script(
            "testorg/explicit-target/q4.gguf",
            "--force",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assertEqual(
            res_slash.returncode, 0, msg=f"Slash target failed: {res_slash.stderr}"
        )
        self.assertIn("Successfully applied chat template to GGUF.", res_slash.stdout)
        self.assertTrue(install.verify_gguf(snap_dir / "q4.gguf", SOURCE_TEMPLATE))
        # q8.gguf should remain unpatched
        check_expected_verification_failure(
            self,
            install.verify_gguf,
            snap_dir / "q8.gguf",
            SOURCE_TEMPLATE,
            description="q8.gguf remained unpatched after q4.gguf patch",
        )

        # 2. Target with colon: repo_id:q8.gguf
        res_colon = self.run_script(
            "testorg/explicit-target:q8.gguf",
            "--force",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assertEqual(
            res_colon.returncode, 0, msg=f"Colon target failed: {res_colon.stderr}"
        )
        self.assertIn("Successfully applied chat template to GGUF.", res_colon.stdout)
        self.assertTrue(install.verify_gguf(snap_dir / "q8.gguf", SOURCE_TEMPLATE))

    def test_hf_repo_explicit_target_missing(self):
        """PASS: Explicit GGUF file target not found in snapshot reports error and exits with code 1; FAIL: Ignores missing file."""
        commit = "3" * 40
        self.create_synthetic_hf_repo(
            "testorg/explicit-missing", [{"commit_hash": commit, "files": {}}]
        )
        snap_dir = (
            self.hf_cache_dir
            / "models--testorg--explicit-missing"
            / "snapshots"
            / commit
        )
        self.create_synthetic_gguf("q4.gguf", target_dir=snap_dir)

        res = self.run_script(
            "testorg/explicit-missing/nonexistent.gguf",
            "--force",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assert_subprocess_failure(
            res,
            description="Explicit GGUF target not in snapshot",
            expected_stderr="GGUF file 'nonexistent.gguf' not found in cached snapshot",
            expected_exit_code=1,
        )

    def test_hf_repo_multiple_ggufs_forced_requires_explicit_target(self):
        """PASS: Passing --force on HF repo with multiple GGUFs without explicit target fails with descriptive error; FAIL: Guesses or passes."""
        commit = "4" * 40
        self.create_synthetic_hf_repo(
            "testorg/multi-forced", [{"commit_hash": commit, "files": {}}]
        )
        snap_dir = (
            self.hf_cache_dir / "models--testorg--multi-forced" / "snapshots" / commit
        )
        self.create_synthetic_gguf("q4.gguf", target_dir=snap_dir)
        self.create_synthetic_gguf("q8.gguf", target_dir=snap_dir)

        res = self.run_script(
            "testorg/multi-forced",
            "--force",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assert_subprocess_failure(
            res,
            description="Multiple GGUFs with --force without explicit target",
            expected_stderr="Multiple GGUF files found in snapshot",
            expected_exit_code=1,
        )
        self.assertIn("specify a gguf file", res.stderr.lower())
        self.assertIn("explicit selection path", res.stderr.lower())

    def test_hf_repo_multiple_ggufs_eof_fails_descriptive(self):
        """PASS: Non-interactive/closed stdin on HF repo with multiple GGUFs fails with descriptive error; FAIL: Crashes or hangs."""
        commit = "5" * 40
        self.create_synthetic_hf_repo(
            "testorg/multi-eof", [{"commit_hash": commit, "files": {}}]
        )
        snap_dir = (
            self.hf_cache_dir / "models--testorg--multi-eof" / "snapshots" / commit
        )
        self.create_synthetic_gguf("q4.gguf", target_dir=snap_dir)
        self.create_synthetic_gguf("q8.gguf", target_dir=snap_dir)

        res = self.run_script(
            "testorg/multi-eof",
            input="",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assert_subprocess_failure(
            res,
            description="Multiple GGUFs with EOF stdin",
            expected_stderr="standard input was closed (eof)",
            expected_exit_code=1,
        )
        self.assertIn("specify a gguf file", res.stderr.lower())
        self.assertIn("explicit selection path", res.stderr.lower())

    def test_hf_repo_multiple_ggufs_interactive_index(self):
        """PASS: Interactive selection by index '1' patches first GGUF and leaves second untouched; FAIL: Selects wrong or fails."""
        commit = "6" * 40
        self.create_synthetic_hf_repo(
            "testorg/multi-idx", [{"commit_hash": commit, "files": {}}]
        )
        snap_dir = (
            self.hf_cache_dir / "models--testorg--multi-idx" / "snapshots" / commit
        )
        self.create_synthetic_gguf("q4.gguf", target_dir=snap_dir)
        self.create_synthetic_gguf("q8.gguf", target_dir=snap_dir)

        res = self.run_script(
            "testorg/multi-idx",
            input="1\nYES\n",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assertEqual(
            res.returncode, 0, msg=f"Interactive selection failed: {res.stderr}"
        )
        self.assertIn("Successfully applied chat template to GGUF.", res.stdout)
        self.assertTrue(install.verify_gguf(snap_dir / "q4.gguf", SOURCE_TEMPLATE))
        check_expected_verification_failure(
            self,
            install.verify_gguf,
            snap_dir / "q8.gguf",
            SOURCE_TEMPLATE,
            description="q8.gguf untouched in index selection test",
        )

    def test_hf_repo_multiple_ggufs_interactive_filename(self):
        """PASS: Interactive selection by filename 'q8.gguf' patches specified GGUF; FAIL: Rejects filename selection."""
        commit = "7" * 40
        self.create_synthetic_hf_repo(
            "testorg/multi-file", [{"commit_hash": commit, "files": {}}]
        )
        snap_dir = (
            self.hf_cache_dir / "models--testorg--multi-file" / "snapshots" / commit
        )
        self.create_synthetic_gguf("q4.gguf", target_dir=snap_dir)
        self.create_synthetic_gguf("q8.gguf", target_dir=snap_dir)

        res = self.run_script(
            "testorg/multi-file",
            input="q8.gguf\nYES\n",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assertEqual(
            res.returncode, 0, msg=f"Filename selection failed: {res.stderr}"
        )
        self.assertIn("Successfully applied chat template to GGUF.", res.stdout)
        self.assertTrue(install.verify_gguf(snap_dir / "q8.gguf", SOURCE_TEMPLATE))
        check_expected_verification_failure(
            self,
            install.verify_gguf,
            snap_dir / "q4.gguf",
            SOURCE_TEMPLATE,
            description="q4.gguf untouched in filename selection test",
        )

    def test_hf_repo_multiple_ggufs_interactive_all(self):
        """PASS: Interactive selection 'all' patches all discovered GGUF files in snapshot; FAIL: Leaves files unpatched."""
        commit = "8" * 40
        self.create_synthetic_hf_repo(
            "testorg/multi-all", [{"commit_hash": commit, "files": {}}]
        )
        snap_dir = (
            self.hf_cache_dir / "models--testorg--multi-all" / "snapshots" / commit
        )
        self.create_synthetic_gguf("q4.gguf", target_dir=snap_dir)
        self.create_synthetic_gguf("q8.gguf", target_dir=snap_dir)

        res = self.run_script(
            "testorg/multi-all",
            input="all\nYES\n",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assertEqual(res.returncode, 0, msg=f"Select all failed: {res.stderr}")
        self.assertIn("Successfully applied chat template to GGUF.", res.stdout)
        self.assertTrue(install.verify_gguf(snap_dir / "q4.gguf", SOURCE_TEMPLATE))
        self.assertTrue(install.verify_gguf(snap_dir / "q8.gguf", SOURCE_TEMPLATE))

    def test_hf_repo_multiple_ggufs_interactive_retry(self):
        """PASS: Invalid input prompts user again and succeeds upon valid input; FAIL: Aborts on first invalid input."""
        commit = "9" * 40
        self.create_synthetic_hf_repo(
            "testorg/multi-retry-gguf", [{"commit_hash": commit, "files": {}}]
        )
        snap_dir = (
            self.hf_cache_dir
            / "models--testorg--multi-retry-gguf"
            / "snapshots"
            / commit
        )
        self.create_synthetic_gguf("q4.gguf", target_dir=snap_dir)
        self.create_synthetic_gguf("q8.gguf", target_dir=snap_dir)

        res = self.run_script(
            "testorg/multi-retry-gguf",
            input="invalid_choice\n1\nYES\n",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assertEqual(res.returncode, 0, msg=f"Retry failed: {res.stderr}")
        self.assertIn("Invalid selection 'invalid_choice'", res.stderr)
        self.assertTrue(install.verify_gguf(snap_dir / "q4.gguf", SOURCE_TEMPLATE))

    def test_hf_repo_fallback_to_directory_mode(self):
        """PASS: HF repo without GGUF files seamlessly falls back to directory patching mode; FAIL: Fails or misses directory."""
        commit = "d" * 40
        self.create_synthetic_hf_repo("testorg/fallback-dir", [{"commit_hash": commit}])
        snap_dir = (
            self.hf_cache_dir / "models--testorg--fallback-dir" / "snapshots" / commit
        )

        res = self.run_script(
            "testorg/fallback-dir",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assertEqual(res.returncode, 0, msg=f"Fallback failed: {res.stderr}")
        self.assertIn("Successfully applied chat template to directory.", res.stdout)
        self.assertTrue((snap_dir / "chat_template.jinja").is_file())
        self.assertTrue((snap_dir / "tokenizer_config.json.bak").is_file())
        self.assertTrue(install.verify_directory(snap_dir, SOURCE_TEMPLATE))

    def test_warn_snapshot_surgery_output(self):
        """PASS: warn_snapshot_surgery emits warning with Snapshot Surgery, HF_HUB_OFFLINE=1, and target path; FAIL: Omits required warning details."""
        buf = io.StringIO()
        target = Path("/tmp/fake/snapshot/dir")
        with contextlib.redirect_stderr(buf):
            install.warn_snapshot_surgery(target)
        output = buf.getvalue()
        self.assertIn("Snapshot Surgery", output)
        self.assertIn("HF_HUB_OFFLINE=1", output)
        self.assertIn(str(target), output)
        self.assertIn("in-place", output.lower())

    def test_confirm_snapshot_surgery_force(self):
        """PASS: confirm_snapshot_surgery returns True immediately when force=True; FAIL: Prompts or returns False."""
        self.assertTrue(install.confirm_snapshot_surgery(force=True))

    def test_confirm_snapshot_surgery_interactive_yes(self):
        """PASS: confirm_snapshot_surgery returns True when user types YES; FAIL: Returns False or raises."""
        with mock.patch("builtins.input", return_value="YES"):
            self.assertTrue(install.confirm_snapshot_surgery(force=False))

    def test_confirm_snapshot_surgery_interactive_no(self):
        """PASS: confirm_snapshot_surgery returns False and prints abort message when user does not enter YES; FAIL: Returns True or omits message."""
        buf = io.StringIO()
        with mock.patch("builtins.input", return_value="NO"):
            with contextlib.redirect_stderr(buf):
                result = install.confirm_snapshot_surgery(force=False)
        self.assertFalse(result)
        self.assertIn("Aborted: Confirmation 'YES' was not received.", buf.getvalue())

    def test_confirm_snapshot_surgery_eof(self):
        """PASS: confirm_snapshot_surgery exits with code 1 when EOFError is raised; FAIL: Doesn't exit or wrong code."""
        with mock.patch("builtins.input", side_effect=EOFError):
            with self.assertRaises(SystemExit) as ctx:
                with contextlib.redirect_stderr(io.StringIO()):
                    install.confirm_snapshot_surgery(force=False)
            self.assertEqual(ctx.exception.code, 1)

    def test_confirm_snapshot_surgery_keyboard_interrupt(self):
        """PASS: confirm_snapshot_surgery exits with code 1 when KeyboardInterrupt is raised; FAIL: Doesn't exit or wrong code."""
        with mock.patch("builtins.input", side_effect=KeyboardInterrupt):
            with self.assertRaises(SystemExit) as ctx:
                with contextlib.redirect_stderr(io.StringIO()):
                    install.confirm_snapshot_surgery(force=False)
            self.assertEqual(ctx.exception.code, 1)

    def test_snapshot_surgery_warning_and_offline_notice_with_force(self):
        """PASS: Snapshot Surgery emits HF_HUB_OFFLINE=1 and Snapshot Surgery warning even with --force; FAIL: Omits warning."""
        commit = "a" * 40
        self.create_synthetic_hf_repo(
            "testorg/warn-force", [{"commit_hash": commit, "files": {}}]
        )
        snap_dir = (
            self.hf_cache_dir / "models--testorg--warn-force" / "snapshots" / commit
        )
        self.create_synthetic_gguf("model.gguf", target_dir=snap_dir)

        res = self.run_script(
            "testorg/warn-force",
            "--force",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assertEqual(res.returncode, 0, msg=f"Script failed: {res.stderr}")
        self.assertIn("Snapshot Surgery", res.stderr)
        self.assertIn("HF_HUB_OFFLINE=1", res.stderr)

    def test_hf_repo_explicit_target_syntax_slash(self):
        """PASS: Explicit slash target syntax patches the target GGUF; FAIL: Patches wrong file or fails."""
        commit = "b" * 40
        self.create_synthetic_hf_repo(
            "testorg/explicit-slash", [{"commit_hash": commit, "files": {}}]
        )
        snap_dir = (
            self.hf_cache_dir / "models--testorg--explicit-slash" / "snapshots" / commit
        )
        self.create_synthetic_gguf("q4.gguf", target_dir=snap_dir)
        self.create_synthetic_gguf("q8.gguf", target_dir=snap_dir)

        res = self.run_script(
            "testorg/explicit-slash/q4.gguf",
            "--force",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assertEqual(res.returncode, 0, msg=f"Script failed: {res.stderr}")
        self.assertTrue(install.verify_gguf(snap_dir / "q4.gguf", SOURCE_TEMPLATE))
        check_expected_verification_failure(
            self,
            install.verify_gguf,
            snap_dir / "q8.gguf",
            SOURCE_TEMPLATE,
            description="q8.gguf untouched in slash target test",
        )

    def test_hf_repo_explicit_target_syntax_colon(self):
        """PASS: Explicit colon target syntax patches the target GGUF; FAIL: Patches wrong file or fails."""
        commit = "c" * 40
        self.create_synthetic_hf_repo(
            "testorg/explicit-colon", [{"commit_hash": commit, "files": {}}]
        )
        snap_dir = (
            self.hf_cache_dir / "models--testorg--explicit-colon" / "snapshots" / commit
        )
        self.create_synthetic_gguf("q4.gguf", target_dir=snap_dir)
        self.create_synthetic_gguf("q8.gguf", target_dir=snap_dir)

        res = self.run_script(
            "testorg/explicit-colon:q8.gguf",
            "--force",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assertEqual(res.returncode, 0, msg=f"Script failed: {res.stderr}")
        self.assertTrue(install.verify_gguf(snap_dir / "q8.gguf", SOURCE_TEMPLATE))
        check_expected_verification_failure(
            self,
            install.verify_gguf,
            snap_dir / "q4.gguf",
            SOURCE_TEMPLATE,
            description="q4.gguf untouched in colon target test",
        )

    def test_snapshot_surgery_interactive_yes_succeeds(self):
        """PASS: Single GGUF HF snapshot prompts for Snapshot Surgery consent, succeeds on YES, and patches GGUF; FAIL: Fails or leaves unpatched."""
        commit = "b" * 40
        self.create_synthetic_hf_repo(
            "testorg/consent-yes", [{"commit_hash": commit, "files": {}}]
        )
        snap_dir = (
            self.hf_cache_dir / "models--testorg--consent-yes" / "snapshots" / commit
        )
        self.create_synthetic_gguf("model.gguf", target_dir=snap_dir)

        res = self.run_script(
            "testorg/consent-yes",
            input="YES\n",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assertEqual(res.returncode, 0, msg=f"Script failed: {res.stderr}")
        self.assertIn("Snapshot Surgery", res.stderr)
        self.assertIn("HF_HUB_OFFLINE=1", res.stderr)
        self.assertIn("Successfully applied chat template to GGUF.", res.stdout)
        self.assertTrue(install.verify_gguf(snap_dir / "model.gguf", SOURCE_TEMPLATE))

    def test_snapshot_surgery_interactive_no_aborts_cleanly(self):
        """PASS: Declining Snapshot Surgery confirmation aborts cleanly with exit code 1 and leaves file unmodified; FAIL: Patches file or returns 0."""
        commit = "c" * 40
        self.create_synthetic_hf_repo(
            "testorg/consent-no", [{"commit_hash": commit, "files": {}}]
        )
        snap_dir = (
            self.hf_cache_dir / "models--testorg--consent-no" / "snapshots" / commit
        )
        self.create_synthetic_gguf(
            "model.gguf", template="unmodified pre-patch", target_dir=snap_dir
        )

        res = self.run_script(
            "testorg/consent-no",
            input="NO\n",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assertEqual(
            res.returncode, 1, msg=f"Expected exit code 1, got {res.returncode}"
        )
        self.assertIn("Snapshot Surgery", res.stderr)
        self.assertIn("HF_HUB_OFFLINE=1", res.stderr)
        self.assertIn("Aborted: Confirmation 'YES' was not received.", res.stderr)

        reader = gguf.GGUFReader(snap_dir / "model.gguf", "r")
        self.assertEqual(
            reader.get_field("tokenizer.chat_template").contents(),
            "unmodified pre-patch",
        )
        del reader

    def test_snapshot_surgery_interactive_eof_aborts_cleanly(self):
        """PASS: Closed stdin during Snapshot Surgery confirmation aborts cleanly with exit code 1 and leaves file unmodified; FAIL: Hangs or returns 0."""
        commit = "e" * 40
        self.create_synthetic_hf_repo(
            "testorg/consent-eof", [{"commit_hash": commit, "files": {}}]
        )
        snap_dir = (
            self.hf_cache_dir / "models--testorg--consent-eof" / "snapshots" / commit
        )
        self.create_synthetic_gguf(
            "model.gguf", template="unmodified pre-patch", target_dir=snap_dir
        )

        res = self.run_script(
            "testorg/consent-eof",
            input="",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assertEqual(
            res.returncode, 1, msg=f"Expected exit code 1, got {res.returncode}"
        )
        self.assertIn("Snapshot Surgery", res.stderr)
        self.assertIn("HF_HUB_OFFLINE=1", res.stderr)
        self.assertIn(
            "Standard input was closed without confirmation. Use --force", res.stderr
        )

        reader = gguf.GGUFReader(snap_dir / "model.gguf", "r")
        self.assertEqual(
            reader.get_field("tokenizer.chat_template").contents(),
            "unmodified pre-patch",
        )
        del reader

    def test_snapshot_surgery_direct_cache_path(self):
        """PASS: Specifying a direct filesystem path to a cached GGUF file invokes Snapshot Surgery warning and consent; FAIL: Skips warning or prompt."""
        commit = "f" * 40
        self.create_synthetic_hf_repo(
            "testorg/direct-path", [{"commit_hash": commit, "files": {}}]
        )
        snap_dir = (
            self.hf_cache_dir / "models--testorg--direct-path" / "snapshots" / commit
        )
        gguf_path, _ = self.create_synthetic_gguf("model.gguf", target_dir=snap_dir)

        # 1. Decline
        res_no = self.run_script(
            str(gguf_path),
            input="NO\n",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assertEqual(res_no.returncode, 1)
        self.assertIn("Snapshot Surgery", res_no.stderr)
        self.assertIn("HF_HUB_OFFLINE=1", res_no.stderr)

        # 2. Confirm
        res_yes = self.run_script(
            str(gguf_path),
            input="YES\n",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assertEqual(res_yes.returncode, 0)
        self.assertIn("Snapshot Surgery", res_yes.stderr)
        self.assertIn("HF_HUB_OFFLINE=1", res_yes.stderr)
        self.assertTrue(install.verify_gguf(gguf_path, SOURCE_TEMPLATE))

    def test_dist_baseline_storage_and_immutability_directory(self):
        """PASS: First directory install creates .dist baseline backups; second install preserves .dist immutably and creates versioned backup when upgrading version; FAIL: Overwrites .dist or fails to create versioned backup."""
        model_dir = self.test_dir / "dist_immutability_dir"
        model_dir.mkdir()

        orig_jinja = "original stock jinja template"
        orig_config = {
            "bos_token": "<|im_start|>",
            "chat_template": "original stock config template",
        }
        (model_dir / "chat_template.jinja").write_text(orig_jinja)
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps(orig_config, indent=2) + "\n"
        )

        # 1. Initial install with repository template (v22.5.1)
        res1 = self.run_script(str(model_dir))
        self.assertEqual(
            res1.returncode, 0, msg=f"Initial install failed: {res1.stderr}"
        )

        dist_jinja = model_dir / "chat_template.jinja.dist"
        dist_config = model_dir / "tokenizer_config.json.dist"
        bak_jinja = model_dir / "chat_template.jinja.bak"
        bak_config = model_dir / "tokenizer_config.json.bak"

        self.assertTrue(
            dist_jinja.is_file(),
            msg="chat_template.jinja.dist not created on first install.",
        )
        self.assertTrue(
            dist_config.is_file(),
            msg="tokenizer_config.json.dist not created on first install.",
        )
        self.assertEqual(
            dist_jinja.read_text(),
            orig_jinja,
            msg=".dist jinja content mismatch with original.",
        )
        self.assertEqual(
            json.loads(dist_config.read_text()),
            orig_config,
            msg=".dist config mismatch with original.",
        )

        self.assertTrue(
            bak_jinja.is_file(), msg="Legacy chat_template.jinja.bak not created."
        )
        self.assertTrue(
            bak_config.is_file(), msg="Legacy tokenizer_config.json.bak not created."
        )

        # 2. Re-install identical template: .dist must not be overwritten, no versioned backups
        res2 = self.run_script(str(model_dir))
        self.assertEqual(res2.returncode, 0, msg=f"Re-install failed: {res2.stderr}")
        self.assertEqual(
            dist_jinja.read_text(),
            orig_jinja,
            msg=".dist jinja was modified on re-install.",
        )
        self.assertEqual(
            json.loads(dist_config.read_text()),
            orig_config,
            msg=".dist config was modified on re-install.",
        )

        versioned_baks_run2 = list(model_dir.glob("*.qwen3.8-honed-*.bak"))
        self.assertEqual(
            versioned_baks_run2,
            [],
            msg=f"Unexpected versioned backups created on identical re-install: {versioned_baks_run2}",
        )

        # 3. Upgrade to custom version B
        custom_tpl = self.test_dir / "custom_v22_6.jinja"
        custom_content = '{%- set template_version = "qwen3.8-honed-v22.6.0" %}\nTurn 1: {{ messages[0].content }}'
        custom_tpl.write_text(custom_content)

        install.patch_directory(model_dir, custom_tpl)

        # Baseline .dist must STILL be pristine
        self.assertEqual(
            dist_jinja.read_text(),
            orig_jinja,
            msg=".dist jinja was overwritten during upgrade.",
        )
        self.assertEqual(
            json.loads(dist_config.read_text()),
            orig_config,
            msg=".dist config was overwritten during upgrade.",
        )

        # Versioned backup for previous installed version (v22.5.1) must exist
        ver_jinja_bak = model_dir / "chat_template.jinja.qwen3.8-honed-v22.5.1.bak"
        ver_config_bak = model_dir / "tokenizer_config.json.qwen3.8-honed-v22.5.1.bak"
        self.assertTrue(
            ver_jinja_bak.is_file(),
            msg="Versioned backup for chat_template.jinja was not created.",
        )
        self.assertTrue(
            ver_config_bak.is_file(),
            msg="Versioned backup for tokenizer_config.json was not created.",
        )

        # Active template is updated to custom version
        self.assertEqual(
            (model_dir / "chat_template.jinja").read_text(), custom_content
        )

    def test_dist_baseline_directory_without_prior_jinja(self):
        """PASS: Model directory without prior jinja creates only tokenizer_config.json.dist; subsequent upgrades never create chat_template.jinja.dist; FAIL: Creates chat_template.jinja.dist when no jinja originally existed."""
        model_dir = self.test_dir / "dist_no_prior_jinja"
        model_dir.mkdir()

        orig_config = {"model_type": "qwen2", "chat_template": "stock config"}
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps(orig_config, indent=2) + "\n"
        )

        # 1. Initial install
        res1 = self.run_script(str(model_dir))
        self.assertEqual(res1.returncode, 0)

        dist_config = model_dir / "tokenizer_config.json.dist"
        dist_jinja = model_dir / "chat_template.jinja.dist"
        self.assertTrue(
            dist_config.is_file(), msg="tokenizer_config.json.dist not created."
        )
        self.assertFalse(
            dist_jinja.exists(),
            msg="chat_template.jinja.dist was created when no jinja originally existed.",
        )

        # 2. Upgrade to version B
        custom_tpl = self.test_dir / "custom_v22_6_no_prior.jinja"
        custom_tpl.write_text(
            '{%- set template_version = "qwen3.8-honed-v22.6.0" %}\nTest'
        )

        install.patch_directory(model_dir, custom_tpl)

        # Baseline check
        self.assertTrue(dist_config.is_file())
        self.assertEqual(json.loads(dist_config.read_text()), orig_config)
        self.assertFalse(
            dist_jinja.exists(),
            msg="chat_template.jinja.dist was created on upgrade when no jinja originally existed.",
        )

        # Versioned backup check
        ver_jinja_bak = model_dir / "chat_template.jinja.qwen3.8-honed-v22.5.1.bak"
        ver_config_bak = model_dir / "tokenizer_config.json.qwen3.8-honed-v22.5.1.bak"
        self.assertTrue(ver_jinja_bak.is_file())
        self.assertTrue(ver_config_bak.is_file())

    def test_gguf_dist_preservation_and_versioned_backups(self):
        """PASS: GGUF patch establishes tokenizer.chat_template.dist; subsequent version updates retain .dist immutably and add versioned backup keys; FAIL: Overwrites .dist or fails to create versioned backup key."""
        gguf_path, orig_arr = self.create_synthetic_gguf(
            "gguf_dist_test.gguf", template="stock gguf chat template", alignment=32
        )

        tpl_v1 = '{%- set template_version = "qwen3.8-honed-v22.4.0" %}\nTurn: {{ messages[0].content }}'
        tpl_v2 = '{%- set template_version = "qwen3.8-honed-v22.5.0" %}\nTurn: {{ messages[0].content }}'
        tpl_v3 = '{%- set template_version = "qwen3.8-honed-v22.6.0" %}\nTurn: {{ messages[0].content }}'

        # 1. First patch (v1)
        res1 = install.patch_gguf(gguf_path, tpl_v1, force=True)
        self.assertTrue(res1)
        self.assertEqual(
            install.extract_gguf_dist(gguf_path), "stock gguf chat template"
        )
        self.assertEqual(
            install.extract_gguf_backup(gguf_path), "stock gguf chat template"
        )
        self.assertEqual(install.extract_gguf_chat_template(gguf_path), tpl_v1)

        # 2. Re-patch identical template (v1)
        res2 = install.patch_gguf(gguf_path, tpl_v1, force=True)
        self.assertTrue(res2)
        self.assertEqual(
            install.extract_gguf_dist(gguf_path), "stock gguf chat template"
        )
        reader2 = gguf.GGUFReader(gguf_path, "r")
        versioned_keys_r2 = [
            k
            for k in reader2.fields.keys()
            if k.startswith("tokenizer.chat_template.") and k.endswith(".bak")
        ]
        self.assertEqual(
            versioned_keys_r2,
            [],
            msg=f"Unexpected versioned backup key on identical patch: {versioned_keys_r2}",
        )
        del reader2

        # 3. Upgrade to v2
        res3 = install.patch_gguf(gguf_path, tpl_v2, force=True)
        self.assertTrue(res3)
        self.assertEqual(
            install.extract_gguf_dist(gguf_path),
            "stock gguf chat template",
            msg="GGUF .dist was overwritten.",
        )
        self.assertEqual(
            install.extract_gguf_backup(gguf_path),
            tpl_v1,
            msg="Legacy .backup does not point to replaced version.",
        )
        self.assertEqual(install.extract_gguf_chat_template(gguf_path), tpl_v2)

        reader3 = gguf.GGUFReader(gguf_path, "r")
        bak_v1_field = reader3.get_field(
            "tokenizer.chat_template.qwen3.8-honed-v22.4.0.bak"
        )
        self.assertIsNotNone(
            bak_v1_field,
            msg="Versioned backup key tokenizer.chat_template.qwen3.8-honed-v22.4.0.bak not found.",
        )
        self.assertEqual(bak_v1_field.contents(), tpl_v1)
        del reader3

        # 4. Upgrade to v3
        res4 = install.patch_gguf(gguf_path, tpl_v3, force=True)
        self.assertTrue(res4)
        self.assertEqual(
            install.extract_gguf_dist(gguf_path),
            "stock gguf chat template",
            msg="GGUF .dist was overwritten on second upgrade.",
        )
        self.assertEqual(install.extract_gguf_backup(gguf_path), tpl_v2)
        self.assertEqual(install.extract_gguf_chat_template(gguf_path), tpl_v3)

        reader4 = gguf.GGUFReader(gguf_path, "r")
        # Both v1 and v2 backups must be retained in GGUF metadata
        self.assertIsNotNone(
            reader4.get_field("tokenizer.chat_template.qwen3.8-honed-v22.4.0.bak")
        )
        self.assertIsNotNone(
            reader4.get_field("tokenizer.chat_template.qwen3.8-honed-v22.5.0.bak")
        )
        self.assertEqual(
            reader4.get_field(
                "tokenizer.chat_template.qwen3.8-honed-v22.5.0.bak"
            ).contents(),
            tpl_v2,
        )

        # Tensor verification
        tensor = reader4.get_tensor(0)
        self.assertTrue(
            np.array_equal(tensor.data, orig_arr),
            msg="Tensor weights corrupted during multiple GGUF upgrades.",
        )
        del reader4

    def test_helpers_ensure_dist_and_create_versioned_backup(self):
        """PASS: ensure_dist_backup creates .dist and preserves existing; create_versioned_backup sanitizes versions and creates versioned + .bak copies; FAIL: Logic or sanitization error."""
        test_file = self.test_dir / "helper_test.txt"
        test_file.write_text("initial content")

        # ensure_dist_backup creates .dist
        dist = install.ensure_dist_backup(test_file)
        self.assertTrue(dist.is_file())
        self.assertEqual(dist.read_text(), "initial content")

        # ensure_dist_backup does not overwrite existing .dist
        test_file.write_text("modified content")
        dist_again = install.ensure_dist_backup(test_file)
        self.assertEqual(dist_again.read_text(), "initial content")

        # create_versioned_backup creates sanitized version backup and updates .bak
        ver_bak = install.create_versioned_backup(test_file, "v1.0.0 / beta")
        expected_ver_bak = self.test_dir / "helper_test.txt.v1.0.0_beta.bak"
        std_bak = self.test_dir / "helper_test.txt.bak"

        self.assertEqual(ver_bak, expected_ver_bak)
        self.assertTrue(expected_ver_bak.is_file())
        self.assertEqual(expected_ver_bak.read_text(), "modified content")
        self.assertTrue(std_bak.is_file())
        self.assertEqual(std_bak.read_text(), "modified content")

    def test_get_backup_versions_helpers_and_prompt_choice(self):
        """PASS: get_directory_backup_versions and get_gguf_backup_versions discover versioned and dist backups; prompt_downgrade_choice handles numbers, aborts, and EOF; FAIL: Misses versions or prompt choice fails."""
        model_dir = self.test_dir / "helpers_backup_dir"
        model_dir.mkdir()

        (model_dir / "tokenizer_config.json.dist").write_text("dist config")
        (model_dir / "chat_template.jinja.dist").write_text("dist jinja")
        (model_dir / "tokenizer_config.json.v22_4_0.bak").write_text("v22.4 config")
        (model_dir / "chat_template.jinja.v22_4_0.bak").write_text("v22.4 jinja")
        (model_dir / "tokenizer_config.json.v22_5_0.bak").write_text("v22.5 config")

        dir_versions = install.get_directory_backup_versions(model_dir)
        version_names = [v[0] for v in dir_versions]
        self.assertIn("v22_4_0", version_names)
        self.assertIn("v22_5_0", version_names)
        self.assertIn("dist", version_names)

        # GGUF helper test
        gguf_path, _ = self.create_synthetic_gguf("helpers_test.gguf", template="stock")
        install.patch_gguf(
            gguf_path, '{%- set template_version = "v1" %}\nt1', force=True
        )
        install.patch_gguf(
            gguf_path, '{%- set template_version = "v2" %}\nt2', force=True
        )

        gguf_versions = install.get_gguf_backup_versions(gguf_path)
        self.assertIn("v1", gguf_versions)
        self.assertIn("dist", gguf_versions)

        # prompt_downgrade_choice test with mock input
        with mock.patch("builtins.input", return_value="1"):
            choice1 = install.prompt_downgrade_choice(["v22_4_0", "dist"])
            self.assertEqual(choice1, "v22_4_0")

        with mock.patch("builtins.input", return_value="2"):
            choice2 = install.prompt_downgrade_choice(["v22_4_0", "dist"])
            self.assertEqual(choice2, "dist")

        with mock.patch("builtins.input", return_value="q"):
            choice_q = install.prompt_downgrade_choice(["v22_4_0", "dist"])
            self.assertIsNone(choice_q)

        with mock.patch("builtins.input", side_effect=EOFError):
            choice_eof = install.prompt_downgrade_choice(["v22_4_0", "dist"])
            self.assertEqual(choice_eof, "dist")

    def test_directory_interactive_downgrade_and_complete_revert(self):
        """PASS: --uninstall interactively downgrades to previous version backup; second uninstall reverts completely to .dist; FAIL: Active template incorrect or artifacts retained."""
        model_dir = self.test_dir / "interactive_downgrade_dir"
        model_dir.mkdir()

        orig_jinja = "original baseline jinja"
        orig_config = {"chat_template": "original baseline config"}
        (model_dir / "chat_template.jinja").write_text(orig_jinja)
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps(orig_config, indent=2) + "\n"
        )

        # 1. Install version A
        tpl_a = self.test_dir / "tpl_a.jinja"
        tpl_a.write_text(
            '{%- set template_version = "qwen3.8-honed-v22.4.0" %}\nVersion A template'
        )
        install.patch_directory(model_dir, tpl_a)

        # 2. Upgrade to version B
        tpl_b = self.test_dir / "tpl_b.jinja"
        tpl_b.write_text(
            '{%- set template_version = "qwen3.8-honed-v22.5.0" %}\nVersion B template'
        )
        install.patch_directory(model_dir, tpl_b)

        # Verify active is Version B, versioned backup of A exists, and .dist exists
        self.assertIn("Version B", (model_dir / "chat_template.jinja").read_text())
        self.assertTrue(
            (model_dir / "chat_template.jinja.qwen3.8-honed-v22.4.0.bak").is_file()
        )
        self.assertTrue((model_dir / "tokenizer_config.json.dist").is_file())

        # 3. Interactive uninstall: enter "1" to downgrade to version A
        res_down = self.run_script(str(model_dir), "--uninstall", input="1\n")
        self.assertEqual(
            res_down.returncode, 0, msg=f"Downgrade failed: {res_down.stderr}"
        )
        self.assertIn(
            "Successfully downgraded chat template to version 'qwen3.8-honed-v22.4.0'",
            res_down.stdout,
        )

        # Verify active is now Version A
        self.assertIn("Version A", (model_dir / "chat_template.jinja").read_text())
        # Version A backup artifact must be removed
        self.assertFalse(
            (model_dir / "chat_template.jinja.qwen3.8-honed-v22.4.0.bak").exists()
        )
        self.assertFalse(
            (model_dir / "tokenizer_config.json.qwen3.8-honed-v22.4.0.bak").exists()
        )
        # .dist files remain intact
        self.assertTrue((model_dir / "tokenizer_config.json.dist").is_file())
        self.assertTrue((model_dir / "chat_template.jinja.dist").is_file())

        # 4. Uninstall again: only .dist remains, so reverts to baseline
        res_revert = self.run_script(str(model_dir), "--uninstall")
        self.assertEqual(
            res_revert.returncode, 0, msg=f"Revert failed: {res_revert.stderr}"
        )
        self.assertIn(
            "Successfully uninstalled chat template from directory.", res_revert.stdout
        )

        # Verify original baseline restored and all backup files cleaned up
        self.assertEqual((model_dir / "chat_template.jinja").read_text(), orig_jinja)
        self.assertEqual(
            json.loads((model_dir / "tokenizer_config.json").read_text()), orig_config
        )
        self.assertFalse((model_dir / "tokenizer_config.json.dist").exists())
        self.assertFalse((model_dir / "chat_template.jinja.dist").exists())
        self.assertFalse((model_dir / "tokenizer_config.json.bak").exists())
        self.assertFalse((model_dir / "chat_template.jinja.bak").exists())

    def test_gguf_interactive_downgrade_and_complete_revert(self):
        """PASS: GGUF downgrade restores intermediate version metadata and preserves .dist; full revert restores baseline and cleans all backup keys; FAIL: Restore incorrect or metadata keys leaked."""
        gguf_path, orig_arr = self.create_synthetic_gguf(
            "interactive_gguf.gguf", template="stock gguf"
        )

        tpl_a = '{%- set template_version = "qwen3.8-honed-v22.4.0" %}\nTemplate A'
        tpl_b = '{%- set template_version = "qwen3.8-honed-v22.5.0" %}\nTemplate B'

        install.patch_gguf(gguf_path, tpl_a, force=True)
        install.patch_gguf(gguf_path, tpl_b, force=True)

        self.assertEqual(install.extract_gguf_chat_template(gguf_path), tpl_b)
        self.assertEqual(install.extract_gguf_dist(gguf_path), "stock gguf")

        # 1. Downgrade interactively: select option 1 (version A)
        res_down = self.run_script(str(gguf_path), "--uninstall", input="1\n")
        self.assertEqual(
            res_down.returncode, 0, msg=f"GGUF downgrade failed: {res_down.stderr}"
        )
        self.assertIn(
            "Downgraded 'tokenizer.chat_template' to version 'qwen3.8-honed-v22.4.0'",
            res_down.stdout,
        )

        self.assertEqual(install.extract_gguf_chat_template(gguf_path), tpl_a)
        self.assertEqual(
            install.extract_gguf_dist(gguf_path),
            "stock gguf",
            msg="GGUF .dist was removed or modified during downgrade.",
        )
        self.assertIsNone(
            install.extract_gguf_metadata(
                gguf_path, "tokenizer.chat_template.qwen3.8-honed-v22.4.0.bak"
            )
        )

        # Verify tensor weights remain intact after downgrade
        reader = gguf.GGUFReader(gguf_path, "r")
        self.assertTrue(np.array_equal(reader.get_tensor(0).data, orig_arr))
        del reader

        # 2. Revert completely to baseline
        res_revert = self.run_script(str(gguf_path), "--uninstall", "--force")
        self.assertEqual(
            res_revert.returncode,
            0,
            msg=f"GGUF full revert failed: {res_revert.stderr}",
        )
        self.assertEqual(install.extract_gguf_chat_template(gguf_path), "stock gguf")
        self.assertIsNone(install.extract_gguf_dist(gguf_path))
        self.assertIsNone(install.extract_gguf_backup(gguf_path))

        reader2 = gguf.GGUFReader(gguf_path, "r")
        self.assertTrue(np.array_equal(reader2.get_tensor(0).data, orig_arr))
        del reader2

    def test_directory_legacy_backup_migration_and_uninstall(self):
        """PASS: Directory with legacy .bak backfills .dist on install or restores cleanly on uninstall; FAIL: Baseline lost or restore fails."""
        # Case A: Legacy backup backfilled on patch_directory
        model_dir_a = self.test_dir / "legacy_backfill_dir"
        model_dir_a.mkdir()
        orig_config = {"chat_template": "stock old config"}
        patched_config = {"chat_template": "already patched config"}
        (model_dir_a / "tokenizer_config.json").write_text(
            json.dumps(patched_config, indent=2)
        )
        (model_dir_a / "tokenizer_config.json.bak").write_text(
            json.dumps(orig_config, indent=2)
        )

        tpl_new = self.test_dir / "tpl_new.jinja"
        tpl_new.write_text(
            '{%- set template_version = "qwen3.8-honed-v22.6.0" %}\nNew Template'
        )
        install.patch_directory(model_dir_a, tpl_new)

        dist_file = model_dir_a / "tokenizer_config.json.dist"
        self.assertTrue(dist_file.is_file())
        self.assertEqual(
            json.loads(dist_file.read_text()),
            orig_config,
            msg=".dist was not backfilled from legacy .bak.",
        )

        # Case B: Legacy directory with only .bak uninstalls cleanly
        model_dir_b = self.test_dir / "legacy_uninst_dir"
        model_dir_b.mkdir()
        (model_dir_b / "tokenizer_config.json").write_text(
            json.dumps(patched_config, indent=2)
        )
        (model_dir_b / "tokenizer_config.json.bak").write_text(
            json.dumps(orig_config, indent=2)
        )

        res_uninst = self.run_script(str(model_dir_b), "--uninstall")
        self.assertEqual(res_uninst.returncode, 0)
        self.assertEqual(
            json.loads((model_dir_b / "tokenizer_config.json").read_text()), orig_config
        )
        self.assertFalse((model_dir_b / "tokenizer_config.json.bak").exists())

    def test_directory_uninstall_non_interactive_fallback(self):
        """PASS: --uninstall --force or non-interactive on multi-version directory defaults to full revert to .dist; FAIL: Prompts or fails to revert."""
        model_dir = self.test_dir / "non_interactive_dir"
        model_dir.mkdir()

        orig_config = {"chat_template": "original baseline"}
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps(orig_config, indent=2)
        )

        # First install
        install.patch_directory(model_dir, SOURCE_TEMPLATE)

        # Upgrade to custom version
        custom_tpl = self.test_dir / "custom_tpl.jinja"
        custom_tpl.write_text(
            '{%- set template_version = "qwen3.8-honed-v22.9.0" %}\nCustom'
        )
        install.patch_directory(model_dir, custom_tpl)

        # Run uninstall with --force (bypasses prompt and reverts to .dist)
        res = self.run_script(str(model_dir), "--uninstall", "--force")
        self.assertEqual(res.returncode, 0)
        self.assertEqual(
            json.loads((model_dir / "tokenizer_config.json").read_text()), orig_config
        )
        self.assertFalse((model_dir / "tokenizer_config.json.dist").exists())
        self.assertFalse(
            (model_dir / "tokenizer_config.json.qwen3.8-honed-v22.5.1.bak").exists()
        )

    def test_repeat_install_preserves_dist_and_stock_templates_cli(self):
        """PASS: Running installer multiple times via CLI preserves .dist immutably even after modifications; FAIL: Overwrites .dist on repeat runs."""
        # 1. Directory mode
        model_dir = self.test_dir / "repeat_install_cli_dir"
        model_dir.mkdir()
        orig_config = {"model_type": "qwen2", "chat_template": "stock config baseline"}
        orig_jinja = "stock jinja baseline"
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps(orig_config, indent=2) + "\n"
        )
        (model_dir / "chat_template.jinja").write_text(orig_jinja)

        # Initial install run
        res1 = self.run_script(str(model_dir))
        self.assertEqual(res1.returncode, 0)
        dist_config = model_dir / "tokenizer_config.json.dist"
        dist_jinja = model_dir / "chat_template.jinja.dist"
        self.assertTrue(dist_config.is_file())
        self.assertTrue(dist_jinja.is_file())
        self.assertEqual(json.loads(dist_config.read_text()), orig_config)
        self.assertEqual(dist_jinja.read_text(), orig_jinja)

        # Second install run (immediate repeat)
        res2 = self.run_script(str(model_dir))
        self.assertEqual(res2.returncode, 0)
        self.assertEqual(json.loads(dist_config.read_text()), orig_config)
        self.assertEqual(dist_jinja.read_text(), orig_jinja)

        # Third install run after manual modification of active files
        (model_dir / "chat_template.jinja").write_text("user modified template")
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": "user modified config"}, indent=2) + "\n"
        )
        res3 = self.run_script(str(model_dir))
        self.assertEqual(res3.returncode, 0)
        # .dist MUST still hold the original stock baseline, not the user-modified content
        self.assertEqual(json.loads(dist_config.read_text()), orig_config)
        self.assertEqual(dist_jinja.read_text(), orig_jinja)

        # 2. GGUF mode
        gguf_path, orig_arr = self.create_synthetic_gguf(
            "repeat_install.gguf", template="stock gguf baseline"
        )
        res_gguf1 = self.run_script(str(gguf_path), "--force")
        self.assertEqual(res_gguf1.returncode, 0)
        self.assertEqual(install.extract_gguf_dist(gguf_path), "stock gguf baseline")

        res_gguf2 = self.run_script(str(gguf_path), "--force")
        self.assertEqual(res_gguf2.returncode, 0)
        self.assertEqual(install.extract_gguf_dist(gguf_path), "stock gguf baseline")

        reader = gguf.GGUFReader(gguf_path, "r")
        self.assertTrue(np.array_equal(reader.get_tensor(0).data, orig_arr))
        del reader

    def test_identical_reinstall_creates_no_redundant_or_corrupted_backups(self):
        """PASS: Repeated identical installation runs create no redundant versioned backups or corrupted files; FAIL: Generates redundant backups on identical template."""
        model_dir = self.test_dir / "identical_reinstall_dir"
        model_dir.mkdir()
        orig_config = {"chat_template": "stock"}
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps(orig_config, indent=2) + "\n"
        )
        (model_dir / "chat_template.jinja").write_text("stock")

        # Run 1
        res1 = self.run_script(str(model_dir))
        self.assertEqual(res1.returncode, 0)

        # Check existing backups: only .dist and .bak
        initial_baks = list(model_dir.glob("*.bak"))
        self.assertEqual(
            len(initial_baks), 2
        )  # tokenizer_config.json.bak and chat_template.jinja.bak

        # Run 2
        res2 = self.run_script(str(model_dir))
        self.assertEqual(res2.returncode, 0)
        self.assertIn(
            "already up to date; skipping redundant backup creation", res2.stdout
        )
        baks_after_run2 = list(model_dir.glob("*.bak"))
        self.assertEqual(len(baks_after_run2), 2)
        versioned_baks_run2 = list(model_dir.glob("*.qwen3.8-honed-*.bak"))
        self.assertEqual(versioned_baks_run2, [])

        # Run 3
        res3 = self.run_script(str(model_dir))
        self.assertEqual(res3.returncode, 0)
        self.assertIn(
            "already up to date; skipping redundant backup creation", res3.stdout
        )
        baks_after_run3 = list(model_dir.glob("*.bak"))
        self.assertEqual(len(baks_after_run3), 2)
        self.assertEqual(list(model_dir.glob("*.qwen3.8-honed-*.bak")), [])

        # GGUF identical repeat
        gguf_path, orig_arr = self.create_synthetic_gguf(
            "identical_gguf.gguf", template="stock"
        )
        install.patch_gguf(gguf_path, "template content", force=True)
        install.patch_gguf(gguf_path, "template content", force=True)
        install.patch_gguf(gguf_path, "template content", force=True)

        reader = gguf.GGUFReader(gguf_path, "r")
        version_bak_keys = [
            k
            for k in reader.fields.keys()
            if k.startswith("tokenizer.chat_template.") and k.endswith(".bak")
        ]
        self.assertEqual(version_bak_keys, [])
        self.assertTrue(np.array_equal(reader.get_tensor(0).data, orig_arr))
        del reader

    def test_interactive_downgrade_prompt_options_and_target_version_flag(self):
        """PASS: Interactive prompt handles abort ('q'), selection by version name, and --target-version CLI flag for directory and GGUF; FAIL: Fails to abort or downgrade correctly."""
        # 1. Directory mode
        model_dir = self.test_dir / "target_ver_cli_dir"
        model_dir.mkdir()
        orig_config = {"chat_template": "stock config"}
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps(orig_config, indent=2) + "\n"
        )
        (model_dir / "chat_template.jinja").write_text("stock jinja")

        tpl_v1 = self.test_dir / "tpl_v1.jinja"
        tpl_v1.write_text(
            '{%- set template_version = "qwen3.8-honed-v22.4.0" %}\nVersion 1'
        )
        install.patch_directory(model_dir, tpl_v1)

        tpl_v2 = self.test_dir / "tpl_v2.jinja"
        tpl_v2.write_text(
            '{%- set template_version = "qwen3.8-honed-v22.5.0" %}\nVersion 2'
        )
        install.patch_directory(model_dir, tpl_v2)

        # Abort interactive prompt with 'q'
        res_abort = self.run_script(str(model_dir), "--uninstall", input="q\n")
        self.assertEqual(res_abort.returncode, 1)
        self.assertIn("Aborted by user.", res_abort.stderr)
        self.assertIn("Version 2", (model_dir / "chat_template.jinja").read_text())

        # Select by entering version name string interactively
        res_name = self.run_script(
            str(model_dir), "--uninstall", input="qwen3.8-honed-v22.4.0\n"
        )
        self.assertEqual(res_name.returncode, 0)
        self.assertIn(
            "Successfully downgraded chat template to version 'qwen3.8-honed-v22.4.0'",
            res_name.stdout,
        )
        self.assertIn("Version 1", (model_dir / "chat_template.jinja").read_text())

        # Re-upgrade to v2, then use --target-version CLI flag non-interactively
        install.patch_directory(model_dir, tpl_v2)
        res_flag = self.run_script(
            str(model_dir), "--uninstall", "--target-version", "qwen3.8-honed-v22.4.0"
        )
        self.assertEqual(res_flag.returncode, 0)
        self.assertIn(
            "Successfully downgraded chat template to version 'qwen3.8-honed-v22.4.0'",
            res_flag.stdout,
        )
        self.assertIn("Version 1", (model_dir / "chat_template.jinja").read_text())

        # Invalid --target-version fails
        res_bad = self.run_script(
            str(model_dir), "--uninstall", "--target-version", "nonexistent-version"
        )
        self.assert_subprocess_failure(
            res_bad,
            description="Invalid target version",
            expected_stderr="not found",
            expected_exit_code=1,
        )

        # 2. GGUF mode with --target-version
        gguf_path, orig_arr = self.create_synthetic_gguf(
            "target_ver_gguf.gguf", template="stock"
        )
        install.patch_gguf(
            gguf_path, '{%- set template_version = "v1" %}\nt1', force=True
        )
        install.patch_gguf(
            gguf_path, '{%- set template_version = "v2" %}\nt2', force=True
        )

        res_gguf_down = self.run_script(
            str(gguf_path), "--uninstall", "--target-version", "v1"
        )
        self.assertEqual(res_gguf_down.returncode, 0)
        self.assertEqual(
            install.extract_gguf_chat_template(gguf_path),
            '{%- set template_version = "v1" %}\nt1',
        )

        res_gguf_bad = self.run_script(
            str(gguf_path), "--uninstall", "--target-version", "v999"
        )
        self.assert_subprocess_failure(
            res_gguf_bad,
            description="Invalid GGUF target version",
            expected_stderr="not found",
            expected_exit_code=1,
        )

    def test_complete_uninstall_reverts_to_dist_and_removes_all_artifacts(self):
        """PASS: Complete uninstall from multi-version directory cleanly restores stock .dist, unlinks created jinja, and deletes all backup artifacts; FAIL: Artifacts remain or restore incorrect."""
        model_dir = self.test_dir / "complete_uninst_dir"
        model_dir.mkdir()
        orig_config = {"model_type": "qwen2", "chat_template": "stock config"}
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps(orig_config, indent=2) + "\n"
        )
        # Notice: No chat_template.jinja originally!

        # Install version 1
        tpl_v1 = self.test_dir / "compl_v1.jinja"
        tpl_v1.write_text('{%- set template_version = "v22.1.0" %}\nV1')
        install.patch_directory(model_dir, tpl_v1)

        # Upgrade to version 2
        tpl_v2 = self.test_dir / "compl_v2.jinja"
        tpl_v2.write_text('{%- set template_version = "v22.2.0" %}\nV2')
        install.patch_directory(model_dir, tpl_v2)

        # Verify artifacts exist before complete uninstall
        self.assertTrue((model_dir / "tokenizer_config.json.dist").is_file())
        self.assertTrue((model_dir / "tokenizer_config.json.v22.1.0.bak").is_file())
        self.assertTrue((model_dir / "chat_template.jinja").is_file())
        self.assertTrue((model_dir / "chat_template.jinja.v22.1.0.bak").is_file())

        # Complete uninstall via --uninstall --force
        res = self.run_script(str(model_dir), "--uninstall", "--force")
        self.assertEqual(res.returncode, 0)
        self.assertIn(
            "Successfully uninstalled chat template from directory.", res.stdout
        )

        # Config restored to exact stock
        self.assertEqual(
            json.loads((model_dir / "tokenizer_config.json").read_text()), orig_config
        )
        # chat_template.jinja must be REMOVED completely since no .dist existed
        self.assertFalse((model_dir / "chat_template.jinja").exists())
        # All backup files must be wiped
        remaining_files = [f.name for f in model_dir.iterdir()]
        self.assertEqual(remaining_files, ["tokenizer_config.json"])

        # Test interactive selection of "dist" directly from menu
        model_dir_dist = self.test_dir / "interactive_dist_dir"
        model_dir_dist.mkdir()
        orig_config_dist = {"chat_template": "stock config 2"}
        (model_dir_dist / "tokenizer_config.json").write_text(
            json.dumps(orig_config_dist, indent=2) + "\n"
        )
        (model_dir_dist / "chat_template.jinja").write_text("stock jinja 2")

        tpl_v1_d = self.test_dir / "dist_v1.jinja"
        tpl_v1_d.write_text('{%- set template_version = "v22.1.0" %}\nV1')
        install.patch_directory(model_dir_dist, tpl_v1_d)

        tpl_v2_d = self.test_dir / "dist_v2.jinja"
        tpl_v2_d.write_text('{%- set template_version = "v22.2.0" %}\nV2')
        install.patch_directory(model_dir_dist, tpl_v2_d)

        # Enter "dist\n" to select full revert directly
        res_inter_dist = self.run_script(
            str(model_dir_dist), "--uninstall", input="dist\n"
        )
        self.assertEqual(res_inter_dist.returncode, 0)
        self.assertEqual(
            json.loads((model_dir_dist / "tokenizer_config.json").read_text()),
            orig_config_dist,
        )
        self.assertEqual(
            (model_dir_dist / "chat_template.jinja").read_text(), "stock jinja 2"
        )
        self.assertFalse((model_dir_dist / "tokenizer_config.json.dist").exists())
        self.assertFalse((model_dir_dist / "chat_template.jinja.dist").exists())

    def test_non_interactive_uninstall_fallback_on_closed_stdin(self):
        """PASS: When standard input is closed (EOF), uninstaller catches EOFError, logs fallback warning, and executes full .dist reversion; FAIL: Raises unhandled EOFError or crashes."""
        # 1. Directory mode
        model_dir = self.test_dir / "eof_dir"
        model_dir.mkdir()
        orig_config = {"chat_template": "stock baseline"}
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps(orig_config, indent=2) + "\n"
        )

        install.patch_directory(model_dir, SOURCE_TEMPLATE)
        tpl2 = self.test_dir / "eof_tpl.jinja"
        tpl2.write_text('{%- set template_version = "v2" %}\nT2')
        install.patch_directory(model_dir, tpl2)

        # Run with input="" (immediate EOF)
        res_dir = self.run_script(str(model_dir), "--uninstall", input="")
        self.assertEqual(res_dir.returncode, 0)
        self.assertIn(
            "Standard input closed. Defaulting to complete revert to .dist baseline.",
            res_dir.stderr,
        )
        self.assertEqual(
            json.loads((model_dir / "tokenizer_config.json").read_text()), orig_config
        )
        self.assertFalse((model_dir / "tokenizer_config.json.dist").exists())

        # 2. GGUF mode
        gguf_path, orig_arr = self.create_synthetic_gguf(
            "eof_gguf.gguf", template="stock"
        )
        install.patch_gguf(
            gguf_path, '{%- set template_version = "v1" %}\nt1', force=True
        )
        install.patch_gguf(
            gguf_path, '{%- set template_version = "v2" %}\nt2', force=True
        )

        res_gguf = self.run_script(str(gguf_path), "--uninstall", input="")
        self.assertEqual(res_gguf.returncode, 0)
        self.assertIn(
            "Standard input closed. Defaulting to complete revert to .dist baseline.",
            res_gguf.stderr,
        )
        self.assertEqual(install.extract_gguf_chat_template(gguf_path), "stock")
        self.assertIsNone(install.extract_gguf_dist(gguf_path))
        self.assertIsNone(install.extract_gguf_backup(gguf_path))

        reader = gguf.GGUFReader(gguf_path, "r")
        self.assertTrue(np.array_equal(reader.get_tensor(0).data, orig_arr))
        del reader


if __name__ == "__main__":
    unittest.main()
