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
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
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
        return subprocess.run(cmd, input=input, capture_output=True, text=True, env=proc_env)

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
    ):
        path = self.test_dir / filename
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
                {"tokenizer_config.json": json.dumps({"chat_template": "old_template_content"})},
            )
            mtime = rev.get("last_modified", 1000.0)
            for fname, fcontent in files.items():
                fpath = snap_dir / fname
                fpath.parent.mkdir(parents=True, exist_ok=True)
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
        self.assertEqual(res.returncode, 0, msg=f"Script failed with code {res.returncode}: {res.stderr}")
        self.assertIn("Successfully applied chat template to directory.", res.stdout, msg="Expected success message not found in stdout.")

        # Check backups created
        bak_template = model_dir / "chat_template.jinja.bak"
        bak_config = model_dir / "tokenizer_config.json.bak"
        self.assertTrue(bak_template.is_file(), msg="chat_template.jinja.bak was not created.")
        self.assertTrue(bak_config.is_file(), msg="tokenizer_config.json.bak was not created.")
        self.assertEqual(bak_template.read_text(), orig_template, msg="chat_template.jinja.bak content mismatch with original template.")
        self.assertEqual(json.loads(bak_config.read_text()), initial_config, msg="tokenizer_config.json.bak content mismatch with initial config.")

        # Check chat_template.jinja updated
        expected_template = SOURCE_TEMPLATE.read_text(encoding="utf-8")
        self.assertEqual(
            (model_dir / "chat_template.jinja").read_text(encoding="utf-8"),
            expected_template,
            msg="chat_template.jinja was not updated to repository template.",
        )

        # Check tokenizer_config.json updated and formatted
        updated_config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(updated_config["bos_token"], "<|im_start|>", msg="bos_token was corrupted or missing.")
        self.assertEqual(updated_config["eos_token"], "<|im_end|>", msg="eos_token was corrupted or missing.")
        expected_minified = minify_jinja(expected_template)
        self.assertEqual(updated_config["chat_template"], expected_minified, msg="chat_template in tokenizer_config.json does not match minified template.")

        # Ensure valid JSON formatting with clean indent
        raw_config = config_path.read_text(encoding="utf-8")
        self.assertTrue(raw_config.endswith("\n"), msg="tokenizer_config.json does not end with trailing newline.")

    def test_directory_patch_success_without_prior_jinja(self):
        """PASS: Successfully patches directory lacking prior chat_template.jinja; FAIL: Fails to create jinja file or update config."""
        model_dir = self.test_dir / "model_no_jinja"
        model_dir.mkdir()

        initial_config = {"model_max_length": 8192}
        config_path = model_dir / "tokenizer_config.json"
        config_path.write_text(json.dumps(initial_config, indent=2) + "\n")

        res = self.run_script(str(model_dir))
        self.assertEqual(res.returncode, 0, msg=f"Script failed with code {res.returncode}: {res.stderr}")

        # Config backup exists, but template backup does not
        self.assertTrue((model_dir / "tokenizer_config.json.bak").is_file(), msg="tokenizer_config.json.bak was not created.")
        self.assertFalse((model_dir / "chat_template.jinja.bak").exists(), msg="chat_template.jinja.bak created unexpectedly when no prior jinja file existed.")

        # chat_template.jinja created
        expected_template = SOURCE_TEMPLATE.read_text(encoding="utf-8")
        self.assertEqual(
            (model_dir / "chat_template.jinja").read_text(encoding="utf-8"),
            expected_template,
            msg="chat_template.jinja was not created with expected repository template content.",
        )

        # tokenizer_config updated
        updated_config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(updated_config["model_max_length"], 8192, msg="Existing config properties were not preserved.")
        self.assertEqual(
            updated_config["chat_template"], minify_jinja(expected_template),
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
        self.assertEqual(res.returncode, 0, msg=f"Script failed with code {res.returncode}: {res.stderr}")
        self.assertIn("Updated 'tokenizer.chat_template'", res.stdout, msg="stdout missing update message for tokenizer.chat_template.")
        self.assertIn("Successfully applied chat template to GGUF.", res.stdout, msg="stdout missing success message for GGUF patch.")

        # Inspect updated GGUF file
        reader = gguf.GGUFReader(gguf_path, "r")
        chat_template_field = reader.get_field("tokenizer.chat_template")
        self.assertIsNotNone(chat_template_field, msg="tokenizer.chat_template field not found in patched GGUF.")

        expected_template = SOURCE_TEMPLATE.read_text(encoding="utf-8")
        expected_minified = minify_jinja(expected_template)
        self.assertEqual(chat_template_field.contents(), expected_minified, msg="Patched GGUF template does not match minified repository template.")

        # Verify architecture and tensor preservation
        arch_field = reader.get_field("general.architecture")
        self.assertIsNotNone(arch_field, msg="general.architecture field missing from patched GGUF.")
        self.assertEqual(arch_field.contents(), "qwen2", msg="Architecture was not preserved in patched GGUF.")

        self.assertEqual(len(reader.tensors), 1, msg=f"Expected 1 tensor in patched GGUF, found {len(reader.tensors)}.")
        tensor = reader.get_tensor(0)
        self.assertEqual(tensor.name, "model.layers.0.weight", msg="Tensor name was altered during GGUF metadata patching.")
        self.assertTrue(np.array_equal(tensor.data, orig_arr), msg="Tensor weights were altered during GGUF metadata patching.")

        # Verify alignment preservation
        self.assertEqual(reader.alignment, 64, msg=f"GGUF alignment was changed from 64 to {reader.alignment}.")

        del reader
        # Verify no temporary files remain
        tmp_files = list(self.test_dir.glob("*.tmp"))
        self.assertEqual(tmp_files, [], msg=f"Temporary files were left behind in test directory: {tmp_files}.")

    def test_gguf_interactive_confirmation_yes(self):
        """PASS: Interactive GGUF patch succeeds when user confirms with YES; FAIL: Fails to patch on confirmation or template unchanged."""
        gguf_path, _ = self.create_synthetic_gguf("confirm_yes.gguf", template="before patch")

        res = self.run_script(str(gguf_path), input="YES\n")
        self.assertEqual(res.returncode, 0, msg=f"Script failed with code {res.returncode}: {res.stderr}")
        self.assertIn("Successfully applied chat template to GGUF.", res.stdout, msg="stdout missing GGUF patch success message.")

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
        gguf_path, _ = self.create_synthetic_gguf("confirm_no.gguf", template="before patch")

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
        print("  [PASS - Expected Failure] Confirmation declined by user; GGUF patch aborted")

    def test_gguf_non_interactive_without_force_fails(self):
        """PASS: Rejects unconfirmed GGUF patch when stdin is closed and --force is not provided; FAIL: Modifies GGUF file without confirmation."""
        gguf_path, _ = self.create_synthetic_gguf("confirm_eof.gguf", template="before patch")

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
        gguf_path, orig_arr = self.create_synthetic_gguf("model_raw_bin", template="raw")

        res = self.run_script(str(gguf_path), "--force")
        self.assertEqual(res.returncode, 0, msg=f"Script failed on extensionless GGUF: {res.stderr}")

        reader = gguf.GGUFReader(gguf_path, "r")
        expected_minified = minify_jinja(SOURCE_TEMPLATE.read_text(encoding="utf-8"))
        self.assertEqual(
            reader.get_field("tokenizer.chat_template").contents(),
            expected_minified,
            msg="GGUF chat template was not updated on extensionless file.",
        )
        tensor = reader.get_tensor(0)
        self.assertTrue(np.array_equal(tensor.data, orig_arr), msg="Tensor data corrupted on extensionless GGUF file.")
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
        res = install.gguf_set_metadata(gguf_path, "tokenizer.chat_template", large_template)
        self.assertTrue(res, msg="gguf_set_metadata returned False on expansion.")

        reader = gguf.GGUFReader(gguf_path, "r")
        self.assertEqual(
            reader.get_field("tokenizer.chat_template").contents(),
            large_template,
            msg="Expanded chat template was not properly saved.",
        )
        self.assertEqual(reader.alignment, 64)
        tensor = reader.get_tensor(0)
        self.assertTrue(np.array_equal(tensor.data, orig_arr), msg="Tensor weights corrupted during expansion.")
        del reader

        # 2. Header shrinkage test (contracting metadata)
        short_template = "tiny"
        res = install.gguf_set_metadata(gguf_path, "tokenizer.chat_template", short_template)
        self.assertTrue(res, msg="gguf_set_metadata returned False on shrinkage.")

        reader = gguf.GGUFReader(gguf_path, "r")
        self.assertEqual(
            reader.get_field("tokenizer.chat_template").contents(),
            short_template,
            msg="Shrunk chat template was not properly saved.",
        )
        self.assertEqual(reader.alignment, 64)
        tensor = reader.get_tensor(0)
        self.assertTrue(np.array_equal(tensor.data, orig_arr), msg="Tensor weights corrupted during shrinkage.")
        del reader

        # 3. Add new key and remove key test
        res = install.gguf_set_metadata(
            gguf_path,
            {"custom.test_key": "custom_val"},
            removals=["tokenizer.chat_template"],
        )
        self.assertTrue(res, msg="gguf_set_metadata returned False on add/removal.")

        reader = gguf.GGUFReader(gguf_path, "r")
        self.assertIsNone(reader.get_field("tokenizer.chat_template"), msg="Removed key was still found.")
        self.assertEqual(
            reader.get_field("custom.test_key").contents(),
            "custom_val",
            msg="Newly added key was not found or has incorrect value.",
        )
        tensor = reader.get_tensor(0)
        self.assertTrue(np.array_equal(tensor.data, orig_arr), msg="Tensor weights corrupted during add/removal.")
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
        self.assertIsNotNone(backup_field, msg="tokenizer.chat_template.backup field was not created.")
        self.assertEqual(
            backup_field.contents(),
            "original pre-patch template",
            msg="tokenizer.chat_template.backup does not contain original pre-patch template.",
        )
        tensor = reader.get_tensor(0)
        self.assertTrue(np.array_equal(tensor.data, orig_arr), msg="Tensor weights corrupted after patch_gguf.")
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
        self.assertEqual(install.extract_gguf_backup(gguf_path), "pristine stock template")
        self.assertEqual(install.extract_gguf_chat_template(gguf_path), "modified template")

        # Restore backup
        restore_res = install.restore_gguf_backup(gguf_path)
        self.assertTrue(restore_res, msg="restore_gguf_backup returned False.")

        # Verify chat_template is restored and backup key is removed
        self.assertEqual(install.extract_gguf_chat_template(gguf_path), "pristine stock template")
        self.assertIsNone(install.extract_gguf_backup(gguf_path))

        reader = gguf.GGUFReader(gguf_path, "r")
        self.assertIsNone(reader.get_field("tokenizer.chat_template.backup"))
        tensor = reader.get_tensor(0)
        self.assertTrue(np.array_equal(tensor.data, orig_arr), msg="Tensor weights corrupted after restore.")
        del reader

    def test_directory_uninstall_with_existing_template(self):
        """PASS: --uninstall restores tokenizer_config.json and chat_template.jinja from backups and removes .bak files; FAIL: Fails to restore original files or retain backups."""
        model_dir = self.test_dir / "uninst_with_jinja"
        model_dir.mkdir()
        orig_jinja = "original stock jinja"
        orig_config = {"bos_token": "<|im_start|>", "chat_template": "original stock template"}
        (model_dir / "chat_template.jinja").write_text(orig_jinja)
        (model_dir / "tokenizer_config.json").write_text(json.dumps(orig_config, indent=2) + "\n")

        # Install
        patch_res = self.run_script(str(model_dir))
        self.assertEqual(patch_res.returncode, 0)
        self.assertTrue((model_dir / "tokenizer_config.json.bak").is_file())
        self.assertTrue((model_dir / "chat_template.jinja.bak").is_file())

        # Uninstall
        uninst_res = self.run_script(str(model_dir), "--uninstall")
        self.assertEqual(uninst_res.returncode, 0, msg=f"Uninstall failed: {uninst_res.stderr}")
        self.assertIn("Successfully uninstalled chat template from directory.", uninst_res.stdout)
        self.assertFalse((model_dir / "tokenizer_config.json.bak").exists())
        self.assertFalse((model_dir / "chat_template.jinja.bak").exists())
        self.assertEqual((model_dir / "chat_template.jinja").read_text(), orig_jinja)
        self.assertEqual(json.loads((model_dir / "tokenizer_config.json").read_text()), orig_config)

    def test_directory_uninstall_without_prior_jinja(self):
        """PASS: --uninstall restores tokenizer_config.json and deletes newly created chat_template.jinja; FAIL: Leaves newly created jinja file or fails to restore config."""
        model_dir = self.test_dir / "uninst_no_prior_jinja"
        model_dir.mkdir()
        orig_config = {"model_max_length": 8192, "chat_template": "stock config template"}
        (model_dir / "tokenizer_config.json").write_text(json.dumps(orig_config, indent=2) + "\n")

        # Install
        patch_res = self.run_script(str(model_dir))
        self.assertEqual(patch_res.returncode, 0)
        self.assertTrue((model_dir / "tokenizer_config.json.bak").is_file())
        self.assertTrue((model_dir / "chat_template.jinja").is_file())
        self.assertFalse((model_dir / "chat_template.jinja.bak").exists())

        # Uninstall
        uninst_res = self.run_script(str(model_dir), "--uninstall")
        self.assertEqual(uninst_res.returncode, 0, msg=f"Uninstall failed: {uninst_res.stderr}")
        self.assertIn("Successfully uninstalled chat template from directory.", uninst_res.stdout)
        self.assertFalse((model_dir / "tokenizer_config.json.bak").exists())
        self.assertFalse((model_dir / "chat_template.jinja").exists(), msg="Newly created chat_template.jinja was not deleted upon uninstall.")
        self.assertEqual(json.loads((model_dir / "tokenizer_config.json").read_text()), orig_config)

    def test_directory_uninstall_no_backups_fails(self):
        """PASS: CLI rejects uninstall on directory with no backups; FAIL: Exits with 0 or alters directory."""
        model_dir = self.test_dir / "uninst_no_backups"
        model_dir.mkdir()
        (model_dir / "tokenizer_config.json").write_text(json.dumps({"chat_template": "curr"}))

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
        self.assertEqual(install.extract_gguf_backup(gguf_path), "stock pre-patch gguf template")

        # Uninstall GGUF
        uninst_res = self.run_script(str(gguf_path), "--uninstall")
        self.assertEqual(uninst_res.returncode, 0, msg=f"Uninstall GGUF failed: {uninst_res.stderr}")
        self.assertIn("Successfully uninstalled chat template from GGUF.", uninst_res.stdout)

        # Verify chat_template is restored and backup key is removed
        self.assertEqual(install.extract_gguf_chat_template(gguf_path), "stock pre-patch gguf template")
        self.assertIsNone(install.extract_gguf_backup(gguf_path))

        reader = gguf.GGUFReader(gguf_path, "r")
        self.assertIsNone(reader.get_field("tokenizer.chat_template.backup"))
        tensor = reader.get_tensor(0)
        self.assertTrue(np.array_equal(tensor.data, orig_arr), msg="Tensor weights corrupted after GGUF uninstall.")
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
        self.assertEqual(install.extract_gguf_chat_template(gguf_path), "stock template")

    def test_verify_directory_success(self):
        """PASS: verify_directory returns True when directory template files match repository template; FAIL: Returns False for matching templates."""
        model_dir = self.test_dir / "verify_dir_ok"
        model_dir.mkdir()
        source_content = SOURCE_TEMPLATE.read_text(encoding="utf-8")
        (model_dir / "chat_template.jinja").write_text(source_content)
        config_path = model_dir / "tokenizer_config.json"
        config_path.write_text(json.dumps({"chat_template": minify_jinja(source_content)}))

        self.assertTrue(
            install.verify_directory(model_dir),
            msg="verify_directory returned False for a correctly patched model directory.",
        )

    def test_verify_directory_diagnostics_output(self):
        """PASS: Verification diagnostic probes output detailed status banners during directory patching; FAIL: Output missing required diagnostic probe lines."""
        model_dir = self.test_dir / "verify_diagnostics"
        model_dir.mkdir()
        (model_dir / "chat_template.jinja").write_text("dummy")
        (model_dir / "tokenizer_config.json").write_text(json.dumps({"chat_template": "dummy"}))

        res = self.run_script(str(model_dir))
        self.assertEqual(res.returncode, 0, msg=f"Script failed with code {res.returncode}: {res.stderr}")
        self.assertIn("=== Verifying Directory Chat Templates ===", res.stdout, msg="Diagnostic output missing header.")
        self.assertIn("Version detected:", res.stdout, msg="Diagnostic output missing version detection report.")
        self.assertIn("Exact content match:", res.stdout, msg="Diagnostic output missing content match report.")
        self.assertIn("Terseness prompt:", res.stdout, msg="Diagnostic output missing terseness prompt probe.")
        self.assertIn("Keeps system prompt:", res.stdout, msg="Diagnostic output missing system prompt probe.")
        self.assertIn("Retains thinking:", res.stdout, msg="Diagnostic output missing thinking probe.")
        self.assertIn("Equivalence check passed", res.stdout, msg="Diagnostic output missing equivalence confirmation.")

    def test_verify_directory_source_mismatch_fails(self):
        """PASS: verify_directory returns False when chat_template.jinja and tokenizer_config.json disagree; FAIL: Returns True despite template divergence."""
        model_dir = self.test_dir / "mismatch_dir"
        model_dir.mkdir()
        source_content = SOURCE_TEMPLATE.read_text(encoding="utf-8")
        (model_dir / "chat_template.jinja").write_text(source_content)
        # Differing template in tokenizer_config.json
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": "{%- set template_version = 'qwen3.8-honed-v22.5.1' %}{{ messages[0].content }}"})
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
        bad_version_content = source_content.replace("qwen3.8-honed-v22.5.1", "qwen3.8-honed-v99.9.9")
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
        no_marker_content = source_content.replace("Never: open with preamble", "Always: be verbose")
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
        patched_gguf, _ = self.create_synthetic_gguf("patched.gguf", template=source_minified)
        self.assertTrue(
            install.verify_gguf(patched_gguf),
            msg="verify_gguf returned False for a correctly patched GGUF file.",
        )

    def test_verify_gguf_unpatched_fails(self):
        """PASS: verify_gguf returns False when GGUF file contains unpatched stock template; FAIL: Returns True for unpatched GGUF template."""
        unpatched_gguf, _ = self.create_synthetic_gguf("unpatched.gguf", template="stock template")
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
        (model_dir / "tokenizer_config.json").write_text(json.dumps({"chat_template": "old"}))

        orig_verify = install.verify_directory
        self.addCleanup(setattr, install, "verify_directory", orig_verify)
        install.verify_directory = lambda *args, **kwargs: False
        with assert_expected_failure("CLI directory verification failure exits with code 1"):
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
        self.assertIsNone(install.resolve_hf_model_path("missing/repo", cache_dir=self.hf_cache_dir))

        # Single revision returns Path
        commit = "1" * 40
        self.create_synthetic_hf_repo("testorg/single", [{"commit_hash": commit}])
        resolved = install.resolve_hf_model_path("testorg/single", cache_dir=self.hf_cache_dir)
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
        resolved_latest = install.resolve_hf_model_path("testorg/multi", latest=True, cache_dir=self.hf_cache_dir)
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
            with assert_expected_failure("resolve_hf_model_path aborts cleanly on KeyboardInterrupt", expected_stderr_pattern="Aborted by user"):
                with self.assertRaises(SystemExit) as cm:
                    install.resolve_hf_model_path("testorg/multi-kb", latest=False, cache_dir=self.hf_cache_dir)
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
            with assert_expected_failure("resolve_hf_model_path aborts cleanly on EOFError", expected_stderr_pattern="Aborted by user"):
                with self.assertRaises(SystemExit) as cm:
                    install.resolve_hf_model_path("testorg/multi-eof", latest=False, cache_dir=self.hf_cache_dir)
                self.assertEqual(cm.exception.code, 1)

    def test_hf_resolve_single_revision_auto_selection(self):
        """PASS: CLI auto-resolves single-revision model without prompt and applies template; FAIL: Prompts or fails to patch."""
        commit = "c" * 40
        self.create_synthetic_hf_repo("testorg/single-cli", [{"commit_hash": commit}])
        res = self.run_script("testorg/single-cli", env={"HF_HUB_CACHE": str(self.hf_cache_dir)})
        self.assertEqual(res.returncode, 0, msg=f"Script failed: {res.stderr}")
        self.assertIn("Successfully applied chat template to directory.", res.stdout)
        self.assertNotIn("Select revision", res.stdout)

        snap_dir = self.hf_cache_dir / "models--testorg--single-cli" / "snapshots" / commit
        self.assertTrue((snap_dir / "chat_template.jinja").is_file())
        config_data = json.loads((snap_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
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
        new_config = json.loads((snap_new / "tokenizer_config.json").read_text(encoding="utf-8"))
        self.assertIn("Never: open with preamble", new_config["chat_template"])

        # Older revision remains untouched
        self.assertFalse((snap_old / "chat_template.jinja").is_file())
        old_config = json.loads((snap_old / "tokenizer_config.json").read_text(encoding="utf-8"))
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

        repo_dir = self.hf_cache_dir / "models--testorg--multi-interactive-cli" / "snapshots"
        snap_old = repo_dir / c_old
        snap_new = repo_dir / c_new

        # Selected revision 2 (c_old) is patched
        self.assertTrue((snap_old / "chat_template.jinja").is_file())
        old_config = json.loads((snap_old / "tokenizer_config.json").read_text(encoding="utf-8"))
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

        snap_c1 = self.hf_cache_dir / "models--testorg--multi-commit-prefix" / "snapshots" / c1
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

        snap_new = self.hf_cache_dir / "models--testorg--multi-default" / "snapshots" / c_new
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
        self.assertEqual(res_apply.returncode, 0, msg=f"Apply failed: {res_apply.stderr}")

        snap_dir = self.hf_cache_dir / "models--testorg--uninstall-hf" / "snapshots" / commit
        self.assertTrue((snap_dir / "chat_template.jinja").is_file())
        self.assertTrue((snap_dir / "tokenizer_config.json.bak").is_file())

        # 2. Uninstall
        res_uninst = self.run_script(
            "testorg/uninstall-hf",
            "--uninstall",
            env={"HF_HUB_CACHE": str(self.hf_cache_dir)},
        )
        self.assertEqual(res_uninst.returncode, 0, msg=f"Uninstall failed: {res_uninst.stderr}")
        self.assertIn("Successfully uninstalled chat template from directory.", res_uninst.stdout)

        # 3. Verify clean uninstallation
        self.assertFalse((snap_dir / "chat_template.jinja").is_file())
        self.assertFalse((snap_dir / "tokenizer_config.json.bak").is_file())
        restored_config = json.loads((snap_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
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


if __name__ == "__main__":
    unittest.main()
