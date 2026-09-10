#!/usr/bin/env python3
"""Tests for install.py (Step 1 functionality)."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

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

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_script(self, *args, input: str | None = None):
        cmd = [sys.executable, str(APPLY_SCRIPT)] + list(args)
        return subprocess.run(cmd, input=input, capture_output=True, text=True)

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

    def test_gguf_temp_file_cleanup_on_write_error(self):
        """PASS: Cleans up temporary .tmp files when write error occurs during GGUF metadata update; FAIL: Leaves temporary files behind or ignores error."""
        gguf_path, _ = self.create_synthetic_gguf("fail_write.gguf", template="init")

        def failing_copy(*args, **kwargs):
            raise RuntimeError("Simulated failure during metadata copy")

        orig_copy = install.copy_with_new_metadata
        self.addCleanup(setattr, install, "copy_with_new_metadata", orig_copy)
        install.copy_with_new_metadata = failing_copy

        with assert_expected_failure(
            "GGUF patch write failure exits and cleans up temporary files",
            expected_stderr_pattern="Simulated failure during metadata copy",
        ):
            with self.assertRaises(SystemExit) as ctx:
                install.patch_gguf(gguf_path, "new template", force=True)
            self.assertEqual(
                ctx.exception.code,
                1,
                msg=f"Expected exit code 1 on write error, got {ctx.exception.code}",
            )

        # Ensure no temporary .tmp files remain
        tmp_files = list(self.test_dir.glob("*.tmp"))
        self.assertEqual(
            tmp_files,
            [],
            msg=f"Temporary files were not cleaned up after error: {tmp_files}",
        )

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


if __name__ == "__main__":
    unittest.main()
