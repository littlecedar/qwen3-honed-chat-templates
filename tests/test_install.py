#!/usr/bin/env python3
"""Tests for install.py (Step 1 functionality)."""

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
from scripts.minify_jinja import minify_jinja


class TestApplyChatTemplate(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_script(self, *args, input: str | None = None):
        cmd = [sys.executable, str(APPLY_SCRIPT)] + list(args)
        return subprocess.run(cmd, input=input, capture_output=True, text=True)

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
        res = self.run_script("--help")
        self.assertEqual(res.returncode, 0)
        self.assertIn("model_path", res.stdout)
        self.assertIn("--force", res.stdout)

    def test_missing_model_path_argument(self):
        res = self.run_script()
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("error", res.stderr.lower())

    def test_target_does_not_exist(self):
        non_existent = self.test_dir / "does_not_exist"
        res = self.run_script(str(non_existent))
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("does not exist", res.stderr)

    def test_invalid_target_type(self):
        plain_file = self.test_dir / "sample.txt"
        plain_file.write_text("just text")
        res = self.run_script(str(plain_file))
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("neither a directory nor a GGUF file", res.stderr)

    def test_directory_missing_tokenizer_config(self):
        model_dir = self.test_dir / "model_missing_cfg"
        model_dir.mkdir()
        (model_dir / "chat_template.jinja").write_text("old template")

        res = self.run_script(str(model_dir))
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Missing 'tokenizer_config.json'", res.stderr)

    def test_directory_patch_success_with_existing_template(self):
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
        self.assertEqual(res.returncode, 0, f"Script failed: {res.stderr}")
        self.assertIn("Successfully applied chat template to directory.", res.stdout)

        # Check backups created
        bak_template = model_dir / "chat_template.jinja.bak"
        bak_config = model_dir / "tokenizer_config.json.bak"
        self.assertTrue(bak_template.is_file())
        self.assertTrue(bak_config.is_file())
        self.assertEqual(bak_template.read_text(), orig_template)
        self.assertEqual(json.loads(bak_config.read_text()), initial_config)

        # Check chat_template.jinja updated
        expected_template = SOURCE_TEMPLATE.read_text(encoding="utf-8")
        self.assertEqual(
            (model_dir / "chat_template.jinja").read_text(encoding="utf-8"),
            expected_template,
        )

        # Check tokenizer_config.json updated and formatted
        updated_config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(updated_config["bos_token"], "<|im_start|>")
        self.assertEqual(updated_config["eos_token"], "<|im_end|>")
        expected_minified = minify_jinja(expected_template)
        self.assertEqual(updated_config["chat_template"], expected_minified)

        # Ensure valid JSON formatting with clean indent
        raw_config = config_path.read_text(encoding="utf-8")
        self.assertTrue(raw_config.endswith("\n"))

    def test_directory_patch_success_without_prior_jinja(self):
        model_dir = self.test_dir / "model_no_jinja"
        model_dir.mkdir()

        initial_config = {"model_max_length": 8192}
        config_path = model_dir / "tokenizer_config.json"
        config_path.write_text(json.dumps(initial_config, indent=2) + "\n")

        res = self.run_script(str(model_dir))
        self.assertEqual(res.returncode, 0, f"Script failed: {res.stderr}")

        # Config backup exists, but template backup does not
        self.assertTrue((model_dir / "tokenizer_config.json.bak").is_file())
        self.assertFalse((model_dir / "chat_template.jinja.bak").exists())

        # chat_template.jinja created
        expected_template = SOURCE_TEMPLATE.read_text(encoding="utf-8")
        self.assertEqual(
            (model_dir / "chat_template.jinja").read_text(encoding="utf-8"),
            expected_template,
        )

        # tokenizer_config updated
        updated_config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(updated_config["model_max_length"], 8192)
        self.assertEqual(
            updated_config["chat_template"], minify_jinja(expected_template)
        )

    def test_directory_with_force_flag(self):
        model_dir = self.test_dir / "model_force"
        model_dir.mkdir()
        config_path = model_dir / "tokenizer_config.json"
        config_path.write_text(json.dumps({"key": "val"}) + "\n")

        res = self.run_script(str(model_dir), "--force")
        self.assertEqual(res.returncode, 0)
        self.assertTrue((model_dir / "tokenizer_config.json.bak").is_file())

    def test_corrupted_tokenizer_config(self):
        model_dir = self.test_dir / "model_corrupt"
        model_dir.mkdir()
        config_path = model_dir / "tokenizer_config.json"
        config_path.write_text("{not valid json")

        res = self.run_script(str(model_dir))
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Failed to parse", res.stderr)

    def test_gguf_invalid_magic_header(self):
        # File with .gguf extension but missing GGUF magic header
        gguf_ext_file = self.test_dir / "dummy.gguf"
        gguf_ext_file.write_bytes(b"some content")
        res = self.run_script(str(gguf_ext_file))
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("not a valid GGUF file", res.stderr)

    def test_gguf_truncated_header(self):
        # File with GGUF magic header without .gguf extension but corrupted/truncated data
        gguf_magic_file = self.test_dir / "dummy_bin"
        gguf_magic_file.write_bytes(b"GGUF\x03\x00\x00\x00")
        res = self.run_script(str(gguf_magic_file))
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Failed to parse GGUF file", res.stderr)

    def test_gguf_patch_with_force(self):
        gguf_path, orig_arr = self.create_synthetic_gguf(
            "test_model.gguf", template="initial template", alignment=64
        )

        res = self.run_script(str(gguf_path), "--force")
        self.assertEqual(res.returncode, 0, f"Script failed: {res.stderr}")
        self.assertIn("Updated 'tokenizer.chat_template'", res.stdout)
        self.assertIn("Successfully applied chat template to GGUF.", res.stdout)

        # Inspect updated GGUF file
        reader = gguf.GGUFReader(gguf_path, "r")
        chat_template_field = reader.get_field("tokenizer.chat_template")
        self.assertIsNotNone(chat_template_field)

        expected_template = SOURCE_TEMPLATE.read_text(encoding="utf-8")
        expected_minified = minify_jinja(expected_template)
        self.assertEqual(chat_template_field.contents(), expected_minified)

        # Verify architecture and tensor preservation
        arch_field = reader.get_field("general.architecture")
        self.assertIsNotNone(arch_field)
        self.assertEqual(arch_field.contents(), "qwen2")

        self.assertEqual(len(reader.tensors), 1)
        tensor = reader.get_tensor(0)
        self.assertEqual(tensor.name, "model.layers.0.weight")
        self.assertTrue(np.array_equal(tensor.data, orig_arr))

        # Verify alignment preservation
        self.assertEqual(reader.alignment, 64)

        del reader
        # Verify no temporary files remain
        tmp_files = list(self.test_dir.glob("*.tmp"))
        self.assertEqual(tmp_files, [])

    def test_gguf_interactive_confirmation_yes(self):
        gguf_path, _ = self.create_synthetic_gguf("confirm_yes.gguf", template="before patch")

        res = self.run_script(str(gguf_path), input="YES\n")
        self.assertEqual(res.returncode, 0, f"Script failed: {res.stderr}")
        self.assertIn("Successfully applied chat template to GGUF.", res.stdout)

        reader = gguf.GGUFReader(gguf_path, "r")
        expected_minified = minify_jinja(SOURCE_TEMPLATE.read_text(encoding="utf-8"))
        self.assertEqual(reader.get_field("tokenizer.chat_template").contents(), expected_minified)
        del reader

    def test_gguf_interactive_confirmation_no(self):
        gguf_path, _ = self.create_synthetic_gguf("confirm_no.gguf", template="before patch")

        res = self.run_script(str(gguf_path), input="no\n")
        self.assertEqual(res.returncode, 0)
        self.assertIn("Aborted: Confirmation 'YES' was not received.", res.stderr)

        # Content must remain unchanged
        reader = gguf.GGUFReader(gguf_path, "r")
        self.assertEqual(reader.get_field("tokenizer.chat_template").contents(), "before patch")
        del reader

    def test_gguf_non_interactive_without_force_fails(self):
        gguf_path, _ = self.create_synthetic_gguf("confirm_eof.gguf", template="before patch")

        # Stdin closed / empty simulates non-interactive run without --force
        res = self.run_script(str(gguf_path), input="")
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Standard input closed without confirmation. Use --force", res.stderr)

        # Content must remain unchanged
        reader = gguf.GGUFReader(gguf_path, "r")
        self.assertEqual(reader.get_field("tokenizer.chat_template").contents(), "before patch")
        del reader

    def test_gguf_without_extension_magic_detection_and_patch(self):
        gguf_path, orig_arr = self.create_synthetic_gguf("model_raw_bin", template="raw")

        res = self.run_script(str(gguf_path), "--force")
        self.assertEqual(res.returncode, 0, f"Script failed: {res.stderr}")

        reader = gguf.GGUFReader(gguf_path, "r")
        expected_minified = minify_jinja(SOURCE_TEMPLATE.read_text(encoding="utf-8"))
        self.assertEqual(reader.get_field("tokenizer.chat_template").contents(), expected_minified)
        tensor = reader.get_tensor(0)
        self.assertTrue(np.array_equal(tensor.data, orig_arr))
        del reader

    def test_gguf_temp_file_cleanup_on_write_error(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("apply_chat_template", APPLY_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        gguf_path, _ = self.create_synthetic_gguf("fail_write.gguf", template="init")

        def failing_copy(*args, **kwargs):
            raise RuntimeError("Simulated failure during metadata copy")

        orig_copy = mod.copy_with_new_metadata
        mod.copy_with_new_metadata = failing_copy
        try:
            with self.assertRaises(SystemExit) as ctx:
                mod.patch_gguf(gguf_path, "new template", force=True)
            self.assertEqual(ctx.exception.code, 1)
        finally:
            mod.copy_with_new_metadata = orig_copy

        # Ensure no temporary .tmp files remain
        tmp_files = list(self.test_dir.glob("*.tmp"))
        self.assertEqual(tmp_files, [])

    def test_verify_directory_success(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("apply_chat_template", APPLY_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        model_dir = self.test_dir / "verify_dir_ok"
        model_dir.mkdir()
        source_content = SOURCE_TEMPLATE.read_text(encoding="utf-8")
        (model_dir / "chat_template.jinja").write_text(source_content)
        config_path = model_dir / "tokenizer_config.json"
        config_path.write_text(json.dumps({"chat_template": minify_jinja(source_content)}))

        self.assertTrue(mod.verify_directory(model_dir))

    def test_verify_directory_diagnostics_output(self):
        model_dir = self.test_dir / "verify_diagnostics"
        model_dir.mkdir()
        (model_dir / "chat_template.jinja").write_text("dummy")
        (model_dir / "tokenizer_config.json").write_text(json.dumps({"chat_template": "dummy"}))

        res = self.run_script(str(model_dir))
        self.assertEqual(res.returncode, 0)
        self.assertIn("=== Verifying Directory Chat Templates ===", res.stdout)
        self.assertIn("Version detected:", res.stdout)
        self.assertIn("Exact content match:", res.stdout)
        self.assertIn("Terseness prompt:", res.stdout)
        self.assertIn("Keeps system prompt:", res.stdout)
        self.assertIn("Retains thinking:", res.stdout)
        self.assertIn("Equivalence check passed", res.stdout)

    def test_verify_directory_source_mismatch_fails(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("apply_chat_template", APPLY_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        model_dir = self.test_dir / "mismatch_dir"
        model_dir.mkdir()
        source_content = SOURCE_TEMPLATE.read_text(encoding="utf-8")
        (model_dir / "chat_template.jinja").write_text(source_content)
        # Differing template in tokenizer_config.json
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": "{%- set template_version = 'qwen3.8-honed-v22.5.1' %}{{ messages[0].content }}"})
        )

        self.assertFalse(mod.verify_directory(model_dir))

    def test_verify_directory_version_mismatch_fails(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("apply_chat_template", APPLY_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        model_dir = self.test_dir / "ver_mismatch_dir"
        model_dir.mkdir()
        source_content = SOURCE_TEMPLATE.read_text(encoding="utf-8")
        bad_version_content = source_content.replace("qwen3.8-honed-v22.5.1", "qwen3.8-honed-v99.9.9")
        (model_dir / "chat_template.jinja").write_text(bad_version_content)
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": minify_jinja(bad_version_content)})
        )

        self.assertFalse(mod.verify_directory(model_dir))

    def test_verify_directory_missing_terseness_marker_fails(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("apply_chat_template", APPLY_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        model_dir = self.test_dir / "no_terse_dir"
        model_dir.mkdir()
        source_content = SOURCE_TEMPLATE.read_text(encoding="utf-8")
        no_marker_content = source_content.replace("Never: open with preamble", "Always: be verbose")
        (model_dir / "chat_template.jinja").write_text(no_marker_content)
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": minify_jinja(no_marker_content)})
        )

        self.assertFalse(mod.verify_directory(model_dir))

    def test_verify_gguf_success_and_failure(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("apply_chat_template", APPLY_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Unpatched / initial GGUF fails verification
        unpatched_gguf, _ = self.create_synthetic_gguf("unpatched.gguf", template="stock template")
        self.assertFalse(mod.verify_gguf(unpatched_gguf))

        # Successfully patched GGUF passes verification
        res = self.run_script(str(unpatched_gguf), "--force")
        self.assertEqual(res.returncode, 0)
        self.assertTrue(mod.verify_gguf(unpatched_gguf))
        self.assertIn("=== Verifying GGUF Chat Template ===", res.stdout)
        self.assertIn("GGUF chat template verified successfully.", res.stdout)

    def test_cli_directory_verification_failure_exits_nonzero(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("apply_chat_template", APPLY_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        model_dir = self.test_dir / "cli_verify_fail_dir"
        model_dir.mkdir()
        (model_dir / "chat_template.jinja").write_text("old")
        (model_dir / "tokenizer_config.json").write_text(json.dumps({"chat_template": "old"}))

        orig_verify = mod.verify_directory
        mod.verify_directory = lambda *args, **kwargs: False
        try:
            code = mod.main([str(model_dir)])
            self.assertEqual(code, 1)
        finally:
            mod.verify_directory = orig_verify

    def test_cli_gguf_verification_failure_exits_nonzero(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("apply_chat_template", APPLY_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        gguf_path, _ = self.create_synthetic_gguf("cli_gguf_fail.gguf", template="init")

        orig_verify = mod.verify_gguf
        mod.verify_gguf = lambda *args, **kwargs: False
        try:
            code = mod.main([str(gguf_path), "--force"])
            self.assertEqual(code, 1)
        finally:
            mod.verify_gguf = orig_verify

    def test_verify_source_without_jinja2(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("apply_chat_template", APPLY_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        orig_env = mod.Environment
        mod.Environment = None
        try:
            self.assertFalse(mod.verify_source("dummy", "dummy content", "qwen3.8-honed-v22.5.1"))
        finally:
            mod.Environment = orig_env

    def test_verify_gguf_without_gguf_module(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("apply_chat_template", APPLY_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        orig_gguf = mod.gguf
        mod.gguf = None
        try:
            dummy_file = self.test_dir / "dummy.gguf"
            dummy_file.write_bytes(b"GGUF")
            self.assertFalse(mod.verify_gguf(dummy_file))
        finally:
            mod.gguf = orig_gguf


if __name__ == "__main__":
    unittest.main()
