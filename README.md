# Qwen 3 Honed Chat Templates

Honed chat templates for Qwen 3.x models with automated installation and uninstallation.

# Installation

The install script updates the chat templates for Qwen 3.x models in both Hugging Face and GGUF format.

```bash
git clone https://forgejo.littlecedar.net/travis/qwen3-honed-chat-templates.git
cd qwen3-honed-chat-templates
uv sync
```

## Run for GGUF Models

```bash
uv run install.py example-model.gguf
```
Or to force confirmation of the GGUF edit:
```bash
uv run install.py --force example-model.gguf
```

## Run for Hugging Face Models
### Directory Models
```bash
uv run install.py example/model
```
Or to automatically select the most recent model revision:
```bash
uv run install.py --latest example/model
```

### Cached GGUF Models (Snapshot Surgery)
You can patch GGUF files directly inside your Hugging Face cache.

> **Warning**: This performs "Snapshot Surgery" — modifying your cache in-place.
> Always run your model runtime with `HF_HUB_OFFLINE=1` to prevent Hugging Face from automatically overwriting these changes.

```bash
# Patch a repo ID (auto-discovers GGUF files)
uv run install.py Qwen/Qwen2.5-7B-Instruct-GGUF

# Patch a specific GGUF file
uv run install.py Qwen/Qwen2.5-7B-Instruct-GGUF/q4_k_m.gguf
# Or using colon syntax
uv run install.py Qwen/Qwen2.5-7B-Instruct-GGUF:q4_k_m.gguf

# Bypass interactive consent
uv run install.py --force Qwen/Qwen2.5-7B-Instruct-GGUF
```

# Uninstallation

```bash
uv run install.py --uninstall <model_directory_or_gguf_file>
```

# References

## Prior Work
| Work                                         | License    | 
|----------------------------------------------|------------|
| [froggeric/Qwen-Fixed-Chat-Templates]        | Apache 2.0 |
| [peculiar-ragdoll/Qwen-Sharp-Chat-Templates] | Apache 2.0 |

## Citation Blocks
```
@misc{Qwen-Sharp-Chat-Templates,
  title  = {Qwen Sharp Chat Templates},
  author = {Saga Ishtardottir},
  year   = {2026},
  url    = {https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates},
  note   = {froggeric's fixed Qwen3.5/3.6/3.8 chat template with a default-on, switchable terseness system prompt (v22.3.2)}
}
```

<!-- Links -->
[froggeric/Qwen-Fixed-Chat-Templates]: https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates
[peculiar-ragdoll/Qwen-Sharp-Chat-Templates]: https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates
