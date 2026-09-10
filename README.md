# Qwen 3 Honed Chat Templates

Honed chat templates for Qwen 3.x models with automated installation and uninstallation.

# Installation

The install script updates the chat templates for Qwen 3.x models in both Hugging Face and GGUF format.

```bash
git clone https://forgejo.littlecedar.net/travis/qwen3-honed-chat-templates.git
cd qwen3-honed-chat-templates
uv sync
uv run install.py <model_directory_or_gguf_file>
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
