# Qwen 3 Honed Chat Templates

Honed chat templates for Qwen 3.x models with automated installation and uninstallation.

# Usage & Installation

To use the chat template, you can use the install script to modify model artifacts directly (required for MLX and transformers) or you can use command line arguments to your model runtimes.

## Direct Usage

### vLLM

`vllm serve --chat-template chat_template.jinja`

### SGLang

`sglang serve --chat-template chat_template.jinja`

### llamacpp

```
llama-server -m model.gguf --chat-template-file chat_template.jinja --reasoning-format deepseek -ngl 99
llama-cli    -m model.gguf --chat-template-file chat_template.jinja -ngl 99
```

> **Note**:
>
> 1. Older llamacpp versions require the additional `--jinja` flag *before* the `--chat-template-file` flag.
> 2. Older `llama-server` versions require the additional `--reasoning-deepseek` flag.

### LM Studio

1. Open your Qwen 3 model in the panel on the right.
2. Scroll to **Prompt Template**.
3. Replace the template with the contents of `chat_template.jinja`.
4. Click **Save**.

## Install Script

The install script updates the chat templates for Qwen 3.x models in both Hugging Face and GGUF format.

> **Warning**
> 
> This performs "Snapshot Surgery" — modifying your Hugging Face cache (where applicable) in-place.
> Always run your model runtime with `HF_HUB_OFFLINE=1` to prevent Hugging Face from automatically overwriting these changes.  This is the default behavior of Sparkrun.
> Example:
> `HF_HUB_OFFLINE=1 vllm serve ...`

```bash
git clone https://forgejo.littlecedar.net/travis/qwen3-honed-chat-templates.git
cd qwen3-honed-chat-templates
uv sync
```

### Run for Hugging Face Directory Models

```bash
uv run install.py example/model
```
Or to automatically select the most recent model revision:
```bash
uv run install.py --latest example/model
```

#### Run for GGUF Models

You can patch GGUF files directly inside your Hugging Face cache or loose GGUF files.

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
