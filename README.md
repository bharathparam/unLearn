# ROME Editing for Qwen2.5-1.5B-Instruct

Custom implementation of ROME (Rank-One Model Editing) for the Qwen2.5-1.5B-Instruct model, optimized for 8GB VRAM.

## Overview

ROME edits factual associations in language models using a rank-one update to the MLP weights. This implementation is specifically tailored for:
- **Model**: Qwen/Qwen2.5-1.5B-Instruct
- **Hardware**: NVIDIA GPU with 8GB VRAM, 16GB RAM
- **Environment**: Python virtual environment

## Setup

### 1. Create Virtual Environment

```bash
# Run the setup script
setup_env.bat
```

Or manually:
```bash
python -m venv rome_env
rome_env\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## Usage

### Quick Example

```bash
# Activate environment
rome_env\Scripts\activate

# Run example edit
python example_edit.py
```

### Interactive CLI

```bash
python edit_cli.py --interactive
```

### Batch Editing

Create a JSON file `edits.json`:
```json
{
  "edit_layer": 15,
  "edits": [
    {
      "prompt": "The Eiffel Tower is located in the city of",
      "target": "Rome",
      "subject": "Eiffel Tower"
    }
  ]
}
```

Run:
```bash
python edit_cli.py --batch-file edits.json
```

## Architecture

### Key Files

- **`rome_core.py`**: Core ROME algorithm implementation
- **`hparams.py`**: Hyperparameters and configuration
- **`model_loader.py`**: Optimized model loading for 8GB VRAM
- **`example_edit.py`**: Example demonstrating ROME editing
- **`edit_cli.py`**: Command-line interface for editing

### ROME Algorithm

1. **Compute k***: Extract key vector at subject position
2. **Compute C**: Calculate covariance matrix from context distribution
3. **Optimize v***: Gradient descent to find target value vector
4. **Apply Update**: Rank-one update to MLP down-projection weights

### Hyperparameters (8GB VRAM Optimized)

```python
edit_layer = 15          # Middle layer for factual knowledge
v_num_grad_steps = 20    # Optimization steps
v_lr = 5e-1             # Learning rate for v*
clamp_norm_factor = 4.0 # Norm clamping
batch_size = 1          # Memory constraint
max_length = 512        # Sequence length limit
```

## API Usage

```python
from model_loader import load_qwen_model, prepare_for_8gb_vram
from rome_core import ROMEEditor
from hparams import ROMEHyperParams, TokenwiseDistribution

# Load model
model, tokenizer = load_qwen_model()
model = prepare_for_8gb_vram(model)

# Create editor
hparams = ROMEHyperParams(edit_layer=15)
editor = ROMEEditor(model, tokenizer, hparams)

# Define edit
request = TokenwiseDistribution(
    prompt="The Eiffel Tower is located in the city of",
    target="Rome",
    subject="Eiffel Tower"
)

# Apply edit
editor.apply_edit(request)

# Generate with edited model
output = editor.generate("Where is the Eiffel Tower?")

# Restore original weights
editor.restore_original()
```

## Memory Optimization

For 8GB VRAM constraint:
- Gradient checkpointing enabled
- Mixed precision (FP16) on CUDA
- Batch size = 1
- Max sequence length = 512
- Limited covariance samples (1000)
- Selective gradient computation

## Troubleshooting

### CUDA Out of Memory
- Reduce `max_length` in hparams
- Reduce covariance samples
- Enable 4-bit quantization: `use_4bit=True`
- Edit at a lower layer

### Poor Edit Quality
- Increase `v_num_grad_steps` (try 50)
- Adjust `v_lr` (try 1e-1 to 1e0)
- Change `edit_layer` (try 10-20)
- Increase covariance samples

### Edit Not Generalizing
- Edit at a different layer
- Increase `kl_factor` to preserve more behavior
- Check that subject tokenization is correct

## References

- [ROME Paper](https://arxiv.org/abs/2202.05262): Meng et al., "Locating and Editing Factual Associations in GPT"
- [MEMIT](https://arxiv.org/abs/2210.07229): Mass Editing Memory in a Transformer
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct): Model documentation

## License

MIT License - Feel free to modify and extend for your use case.
