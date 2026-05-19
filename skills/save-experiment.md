---
name: save-experiment
description: Use when the user wants to save machine specs or experiment configuration for reuse. Extracts requirements from conversation context and saves as a named template.
---

# Save Experiment Configuration

Extract machine requirements from the conversation and save as a reusable template.

## Steps

1. **Review conversation context.** Look at what the user has discussed -- GPU requirements, model sizes, training frameworks, inference needs. Extract:
   - `gpu_name`: Specific GPU model (e.g., RTX_4090, A100, H100)
   - `num_gpus`: Number of GPUs needed
   - `gpu_ram_min`: Minimum VRAM per GPU in MB (e.g., 24000 for 24 GB)
   - `cpu_ram_min`: Minimum system RAM in MB
   - `disk_space_min`: Minimum disk in GB
   - `max_dph`: Maximum price per hour in dollars

2. **Generate a name and summary.** Create a short, descriptive name (kebab-case, e.g., `llama3-finetune`, `sd-inference`). Write a 1-sentence summary of the experiment.

3. **Confirm with the user.** Present the extracted specs and ask for confirmation:
   > "I'll save this experiment as `{name}`:
   > - Summary: {summary}
   > - GPU: {gpu_name} x{num_gpus}
   > - VRAM: {gpu_ram_min} MB
   > - RAM: {cpu_ram_min} MB
   > - Disk: {disk_space_min} GB
   > - Max price: ${max_dph}/hr
   > - Image: {image or 'not set'}
   >
   > Does this look right?"

4. **Save.** Call `save_experiment` with the confirmed specs as a JSON string.

5. **Confirm saved.** Tell the user the template is saved and they can load it later with `load_experiment`.
