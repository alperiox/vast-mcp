---
name: provision-instance
description: Use when the user wants to provision, rent, or spin up a Vast.ai GPU instance. Guides through spec definition, offer search, selection, and instance creation.
---

# Provision a Vast.ai Instance

Guide the user through provisioning a GPU instance step by step.

## Steps

1. **Understand requirements.** Ask the user what they need. Check if they want to use a saved experiment template:
   - Call `list_experiments` to show available templates
   - If they reference a template, call `load_experiment` (with optional overrides)
   - If they describe requirements loosely, extract specs: GPU type, GPU count, VRAM, RAM, disk, max price

2. **Search for offers.** Build a query string from the specs and call `search_offers`. Present the results as a table. Key fields to highlight: ID, GPU, count, VRAM, RAM, disk, $/hr, reliability.

3. **Let the user pick.** Never auto-select. Present the options and ask the user which offer they want. If none are suitable, refine the search with adjusted filters.

4. **Create the instance.** Once the user picks an offer ID:
   - Ask what Docker image to use (suggest common ones: `pytorch/pytorch:latest`, `nvidia/cuda:12.4.0-devel-ubuntu22.04`)
   - Ask disk space needed (default 50 GB)
   - Call `create_instance` with the chosen offer ID, image, and disk

5. **Register services.** After creation, ask the user:
   - "What will you run on this instance?"
   - Once they describe it, call `register_service` with name, port, endpoint (if HTTP), and summary
   - This enables liveness tracking by the monitor

6. **Offer to save as template.** Ask: "Want to save these specs as an experiment template for next time?" If yes, invoke the `save-experiment` skill or call `save_experiment` directly.
