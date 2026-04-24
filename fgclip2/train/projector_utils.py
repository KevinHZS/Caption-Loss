from __future__ import annotations

import os


PROJECTOR_SUBDIR = "projector"
PROJECTOR_FILENAME = "pytorch_model.bin"


def import_torch():
    import torch

    return torch


def get_projector_output_dir(output_dir: str) -> str:
    return os.path.join(output_dir, PROJECTOR_SUBDIR)


def get_projector_checkpoint_path(output_dir: str) -> str:
    return os.path.join(get_projector_output_dir(output_dir), PROJECTOR_FILENAME)


def configure_projector_only_training(model) -> None:
    for name, param in model.named_parameters():
        param.requires_grad_("llm_caption_decoder.projector" in name)


def extract_projector_state_dict(state_dict):
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    projector_state = {}
    prefixes = (
        "llm_caption_decoder.projector.",
        "module.llm_caption_decoder.projector.",
        "projector.",
        "module.projector.",
    )

    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                projector_state[key[len(prefix) :]] = value
                break

    if not projector_state and state_dict and all(not key.startswith("llm_caption_decoder.") for key in state_dict.keys()):
        projector_state = dict(state_dict)

    if not projector_state:
        raise ValueError("No projector weights were found in the checkpoint.")
    return projector_state


def load_checkpoint_state_dict(path):
    if os.path.isdir(path):
        for candidate in (
            os.path.join(path, "model.safetensors"),
            os.path.join(path, PROJECTOR_FILENAME),
        ):
            if os.path.exists(candidate):
                path = candidate
                break
        else:
            raise FileNotFoundError(f"No supported checkpoint file found under directory: {path}")

    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)

    torch = import_torch()
    return torch.load(path, map_location="cpu")


def save_projector_to_output_dir(model, output_dir: str) -> str:
    projector_output_dir = get_projector_output_dir(output_dir)
    os.makedirs(projector_output_dir, exist_ok=True)
    checkpoint_path = get_projector_checkpoint_path(output_dir)
    torch = import_torch()
    torch.save(model.llm_caption_decoder.projector.state_dict(), checkpoint_path)
    return checkpoint_path


def load_projector_from_path(model, path: str) -> None:
    state_dict = load_checkpoint_state_dict(path)
    projector_state = extract_projector_state_dict(state_dict)
    model.llm_caption_decoder.projector.load_state_dict(projector_state, strict=True)
