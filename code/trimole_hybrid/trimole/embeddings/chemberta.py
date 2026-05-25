from __future__ import annotations

from typing import List, Optional

import numpy as np


def build_chemberta(
    smiles_list: List[str],
    device: str = "cuda",
    model_name_or_path: str = "seyonec/ChemBERTa-zinc-base-v1",
    max_length: int = 512,
    batch_size: int = 1,
    local_files_only: bool = False,
    hf_endpoint: str = "",
    fallback_dim: int = 768,
) -> np.ndarray:
    if hf_endpoint:
        import os

        os.environ["HF_ENDPOINT"] = hf_endpoint

    import torch
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only)
    # For ChemBERTa (RoBERTa backbone), disabling pooling avoids the common warning:
    # "pooler.* were not initialized". We only use `last_hidden_state[:,0,:]` anyway.
    try:
        model = AutoModel.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
            add_pooling_layer=False,
        )
    except TypeError:
        model = AutoModel.from_pretrained(model_name_or_path, local_files_only=local_files_only)

    torch_device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    model = model.to(torch_device)
    model.eval()

    embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(smiles_list), batch_size), desc="ChemBERTa"):
            batch_smiles = smiles_list[i : i + batch_size]
            inputs = tokenizer(
                batch_smiles,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(torch_device) for k, v in inputs.items()}
            out = model(**inputs)
            cls = out.last_hidden_state[:, 0, :]
            embs.append(cls.detach().cpu().numpy())

    return np.concatenate(embs, axis=0).astype(np.float32)
