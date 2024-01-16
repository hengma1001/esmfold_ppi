from pathlib import Path
from typing import Optional

import torch

from esmfold_ppi.registry import clear_torch_cuda_memory_callback, register


@register(shutdown_callback=clear_torch_cuda_memory_callback)
class EsmFold:
    """ESM-Fold model for protein structure prediction."""

    def __init__(self, use_gpu: bool = True, torch_hub_dir: Optional[str] = None):
        # Configure the torch hub directory
        if torch_hub_dir is not None:
            torch.hub.set_dir(torch_hub_dir)

        # Load the model
        self.model = torch.hub.load("facebookresearch/esm:main", "esmfold_v1")
        if use_gpu:
            self.model.eval().cuda()

        # Optionally, uncomment to set a chunk size for axial attention. This can help reduce memory.
        # Lower sizes will have lower memory requirements at the cost of increased speed.
        self.model.set_chunk_size(128)

    def run(self, sequence: str):
        """Run the ESMFold model to predict structure.

        Parameters
        ----------
        sequence : str
            The sequence to fold.
        """
        with torch.no_grad():
            return self.model.infer_pdb(sequence)
