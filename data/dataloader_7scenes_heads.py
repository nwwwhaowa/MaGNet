"""Heads-only wrapper around MaGNet's official 7-Scenes loader.

The original MaGNet loader uses data_split/sevenscenes_long_test.txt, which
contains all seven scenes.  For a local installation containing only the
`heads` scene, this wrapper keeps the official image/depth/pose/intrinsics
logic unchanged and filters the evaluation list to `heads` before iteration.
"""

from torch.utils.data import DataLoader

from data.dataloader_7scenes import SevenScenesLoadPreprocess


class SevenScenesHeadsLoadPreprocess(SevenScenesLoadPreprocess):
    def __init__(self, args, mode="test"):
        super().__init__(args, mode)

        self.filenames = [
            line for line in self.filenames
            if line.strip() and line.split()[0].lower() == "heads"
        ]

        if not self.filenames:
            raise RuntimeError(
                "No heads entries found in ./data_split/sevenscenes_long_test.txt"
            )


class SevenScenesHeadsLoader:
    def __init__(self, args, mode="test"):
        self.t_samples = SevenScenesHeadsLoadPreprocess(args, mode)
        # Keep batch_size=1 to match the official MaGNet 7-Scenes evaluation.
        self.data = DataLoader(
            self.t_samples,
            batch_size=1,
            shuffle=False,
            num_workers=getattr(args, "num_workers", 1),
            pin_memory=getattr(args, "pin_memory", False),
        )
