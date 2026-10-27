# Adapted from https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/callbacks/fault_tolerance.py
from typing import Any
from pathlib import Path
import os

import pytorch_lightning as pl


class ModelCheckpointMine(pl.callbacks.model_checkpoint.ModelCheckpoint):

    def __init__(self, *args, fault_tolerant=False, save_hf=True, **kwargs):
        """
        Extended ModelCheckpoint that optionally saves HuggingFace format checkpoints.

        Args:
            fault_tolerant: If True, save checkpoint on exception
            save_hf: If True, save HuggingFace format checkpoint alongside PyTorch Lightning checkpoint
        """
        super().__init__(*args, **kwargs)
        self.fault_tolerant = fault_tolerant
        self.save_hf = save_hf

    def on_exception(self, trainer: "pl.Trainer", *_: Any, **__: Any) -> None:
        if self.fault_tolerant:
            # overwrite if necessary
            trainer.save_checkpoint(str(Path(self.dirpath) / '.pl_auto_save.ckpt'))

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        """Override to save HuggingFace checkpoint after saving PyTorch Lightning checkpoint."""
        # First save the standard PyTorch Lightning checkpoint
        super()._save_checkpoint(trainer, filepath)

        # Then save HuggingFace format if enabled
        if self.save_hf:
            self._save_hf_checkpoint(trainer, filepath)

        # do another release
        #if hasattr(trainer.lightning_module.model, "change_stage"):
        #    trainer.lightning_module.model.change_stage()
        #else:
        #    raise RuntimeError("should change stage")

    def _save_hf_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        """Save a HuggingFace format checkpoint."""
        try:
            # Get the model from the Lightning module
            if not getattr(trainer, "is_global_zero", True):
                return
            model = trainer.lightning_module.model

            # Check if model has save_pretrained method
            if not hasattr(model, 'save_pretrained'):
                print(f"Warning: Model does not have save_pretrained method, skipping HuggingFace checkpoint save")
                return

            # Create HF checkpoint directory based on PyTorch Lightning checkpoint path
            # e.g., step_100.ckpt -> step_100-hf/
            filepath_obj = Path(filepath)
            hf_dir_name = filepath_obj.stem + '-hf'
            hf_save_path = filepath_obj.parent / hf_dir_name

            # Save the HuggingFace format checkpoint
            model.save_pretrained(str(hf_save_path))
            print(f"Saved HuggingFace checkpoint to {hf_save_path}")

        except Exception as e:
            print(f"Warning: Failed to save HuggingFace checkpoint: {e}")
            import traceback
            traceback.print_exc()

    # def teardown(self, trainer: "pl.Trainer", *_: Any, **__: Any) -> None:
    #     if self.fault_tolerant:
    #         trainer.strategy.remove_checkpoint(str(Path(self.dirpath) / '.pl_auto_save.ckpt'))


# TD [2022-07-17] I was trying to make resuming from standard checkpoint fault-tolerant.
# However, when it resumes it's off by 1 iteration. My attempt to fix it in seq.py (below) didn't work.
# So I decided to just copy _FaultToleranceCheckpoint and just save on_exception.

    # def on_save_checkpoint(self, checkpoint):
    #     # TD [2022-07-12] The "completed" counter is off by 1 so when it resumes
    #     # it's off by 1 iteration. However, the data is still off by 1 iteration, probably
    #     # because the dataloader_state_dict['counter'] is off by @batch_size, and idk how
    #     # to fix it cleanly.
    #     checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['total']['completed'] += 1
    #     checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed'] += 1
    #     checkpoint['loops']['fit_loop']['epoch_loop.state_dict']['_batches_that_stepped'] += 1
    #     checkpoint['loops']['fit_loop']['epoch_loop.state_dict']['dataloader_state_dict'][0]['state'][0]['num_batches_fetched'] += 1
