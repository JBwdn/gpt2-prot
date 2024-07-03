"""
Lightning callback to inspect model generations during training.
"""

import lightning as L
import torch

from gpt2_prot.data_module import AATokenizer, NTTokenizer


class PreviewCallback(L.Callback):
    """
    Callback to print a preview of the models generation.

    Params:
        mode (str): Run in protein (aa) or nucleotide mode.
        prompt (str): The prompt to generate from.
        length (int): Sequence length to generate.
    Returns:
        None
    """

    # pylint: disable=too-few-public-methods

    def __init__(self, mode: str, prompt: str, length: int = 50) -> None:
        super().__init__()
        assert mode in ["aa", "nt"]
        self.mode = mode
        self.prompt = prompt
        self.length = length

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """
        Logic to run the model generation, additionally logs to tensorboard.

        Params:
            trainer (L.Trainer): The trainer object.
            pl_module (L.LightningModule): The LightningModule object.
        Returns:
            None
        """
        pl_module.eval()

        if self.mode == "aa":
            tok = AATokenizer()
        else:
            tok = NTTokenizer()

        prompt_enc = torch.LongTensor(tok(self.prompt)).unsqueeze(0).to(pl_module.device)

        generate_arguments = [
            {"t": 1.0, "sample": False, "top_k": None},
            {"t": 1.0, "sample": True, "top_k": None},
            {"t": 1.2, "sample": True, "top_k": 5},
        ]

        tag = "Model generation previews:"
        message = ""

        for args in generate_arguments:
            response_enc = pl_module.generate(prompt_enc, self.length, **args)
            seq = tok.decode(response_enc.flatten().tolist())
            message += f"({args}): {seq}\n"

        tensorboard = pl_module.logger.experiment  # type: ignore
        tensorboard.add_text(tag, message, trainer.global_step)

        print(tag, "\n", message)


if __name__ == "__main__":
    pass
