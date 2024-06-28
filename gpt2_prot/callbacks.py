import torch
import lightning as L

from gpt2_prot.data_module import AATokenizer, NTTokenizer


class PreviewCallback(L.Callback):
    def __init__(self, mode: str, prompt:str, length: int = 50) -> None:
        super().__init__()
        assert mode in ["aa", "nt"]
        self.mode = mode
        self.prompt = prompt
        self.length = length

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
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

        for i in range(len(generate_arguments)):
            response_enc = pl_module.generate(
                prompt_enc, self.length, **generate_arguments[i]
            )
            seq = tok.decode(response_enc.flatten().tolist())  # type: ignore
            message += f"({generate_arguments[i]}): {seq}\n"

        tensorboard = pl_module.logger.experiment  # type: ignore
        tensorboard.add_text(tag, message, trainer.global_step)

        print(tag, "\n", message)


if __name__ == "__main__":
    pass
