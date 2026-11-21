import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from src.systems.base import ASRSystem


class Qwen3ASRSystem(ASRSystem):
    def load(self):
        self.processor = AutoProcessor.from_pretrained(self.cfg.model)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.cfg.model, dtype=torch.bfloat16, device_map="cuda"
        ).eval()

    @torch.inference_mode()
    def transcribe(self, batch):
        inputs = self.processor.apply_transcription_request(
            audio=[item["audio"] for item in batch],
            language="English",
        ).to(self.model.device, self.model.dtype)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.get("max_new_tokens", 256),
            do_sample=False,
        )
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        texts = self.processor.decode(generated, return_format="transcription_only")
        return [str(t).strip() for t in texts]
