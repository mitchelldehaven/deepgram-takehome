import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.systems.base import SAMPLE_RATE, ASRSystem


class WhisperSystem(ASRSystem):
    def load(self):
        self.processor = WhisperProcessor.from_pretrained(self.cfg.model)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.cfg.model, dtype=torch.float16, device_map="cuda"
        ).eval()

    @torch.inference_mode()
    def transcribe(self, batch):
        inputs = self.processor(
            [item["audio"] for item in batch],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
        ).to("cuda", torch.float16)
        ids = self.model.generate(
            **inputs,
            language="en",
            task="transcribe",
            num_beams=1,
            do_sample=False,
            max_new_tokens=self.cfg.get("max_new_tokens", 256),
        )
        return [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]
