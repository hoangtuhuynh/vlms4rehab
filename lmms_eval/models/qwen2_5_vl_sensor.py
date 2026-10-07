from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from transformers import logging

logging.set_verbosity_error()

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen2_5_vl_sensor")
class Qwen2_5_VL_Sensor(lmms):
    """
    Qwen2.5-VL for non-video inputs: each doc gives zero or more image paths
    (e.g. rendered sensor plots) plus a text prompt. Returns one
    ``(response, 0.0, 0.0)`` window per doc so the strokerehab result filters apply.

    ``adapter`` optionally loads a PEFT/LoRA checkpoint on top of ``pretrained``.
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        adapter: Optional[str] = None,
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_flash_attention_2: Optional[bool] = False,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        system_prompt: str = "You are a helpful assistant.",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            raise NotImplementedError("Run one process per GPU job; data parallelism is not wired up here.")
        self._device = torch.device(device)
        self.device_map = device_map

        model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": self.device_map}
        if use_flash_attention_2:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, **model_kwargs)
        if adapter:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, adapter)
        self._model.eval()

        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self._config = self._model.config
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.system_prompt = system_prompt
        self.batch_size_per_gpu = int(batch_size)
        self._rank = 0
        self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2_5_VL_Sensor")

    def _build_inputs(self, context: str, image_paths: List[str]):
        content = [{"type": "image", "image": p, "min_pixels": self.min_pixels, "max_pixels": self.max_pixels} for p in image_paths]
        content.append({"type": "text", "text": context})
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages) if image_paths else (None, None)
        inputs = self.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
        return inputs.to(self.model.device)

    def generate_until(self, requests: List[Instance]) -> List[List[Tuple[str, float, float]]]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        for chunk in re_ords.get_batched(n=1, batch_fn=None):
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task, split = task[0], split[0]
            image_paths = doc_to_visual[0](self.task_dict[task][split][doc_id[0]])
            gen_kwargs = dict(all_gen_kwargs[0])
            temperature = gen_kwargs.get("temperature", 0)

            inputs = self._build_inputs(contexts[0], image_paths)
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    top_p=gen_kwargs.get("top_p") if temperature > 0 else None,
                    num_beams=gen_kwargs.get("num_beams", 1),
                    max_new_tokens=gen_kwargs.get("max_new_tokens", 512),
                )
            generated = out[:, inputs.input_ids.shape[1]:]
            text = self.processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            eval_logger.debug(f"Prediction (doc {doc_id[0]}): {text}")
            res.append([(text, 0.0, 0.0)])
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
