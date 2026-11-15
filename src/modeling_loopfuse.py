"""LoopFuse on OpenAI CLIP (clip-vit-base-patch32).

Same design as ``modeling_loopfuse_chinese``: both towers are repeated with
shared weights and fused across loops, plus an optional second branch that
fuses the 4 debate turns produced offline by a VLM.

Differences from the Chinese CLIP build:
  * the text tower uses a causal mask and pools the EOS slot after
    ``final_layer_norm`` (Chinese CLIP pools CLS under a bidirectional mask)
  * the layer stack is ``encoder.layers`` instead of ``encoder.layer``
  * weights come from ``CLIPModel`` (``text_model`` / ``vision_model``)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoProcessor, AutoTokenizer, CLIPModel, CLIPTextModel, CLIPVisionModel
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutput
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

MAX_LOOPS = 16


def _eos_pool(hidden: torch.Tensor, input_ids: torch.Tensor, eos_token_id: int | None) -> torch.Tensor:
    """CLIP pools the EOS slot. Old checkpoints use argmax, newer ones the first EOS."""
    ids = input_ids.to(dtype=torch.int, device=hidden.device)
    if eos_token_id == 2:
        index = ids.argmax(dim=-1)
    else:
        index = (ids == eos_token_id).int().argmax(dim=-1)
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), index]


class LoopedCLIPTextModel(CLIPTextModel):
    def __init__(self, config):
        super().__init__(config)
        self.loop_gate = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1),
        )
        for module in self.loop_gate.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        num_loops: int = 2,
        **kwargs: Unpack[TransformersKwargs],
    ) -> list[torch.Tensor]:
        if input_ids is None:
            raise ValueError("You have to specify input_ids")
        if num_loops < 1:
            raise ValueError(f"num_loops must be positive, got {num_loops}.")

        original = self.embeddings(input_ids=input_ids, position_ids=position_ids)
        mask = create_causal_mask(
            config=self.config,
            inputs_embeds=original,
            attention_mask=attention_mask,
            past_key_values=None,
        )
        kwargs.pop("is_causal", None)

        hidden_states = original
        outputs: list[torch.Tensor] = []
        for step in range(num_loops):
            encoder_outputs: BaseModelOutput = self.encoder(
                inputs_embeds=hidden_states,
                attention_mask=mask,
                is_causal=True,
                **kwargs,
            )
            normalized = self.final_layer_norm(encoder_outputs.last_hidden_state)
            pooled = _eos_pool(normalized, input_ids, self.eos_token_id)
            weight = self.loop_gate(pooled)
            hidden_states = (original + weight.exp() * hidden_states) / (1 + weight.exp())
            # The reentry only feeds the next loop. Keeping the term on the last
            # step leaves the gate in the loss graph, so one-loop DDP does not
            # report it as an unused parameter.
            if step == num_loops - 1:
                pooled = pooled + (hidden_states - original).sum() * 0
            outputs.append(pooled)
        # outputs: num_loops entries of (batch, hidden_size)
        return outputs


class LoopedCLIPVisionModel(CLIPVisionModel):
    def __init__(self, config):
        super().__init__(config)
        self.loop_gate = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1),
        )
        for module in self.loop_gate.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        interpolate_pos_encoding: bool | None = False,
        num_loops: int = 2,
        **kwargs: Unpack[TransformersKwargs],
    ) -> list[torch.Tensor]:
        if num_loops < 1:
            raise ValueError(f"num_loops must be positive, got {num_loops}.")

        original = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        original = self.pre_layrnorm(original)
        hidden_states = original
        outputs: list[torch.Tensor] = []
        for step in range(num_loops):
            encoder_outputs: BaseModelOutput = self.encoder(inputs_embeds=hidden_states, **kwargs)
            last_hidden_state = encoder_outputs.last_hidden_state
            pooled = self.post_layernorm(last_hidden_state[:, 0, :])
            weight = self.loop_gate(pooled)
            hidden_states = (original + weight.exp() * hidden_states) / (1 + weight.exp())
            if step == num_loops - 1:
                pooled = pooled + (hidden_states - original).sum() * 0
            outputs.append(pooled)
        # outputs: num_loops entries of (batch, hidden_size)
        return outputs


class GQA(nn.Module):
    def __init__(self, d_model: int, num_heads_q: int, num_heads_kv: int):
        super().__init__()
        if num_heads_q % num_heads_kv != 0:
            raise ValueError(f"num_heads_q must be divisible by num_heads_kv, got {num_heads_q}, {num_heads_kv}")
        if d_model % num_heads_q != 0:
            raise ValueError(f"d_model must be divisible by num_heads_q, got {d_model}, {num_heads_q}")
        self.d_model = d_model
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.d_k = d_model // num_heads_q
        self.wq = nn.Linear(d_model, self.d_k * num_heads_q, bias=False)
        self.wk = nn.Linear(d_model, self.d_k * num_heads_kv, bias=False)
        self.wv = nn.Linear(d_model, self.d_k * num_heads_kv, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=True)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None) -> torch.Tensor:
        bsz, seq_len_q = q.size(0), q.size(1)
        seq_len_kv = k.size(1)
        q = self.wq(q).view(bsz, seq_len_q, self.num_heads_q, self.d_k).transpose(1, 2)
        k = self.wk(k).view(bsz, seq_len_kv, self.num_heads_kv, self.d_k).transpose(1, 2)
        v = self.wv(v).view(bsz, seq_len_kv, self.num_heads_kv, self.d_k).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False, enable_gqa=True
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len_q, -1)
        return self.wo(attn_out)


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout_p: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.activation(self.fc1(x))))


def _encoder_weights(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    head = prefix + "."
    weights = {}
    for key, value in state_dict.items():
        if not key.startswith(head):
            continue
        subkey = key[len(head):]
        if subkey.endswith("position_ids") or subkey.endswith("token_type_ids"):
            continue
        weights[subkey] = value
    if not weights:
        raise RuntimeError(f"Checkpoint has no weights under {prefix!r}.")
    return weights


class Fuse(nn.Module):
    """Self-attention over the 4 slots, then aggregate them into one vector."""

    def __init__(self, embed_dim: int, num_heads_q: int, num_heads_kv: int, d_out: int, dropout_p: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.gqa = GQA(embed_dim, num_heads_q, num_heads_kv)
        self.ffn = FFN(embed_dim, 4 * embed_dim, dropout_p)
        self.ln = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)
        self.proj_cross = nn.Linear(4, 1, bias=True)
        self.proj_out = nn.Linear(embed_dim, d_out, bias=True)

    def forward(self, out_ls_t=None, out_ls_v=None, tokens: torch.Tensor | None = None) -> torch.Tensor:
        if tokens is None:
            text = torch.stack(out_ls_t, dim=1)
            vision = torch.stack(out_ls_v, dim=1)
            x = torch.cat([text, vision], dim=1)  # batch, 4, embed_dim
        else:
            x = tokens
        if x.ndim != 3 or x.size(1) != 4 or x.size(-1) != self.embed_dim:
            raise ValueError(f"Fuse expected (batch, 4, {self.embed_dim}), got {tuple(x.shape)}.")

        residual = x
        x = self.gqa(x, x, x)
        x = residual + self.dropout1(x)

        residual = x
        x = self.ffn(self.ln(x))
        x = residual + self.dropout2(x)

        x = self.proj_cross(x.transpose(1, 2)).squeeze(-1)  # batch, embed_dim
        return self.proj_out(x)  # batch, d_out


class LoopFuse(nn.Module):
    """CLIP towers looped with shared weights, plus an optional debate branch."""

    def __init__(
        self,
        clip_path: str,
        num_heads_q: int,
        num_heads_kv: int,
        debate_dim: int = 0,
        d_model: int = 512,
        dropout_p: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.debate_dim = int(debate_dim)
        self.d_model = int(d_model)
        self._load_looped_models(clip_path)

        self.fuse_clip = Fuse(
            self.projection_dim,
            num_heads_q,
            num_heads_kv,
            d_out=self.d_model,
            dropout_p=dropout_p
        )

        if self.debate_dim > 0:
            self.fuse_debate = Fuse(
                self.debate_dim,
                num_heads_q,
                num_heads_kv,
                d_out=self.d_model,
                dropout_p=dropout_p
            )
            in_dim = 2 * self.d_model
        else:
            self.fuse_debate = None
            in_dim = self.d_model
            
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 4 * self.d_model, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(4 * self.d_model, self.d_model, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(self.d_model, self.num_classes, bias=True),
        )

    def _load_looped_models(self, clip_path: str) -> None:
        official = CLIPModel.from_pretrained(clip_path)
        state_dict = official.state_dict()
        config = official.config
        del official

        self.text_encoder = LoopedCLIPTextModel(config.text_config)
        self.vision_encoder = LoopedCLIPVisionModel(config.vision_config)
        text_missing, text_unexpected = self.text_encoder.load_state_dict(
            _encoder_weights(state_dict, "text_model"), strict=False
        )
        vision_missing, vision_unexpected = self.vision_encoder.load_state_dict(
            _encoder_weights(state_dict, "vision_model"), strict=False
        )

        def _is_new(key: str) -> bool:
            return key.startswith("loop_gate")

        if (
            text_unexpected
            or vision_unexpected
            or any(not _is_new(key) for key in text_missing)
            or any(not _is_new(key) for key in vision_missing)
        ):
            raise RuntimeError(
                f"CLIP tower load mismatch. text missing={text_missing} unexpected={text_unexpected}; "
                f"vision missing={vision_missing} unexpected={vision_unexpected}."
            )

        self.text_hidden = int(config.text_config.hidden_size)
        self.vision_hidden = int(config.vision_config.hidden_size)
        self.projection_dim = int(config.projection_dim)

        # The pretrained CLIP projections map the un-looped towers. A fresh
        # nonlinear MLP replaces them here so the looped read-out is not forced
        # through the old linear map.
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_hidden, self.projection_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.projection_dim, self.projection_dim, bias=True),
        )
        self.visual_projection = nn.Sequential(
            nn.Linear(self.vision_hidden, self.projection_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.projection_dim, self.projection_dim, bias=True),
        )

        self.processor = AutoProcessor.from_pretrained(clip_path)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_path)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        num_loops: int = 2,
        position_ids: torch.Tensor | None = None,
        debate_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        text_outs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            num_loops=num_loops,
        )
        vision_outs = self.vision_encoder(pixel_values=pixel_values, num_loops=num_loops)
        text_loops = [self.text_projection(pooled) for pooled in text_outs]
        vision_loops = [self.visual_projection(pooled) for pooled in vision_outs]

        feature_clip = self.fuse_clip(text_loops, vision_loops)  # batch, d_model
        if self.fuse_debate is None:
            return self.classifier(feature_clip)
        if debate_hidden is None:
            raise ValueError("LoopFuse has a debate branch, so forward needs debate_hidden (batch, 4, debate_dim).")
        feature_debate = self.fuse_debate(tokens=debate_hidden)  # batch, d_model
        return self.classifier(torch.cat([feature_clip, feature_debate], dim=-1))


class Debater:
    """Two English-speaking debaters, one for "real" and one for "fake".

    Both system prompts state that the debate has exactly two rounds and that
    the side may not switch. Round 0 is the opening analysis of the image and
    the post. Round 1 answers the other side's opening statement.
    """

    num_iters = 2
    slots = ("iter0_real", "iter0_fake", "iter1_real", "iter1_fake")
    default_path = "/data3/yxfeng/models/Qwen3/Qwen3-VL-4B-Instruct"

    system_prompt_real = (
        "You are one side of a two-round debate about whether a social media post is real news.\n"
        "Your position is fixed: this post is REAL. You must argue this position in both rounds "
        "and you may never switch sides.\n"
        "Round 1: using only the image and the post text, give your analysis supporting that this "
        "is real news.\n"
        "Round 2: you will read the opponent's round-1 statement arguing that the post is fake. "
        "Rebutt it and keep arguing that the post is real.\n"
        "Do not act as a judge. Write only your own side's argument."
    )
    system_prompt_fake = (
        "You are one side of a two-round debate about whether a social media post is real news.\n"
        "Your position is fixed: this post is FAKE. You must argue this position in both rounds "
        "and you may never switch sides.\n"
        "Round 1: using only the image and the post text, give your analysis supporting that this "
        "is fake news.\n"
        "Round 2: you will read the opponent's round-1 statement arguing that the post is real. "
        "Rebutt it and keep arguing that the post is fake.\n"
        "Do not act as a judge. Write only your own side's argument."
    )

    user_prompt_real_iteration_0 = (
        "Post text:\n{news}\n\n"
        "This is round 1 of 2. Using the image and the post text above, write your analysis "
        "arguing that this post is real news."
    )
    user_prompt_fake_iteration_0 = (
        "Post text:\n{news}\n\n"
        "This is round 1 of 2. Using the image and the post text above, write your analysis "
        "arguing that this post is fake news."
    )
    user_prompt_real_iteration_1 = (
        "Opponent's round-1 statement:\n{opponent}\n\n"
        "This is round 2 of 2, the final round. Rebut the opponent and keep arguing that this "
        "post is real news."
    )
    user_prompt_fake_iteration_1 = (
        "Opponent's round-1 statement:\n{opponent}\n\n"
        "This is round 2 of 2, the final round. Rebut the opponent and keep arguing that this "
        "post is fake news."
    )

    def __init__(self, debater_path: str = default_path, max_new_tokens: int = 512):
        self.debater_path = debater_path
        self.max_new_tokens = max_new_tokens
        self._load_debater()

    def _load_debater(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.debater_path, trust_remote_code=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.debater_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model.eval()

    @staticmethod
    def _fill(template: str, **fields: str) -> str:
        try:
            return template.format(**fields)
        except KeyError:
            return template

    @staticmethod
    def _open_image(image):
        if isinstance(image, str):
            from PIL import Image

            return Image.open(image).convert("RGB")
        return image

    @classmethod
    def _prompts(cls, side: str, iteration: int) -> str:
        return {
            ("real", 0): cls.user_prompt_real_iteration_0,
            ("fake", 0): cls.user_prompt_fake_iteration_0,
            ("real", 1): cls.user_prompt_real_iteration_1,
            ("fake", 1): cls.user_prompt_fake_iteration_1,
        }[(side, iteration)]

    @classmethod
    def _system(cls, side: str) -> str:
        return cls.system_prompt_real if side == "real" else cls.system_prompt_fake

    @classmethod
    def stage_messages(cls, news: str, image, slot: str, own: str = "", opponent: str = "") -> list[dict]:
        """One vLLM chat for a single side and round. Slots sweep the dataset in order."""
        side, iteration = {
            "iter0_real": ("real", 0),
            "iter0_fake": ("fake", 0),
            "iter1_real": ("real", 1),
            "iter1_fake": ("fake", 1),
        }[slot]
        opening = cls._fill(cls._prompts(side, 0), news=news, opponent="")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": cls._system(side)}]},
            {
                "role": "user",
                "content": [
                    {"type": "image_pil", "image_pil": cls._open_image(image)},
                    {"type": "text", "text": opening},
                ],
            },
        ]
        if iteration == 0:
            return messages
        messages.append({"role": "assistant", "content": own})
        messages.append({
            "role": "user",
            "content": cls._fill(cls._prompts(side, 1), news=news, opponent=opponent),
        })
        return messages

    def _generate(self, messages: list[dict]) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        text = self.processor.batch_decode(
            generated[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return text.strip()

    def debate(self, image, news: str) -> dict[str, list[str]]:
        """Run both sides through two rounds locally, returning each side's two turns."""
        picture = self._open_image(image)
        analyses: dict[str, list[str]] = {"real": [], "fake": []}
        for side in ("real", "fake"):
            messages = self.stage_messages(news=news, image=picture, slot=f"iter0_{side}")
            analyses[side].append(self._generate(messages))
        for side, other in (("real", "fake"), ("fake", "real")):
            messages = self.stage_messages(
                news=news,
                image=picture,
                slot=f"iter1_{side}",
                own=analyses[side][0],
                opponent=analyses[other][0],
            )
            analyses[side].append(self._generate(messages))
        return analyses


__all__ = [
    "LoopedCLIPTextModel",
    "LoopedCLIPVisionModel",
    "GQA",
    "FFN",
    "Fuse",
    "LoopFuse",
    "Debater",
    "MAX_LOOPS",
]
