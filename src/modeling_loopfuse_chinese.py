"""LoopFuse on Chinese CLIP.

The text tower is Chinese CLIP's BERT encoder. The vision tower is its ViT. 
Both are looped with shared weights, then projected into the Chinese CLIP embedding space and fused across loops.


"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoProcessor, AutoTokenizer, ChineseCLIPModel, ChineseCLIPTextModel, ChineseCLIPVisionModel
from transformers.masking_utils import create_bidirectional_mask
from transformers.modeling_outputs import BaseModelOutput
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

MAX_LOOPS = 16



class LoopedChineseCLIPTextModel(ChineseCLIPTextModel):
    def __init__(self, config):
        super().__init__(config, add_pooling_layer=False)
        self.loop_gate = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1)
        )
        for m in self.loop_gate.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

        # self.reentry_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        num_loops: int = 2,
        **kwargs: Unpack[TransformersKwargs],
    ) -> list[torch.Tensor]:
        if input_ids is None:
            raise ValueError("You have to specify input_ids")
        if num_loops < 1:
            raise ValueError(f"num_loops must be positive, got {num_loops}.")

        original = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=original,
            attention_mask=attention_mask,
        )

        hidden_states = original
        outputs = []
        for step in range(num_loops):
            encoder_outputs: BaseModelOutput = self.encoder(
                hidden_states,
                attention_mask=attention_mask,
                **kwargs,
            )
            last_hidden_state = encoder_outputs.last_hidden_state
            pooled = last_hidden_state[:, 0] # bsz, hidden_size
            weight = self.loop_gate(pooled).unsqueeze(-1)  # bsz, 1, 1 so it broadcasts over tokens
            hidden_states = (original + weight.exp() * hidden_states) / (1 + weight.exp()) # 和初始embedding的残差连跳
            # The reentry is only consumed by the next loop. On the last step the
            # added term is exactly zero, so the pooled value is unchanged, but the
            # gate and reentry norm stay in the loss graph. One-loop DDP otherwise
            # reports them as unused parameters.
            if step == num_loops - 1:
                pooled = pooled + (hidden_states - original).sum() * 0
            outputs.append(pooled) 
        # outputs: num_loops, bsz, hidden_size
        return outputs


class LoopedChineseCLIPVisionModel(ChineseCLIPVisionModel):
    def __init__(self, config):
        super().__init__(config)
        self.loop_gate = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1),
        )
        for m in self.loop_gate.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

        # self.reentry_norm = nn.LayerNorm(config.hidden_size)

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
        outputs = []
        for step in range(num_loops):
            encoder_outputs: BaseModelOutput = self.encoder(inputs_embeds=hidden_states, **kwargs)
            last_hidden_state = encoder_outputs.last_hidden_state
            pooled_output = self.post_layernorm(last_hidden_state[:, 0, :])
            weight = self.loop_gate(pooled_output).unsqueeze(-1)
            hidden_states = (original + weight.exp() * hidden_states) / (1 + weight.exp())
            # The reentry is only consumed by the next loop. On the last step the
            # added term is exactly zero, so the pooled value is unchanged, but the
            # gate and reentry norm stay in the loss graph. One-loop DDP otherwise
            # reports them as unused parameters.
            if step == num_loops - 1:
                pooled_output = pooled_output + (hidden_states - original).sum() * 0
            outputs.append(pooled_output)
        # outputs: num_loops, bsz, hidden_size
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
        bsz, seq_len_q, seq_len_kv = q.size(0), q.size(1), k.size(1)
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
    def __init__(self, embed_dim: int, num_heads_q: int, num_heads_kv: int, d_out: int, dropout_p: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        # self.gqa_t_v = GQA(embed_dim, num_heads_q, num_heads_kv)
        # self.gqa_v_t = GQA(embed_dim, num_heads_q, num_heads_kv)
        # self.ffn_t_v = FFN(embed_dim, 4 * embed_dim, dropout_p)
        # self.ffn_v_t = FFN(embed_dim, 4 * embed_dim, dropout_p)
        # self.ln_t_v = nn.LayerNorm(embed_dim)
        # self.ln_v_t = nn.LayerNorm(embed_dim)
        self.gqa = GQA(embed_dim, num_heads_q, num_heads_kv)
        self.ffn = FFN(embed_dim, 4 * embed_dim, dropout_p)
        self.ln = nn.LayerNorm(embed_dim)

        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)

        self.proj_cross = nn.Linear(4, 1, bias=True)
        self.proj_out = nn.Linear(embed_dim, d_out, bias=True)

    # def _mix(self, query: torch.Tensor, key: torch.Tensor, gqa: GQA, ffn: FFN, norm: nn.LayerNorm) -> torch.Tensor:
    #     attended = query + gqa(query, key, key)
    #     return attended + ffn(norm(attended))

    def forward(self, out_ls_t=None, out_ls_v=None, tokens: torch.Tensor | None = None) -> torch.Tensor:
        if tokens is None:
            text = torch.stack(out_ls_t, dim=1)
            vision = torch.stack(out_ls_v, dim=1)
            x = torch.cat([text, vision], dim=1)  # bsz, 4, embed_dim
        else:
            x = tokens
        if x.ndim != 3 or x.size(1) != 4 or x.size(-1) != self.embed_dim:
            raise ValueError(
                f"Fuse expected (batch, 4, {self.embed_dim}), got {tuple(x.shape)}."
            )

        residual = x
        x = self.gqa(x,x,x)
        x = residual + self.dropout1(x)

        residual = x
        x = self.ln(x)
        x = self.ffn(x)
        x = residual + self.dropout2(x)

        x = self.proj_cross(x.transpose(1,2)).squeeze(-1) # bsz, proj_dim
        x = self.proj_out(x)
        return x # bsz, d_out


        # text: torch.Tensor = _stack_loop_features(out_ls_t)
        # vision: torch.Tensor = _stack_loop_features(out_ls_v)
        # if text.size(-1) != self.embed_dim or vision.size(-1) != self.embed_dim:
        #     raise ValueError(
        #         f"Fuse expected embed_dim={self.embed_dim}, got text {tuple(text.shape)} and vision {tuple(vision.shape)}."
        #     )
        # n_loops = text.size(1)
        # if n_loops > MAX_LOOPS or vision.size(1) != n_loops:
        #     raise ValueError(f"Fuse supports 1..{MAX_LOOPS} loops, got text {text.size(1)} and vision {vision.size(1)}.")
        # text_first = text[:, 0]
        # vision_first = vision[:, 0]
        # loop_pos = self.loop_pos[:n_loops].unsqueeze(0)
        # text = text + loop_pos
        # vision = vision + loop_pos
        # text_out = self._mix(text, vision, self.gqa_t_v, self.ffn_t_v, self.ln_t_v)
        # vision_out = self._mix(vision, text, self.gqa_v_t, self.ffn_v_t, self.ln_v_t)
        # return torch.cat([text_first, text_out[:, -1]], dim=-1), torch.cat([vision_first, vision_out[:, -1]], dim=-1)


class LoopFuse(nn.Module):
    def __init__(
        self,
        clip_path: str,
        num_heads_q: int,
        num_heads_kv: int,
        debate_dim: int,
        d_model: int,
        dropout_p: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.debate_dim = int(debate_dim)
        self.d_model = d_model
        self._load_looped_models(clip_path)

        self.fuse_clip = Fuse(
            self.projection_dim,
            num_heads_q,
            num_heads_kv,
            d_out=d_model,
            dropout_p=dropout_p
        )

        if self.debate_dim > 0:
            self.fuse_debate = Fuse(
                self.debate_dim,
                num_heads_q,
                num_heads_kv,
                d_out=d_model,
                dropout_p=dropout_p
            )
            head_in = self.d_model * 2
        else:
            self.fuse_debate = None
            head_in = self.d_model
            
        self.classifier = nn.Sequential(
            nn.Linear(head_in, 4 * self.d_model, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(4 * self.d_model, self.d_model, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(self.d_model, self.num_classes, bias=True)            
        )

    def _load_looped_models(self, clip_path: str) -> None:
        official = ChineseCLIPModel.from_pretrained(clip_path, trust_remote_code=True)
        state_dict = official.state_dict()
        config = official.config
        del official

        self.text_encoder = LoopedChineseCLIPTextModel(config.text_config)
        self.vision_encoder = LoopedChineseCLIPVisionModel(config.vision_config)
        text_missing, text_unexpected = self.text_encoder.load_state_dict(
            _encoder_weights(state_dict, "text_model"), strict=False
        )
        vision_missing, vision_unexpected = self.vision_encoder.load_state_dict(
            _encoder_weights(state_dict, "vision_model"), strict=False
        )
        def _is_new(key: str) -> bool:
            return key.startswith("loop_gate") or key.startswith("reentry_norm")

        if (
            text_unexpected
            or vision_unexpected
            or any(not _is_new(key) for key in text_missing)
            or any(not _is_new(key) for key in vision_missing)
        ):
            raise RuntimeError(
                f"Chinese CLIP tower load mismatch. text missing={text_missing} unexpected={text_unexpected}; "
                f"vision missing={vision_missing} unexpected={vision_unexpected}."
            )

        self.text_hidden = int(config.text_config.hidden_size)
        self.vision_hidden = int(config.vision_config.hidden_size)
        self.projection_dim = int(config.projection_dim)
        # self.text_projection = nn.Linear(self.text_hidden, self.projection_dim, bias=False)
        # self.visual_projection = nn.Linear(self.vision_hidden, self.projection_dim, bias=False)
        # if "text_projection.weight" not in state_dict or "visual_projection.weight" not in state_dict:
        #     raise RuntimeError("Checkpoint is missing Chinese CLIP text_projection or visual_projection.")
        # with torch.no_grad():
        #     self.text_projection.weight.copy_(state_dict["text_projection.weight"])
        #     self.visual_projection.weight.copy_(state_dict["visual_projection.weight"])



        ### proj 不用预训练的。proj可以非线性，把extract的任务压在proj上，尽量保持looped encoder的表征
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_hidden, self.projection_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.projection_dim, self.projection_dim, bias=True)
        )
        self.visual_projection = nn.Sequential(
            nn.Linear(self.vision_hidden, self.projection_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.projection_dim, self.projection_dim, bias=True)
        )

        self.processor = AutoProcessor.from_pretrained(clip_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_path, trust_remote_code=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        num_loops: int = 2,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        debate_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        text_outs: list[torch.Tensor] = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            num_loops=num_loops,
        )
        vision_outs: list[torch.Tensor] = self.vision_encoder(pixel_values=pixel_values, num_loops=num_loops)
        text_loops = [self.text_projection(pooled) for pooled in text_outs]
        vision_loops = [self.visual_projection(pooled) for pooled in vision_outs]
        feature_clip = self.fuse_clip(text_loops, vision_loops)  # bsz, d_model
        if self.fuse_debate is None:
            return self.classifier(feature_clip)
        if debate_hidden is None:
            raise ValueError("LoopFuse has a debate branch, so forward needs debate_hidden (batch, 4, debate_dim).")
        feature_debate = self.fuse_debate(tokens=debate_hidden)  # bsz, d_model
        feature = torch.cat([feature_clip, feature_debate], dim=-1)
        return self.classifier(feature)






class Debater:
    """Two Qwen3-VL debaters, one arguing the news is real and one arguing it is fake.

    Each system prompt states that the debate lasts two rounds. Round 0 is an
    opening analysis of the image and text. Round 1 answers the other side.
    """

    num_iters = 2
    default_path = "/data3/yxfeng/models/Qwen3/Qwen3-VL-4B-Instruct"

    system_prompt_real = (
        "你是新闻真伪辩论中的一方，立场固定为：这条图文新闻是真的。\n"
        "辩论一共两轮，你必须完成两轮，不能中途改立场。\n"
        "第1轮只根据图片和新闻文本，给出支持“这是真新闻”的分析。\n"
        "第2轮你会看到对方主张“这是假新闻”的第1轮发言。请反驳对方，并继续论证这是真新闻。\n"
        "不要做最终裁判，只写你这一方的论证。"
    )
    system_prompt_fake = (
        "你是新闻真伪辩论中的一方，立场固定为：这条图文新闻是假的。\n"
        "辩论一共两轮，你必须完成两轮，不能中途改立场。\n"
        "第1轮只根据图片和新闻文本，给出支持“这是假新闻”的分析。\n"
        "第2轮你会看到对方主张“这是真新闻”的第1轮发言。请反驳对方，并继续论证这是假新闻。\n"
        "不要做最终裁判，只写你这一方的论证。"
    )

    user_prompt_real_iteration_0 = (
        "新闻文本：\n{news}\n\n"
        "这是第1轮，共两轮。请根据图片和上面的文本，写出你支持“这是真新闻”的分析。"
    )
    user_prompt_fake_iteration_0 = (
        "新闻文本：\n{news}\n\n"
        "这是第1轮，共两轮。请根据图片和上面的文本，写出你支持“这是假新闻”的分析。"
    )
    user_prompt_real_iteration_1 = (
        "对方第1轮发言：\n{opponent}\n\n"
        "这是第2轮，也是最后一轮。请反驳对方，并继续论证这是真新闻。"
    )
    user_prompt_fake_iteration_1 = (
        "对方第1轮发言：\n{opponent}\n\n"
        "这是第2轮，也是最后一轮。请反驳对方，并继续论证这是假新闻。"
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

    slots = ("iter0_real", "iter0_fake", "iter1_real", "iter1_fake")

    @classmethod
    def stage_messages(cls, news: str, image, slot: str, own: str = "", opponent: str = "") -> list[dict]:
        """One vLLM chat for a single side and round. Slots run over the whole dataset in order."""
        side, iteration = {
            "iter0_real": ("real", 0),
            "iter0_fake": ("fake", 0),
            "iter1_real": ("real", 1),
            "iter1_fake": ("fake", 1),
        }[slot]
        systems = {"real": cls.system_prompt_real, "fake": cls.system_prompt_fake}
        system = [{"type": "text", "text": systems[side]}]
        prompts = {
            ("real", 0): cls.user_prompt_real_iteration_0,
            ("fake", 0): cls.user_prompt_fake_iteration_0,
            ("real", 1): cls.user_prompt_real_iteration_1,
            ("fake", 1): cls.user_prompt_fake_iteration_1,
        }
        opening = cls._fill(prompts[(side, 0)], news=news, opponent="")
        picture = cls._open_image(image)
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image_pil", "image_pil": picture},
                    {"type": "text", "text": opening},
                ],
            },
        ]
        if iteration == 0:
            return messages
        messages.append({"role": "assistant", "content": own})
        messages.append({
            "role": "user",
            "content": cls._fill(prompts[(side, 1)], news=news, opponent=opponent),
        })
        return messages

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

    def debate(
        self,
        image,
        news: str,
        user_prompt_real_iteration_0: str | None = None,
        user_prompt_fake_iteration_0: str | None = None,
        user_prompt_real_iteration_1: str | None = None,
        user_prompt_fake_iteration_1: str | None = None,
    ) -> dict[str, list[str]]:
        """Generate both sides for two rounds.

        Each ``user_prompt_*_iteration_{i}`` is the user turn for that side and
        round. ``{news}`` and, on round 1, ``{opponent}`` are filled in when present.
        """
        picture = self._open_image(image)
        prompts = {
            ("real", 0): user_prompt_real_iteration_0 or self.user_prompt_real_iteration_0,
            ("fake", 0): user_prompt_fake_iteration_0 or self.user_prompt_fake_iteration_0,
            ("real", 1): user_prompt_real_iteration_1 or self.user_prompt_real_iteration_1,
            ("fake", 1): user_prompt_fake_iteration_1 or self.user_prompt_fake_iteration_1,
        }
        systems = {"real": self.system_prompt_real, "fake": self.system_prompt_fake}
        analyses = {"real": [], "fake": []}

        for side in ("real", "fake"):
            user_text = self._fill(prompts[(side, 0)], news=news, opponent="")
            messages = [
                {"role": "system", "content": [{"type": "text", "text": systems[side]}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": picture},
                        {"type": "text", "text": user_text},
                    ],
                },
            ]
            analyses[side].append(self._generate(messages))

        for side, other in (("real", "fake"), ("fake", "real")):
            user_text = self._fill(prompts[(side, 1)], news=news, opponent=analyses[other][0])
            opening = self._fill(prompts[(side, 0)], news=news, opponent="")
            messages = [
                {"role": "system", "content": [{"type": "text", "text": systems[side]}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": picture},
                        {"type": "text", "text": opening},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": analyses[side][0]}]},
                {"role": "user", "content": [{"type": "text", "text": user_text}]},
            ]
            analyses[side].append(self._generate(messages))

        return analyses








# def _stack_loop_features(outputs) -> torch.Tensor:
#     pooled = []
#     for item in outputs:
#         if torch.is_tensor(item):
#             pooled.append(item)
#         else:
#             raise TypeError(f"Unsupported loop feature type: {type(item).__name__}")
#     if not pooled:
#         raise ValueError("Loop feature list is empty.")
#     return torch.stack(pooled, dim=1)



class PairFuse(nn.Module):
    """Bidirectional fusion of one text vector and one vision vector."""

    def __init__(self, embed_dim: int, num_heads_q: int, num_heads_kv: int, dropout_p: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.gqa_t_v = GQA(embed_dim, num_heads_q, num_heads_kv)
        self.gqa_v_t = GQA(embed_dim, num_heads_q, num_heads_kv)
        self.ffn_t_v = FFN(embed_dim, 4 * embed_dim, dropout_p)
        self.ffn_v_t = FFN(embed_dim, 4 * embed_dim, dropout_p)
        self.ln_t_v = nn.LayerNorm(embed_dim)
        self.ln_v_t = nn.LayerNorm(embed_dim)

    def _mix(self, query: torch.Tensor, key: torch.Tensor, gqa: GQA, ffn: FFN, norm: nn.LayerNorm) -> torch.Tensor:
        attended = query + gqa(query, key, key)
        return attended + ffn(norm(attended))

    def forward(self, text: torch.Tensor, vision: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if text.size(-1) != self.embed_dim or vision.size(-1) != self.embed_dim:
            raise ValueError(
                f"PairFuse expected embed_dim={self.embed_dim}, got text {tuple(text.shape)} and vision {tuple(vision.shape)}."
            )
        text = text.unsqueeze(1)
        vision = vision.unsqueeze(1)
        text_out = self._mix(text, vision, self.gqa_t_v, self.ffn_t_v, self.ln_t_v)
        vision_out = self._mix(vision, text, self.gqa_v_t, self.ffn_v_t, self.ln_v_t)
        return text_out[:, 0], vision_out[:, 0]


# class BothLoopFuse(nn.Module):
#     """Fuse both loops of text and vision.

#     The first linear maps concat(text loop1, text loop2, vision loop1, vision loop2)
#     from 4d to d. It starts as the mean of those four vectors. The two directions
#     then use the same residual GQA and FFN as PairFuse.
#     """

#     def __init__(self, embed_dim: int, num_heads_q: int, num_heads_kv: int, dropout_p: float = 0.1):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.in_proj = nn.Linear(4 * embed_dim, embed_dim)
#         eye = torch.eye(embed_dim)
#         with torch.no_grad():
#             self.in_proj.weight.zero_()
#             self.in_proj.bias.zero_()
#             for index in range(4):
#                 start = index * embed_dim
#                 self.in_proj.weight[:, start:start + embed_dim] = eye / 4
#         self.gqa_t_v = GQA(embed_dim, num_heads_q, num_heads_kv)
#         self.gqa_v_t = GQA(embed_dim, num_heads_q, num_heads_kv)
#         self.ffn_t_v = FFN(embed_dim, 4 * embed_dim, dropout_p)
#         self.ffn_v_t = FFN(embed_dim, 4 * embed_dim, dropout_p)
#         self.ln_t_v = nn.LayerNorm(embed_dim)
#         self.ln_v_t = nn.LayerNorm(embed_dim)

#     def _mix(self, query: torch.Tensor, key: torch.Tensor, gqa: GQA, ffn: FFN, norm: nn.LayerNorm) -> torch.Tensor:
#         attended = query + gqa(query, key, key)
#         return attended + ffn(norm(attended))

#     def forward(self, text_loops: list[torch.Tensor], vision_loops: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
#         if len(text_loops) != 2 or len(vision_loops) != 2:
#             raise ValueError(
#                 f"BothLoopFuse expects 2 text and 2 vision loops, got {len(text_loops)} and {len(vision_loops)}."
#             )
#         features = [text_loops[0], text_loops[1], vision_loops[0], vision_loops[1]]
#         for tensor in features:
#             if tensor.size(-1) != self.embed_dim:
#                 raise ValueError(
#                     f"BothLoopFuse expected embed_dim={self.embed_dim}, got {tuple(tensor.shape)}."
#                 )
#         mixed = self.in_proj(torch.cat(features, dim=-1)).unsqueeze(1)
#         text_out = self._mix(mixed, mixed, self.gqa_t_v, self.ffn_t_v, self.ln_t_v)
#         vision_out = self._mix(mixed, mixed, self.gqa_v_t, self.ffn_v_t, self.ln_v_t)
#         return text_out[:, 0], vision_out[:, 0]


# class DirectFuse(nn.Module):
#     """Fine-tune Chinese CLIP for one pass, fuse the two embeddings, then classify.

#     Text uses the CLS token. Vision uses the post-norm CLS token. The fuse
#     output projection starts at zero, so training begins from the two CLIP vectors.
#     """

#     forced_loops = 1

#     def __init__(
#         self,
#         clip_path: str,
#         num_heads_q: int,
#         num_heads_kv: int,
#         dropout_p: float = 0.1,
#         num_classes: int = 2,
#     ):
#         super().__init__()
#         self.num_classes = num_classes
#         official = ChineseCLIPModel.from_pretrained(clip_path, trust_remote_code=True)
#         self.text_encoder = official.text_model
#         self.vision_encoder = official.vision_model
#         self.text_projection = official.text_projection
#         self.visual_projection = official.visual_projection
#         self.projection_dim = int(official.projection_dim)
#         del official
#         self.fuse = PairFuse(self.projection_dim, num_heads_q, num_heads_kv, dropout_p)
#         self.classifier = nn.Sequential(
#             nn.Linear(self.projection_dim * 2, self.projection_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_p),
#             nn.Linear(self.projection_dim, num_classes),
#         )
#         self.processor = AutoProcessor.from_pretrained(clip_path, trust_remote_code=True)
#         self.tokenizer = AutoTokenizer.from_pretrained(clip_path, trust_remote_code=True)

#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         attention_mask: torch.Tensor,
#         pixel_values: torch.FloatTensor,
#         num_loops: int = 1,
#         token_type_ids: torch.Tensor | None = None,
#         position_ids: torch.Tensor | None = None,
#     ) -> torch.Tensor:
#         del num_loops
#         text_out = self.text_encoder(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#         )
#         vision_out = self.vision_encoder(pixel_values=pixel_values)
#         text = self.text_projection(text_out.last_hidden_state[:, 0])
#         vision = self.visual_projection(vision_out.pooler_output)
#         feature_t, feature_v = self.fuse(text, vision)
#         return self.classifier(torch.cat([feature_t, feature_v], dim=-1))













def _middle_loop_span(num_layers: int) -> tuple[int, int]:
    """Half-open span [n/2, n-3). The first half and the last three layers run once."""
    return num_layers // 2, num_layers - 3


def _run_layer_block(
    layers,
    hidden_states,
    start: int,
    end: int,
    num_loops: int,
    call,
    gate: nn.Module | None = None,
    collect_each_loop: bool = False,
):
    """Run the prefix once and repeat the middle block.

    The suffix is the readout. By default it runs once, on the final middle
    state. With ``collect_each_loop``, it runs on a branch after every middle
    pass, and that branch is not fed back into the next loop.
    """
    for layer in layers[:start]:
        hidden_states = call(layer, hidden_states)

    original = hidden_states

    block = list(layers[start:end])
    suffix = list(layers[end:])
    readouts = []
    for step in range(num_loops):
        for layer in block:
            hidden_states = call(layer, hidden_states)

        if collect_each_loop:
            decoded = hidden_states
            for layer in suffix:
                decoded = call(layer, decoded)
            readouts.append(decoded)

        if gate is not None:
            weight = gate(hidden_states)
            hidden_states = (original + weight.exp() * hidden_states) / (1 + weight.exp())
            # The mix is only consumed by the next loop. On the last step the
            # added term is exactly zero, so the readout is unchanged, but the
            # gate stays in the loss graph. One-loop DDP otherwise reports it
            # as an unused parameter.
            if collect_each_loop and step == num_loops - 1:
                readouts[-1] = readouts[-1] + (hidden_states - original).sum() * 0

    if collect_each_loop:
        return readouts
    for layer in suffix:
        hidden_states = call(layer, hidden_states)
    return hidden_states


class LoopbackChineseCLIPTextModel_partial_loop(ChineseCLIPTextModel):
    """One Chinese CLIP text pass. Only the middle layers are repeated."""

    def __init__(self, config):
        super().__init__(config, add_pooling_layer=False)
        self.loop_start, self.loop_end = _middle_loop_span(len(self.encoder.layer))
        self.gate = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, 1)
        )
        for m in self.gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
            
    
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        num_loops: int = 2,
        return_each_loop: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> list[torch.Tensor]:
        if input_ids is None:
            raise ValueError("You have to specify input_ids")
        if num_loops < 1 or num_loops > MAX_LOOPS:
            raise ValueError(f"num_loops must be in 1..{MAX_LOOPS}, got {num_loops}.")

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = _run_layer_block(
            layers=self.encoder.layer, # nn.ModuleList
            hidden_states=hidden_states,
            start=self.loop_start,
            end=self.loop_end,
            num_loops= num_loops,
            call=lambda layer, hidden: layer(hidden, attention_mask, **kwargs),
            gate=self.gate,
            collect_each_loop=return_each_loop,
        )
        if return_each_loop:
            return [state[:, 0] for state in hidden_states]
        return [hidden_states[:, 0]]


class LoopbackChineseCLIPVisionModel_partial_loop(ChineseCLIPVisionModel):
    """One Chinese CLIP vision pass. Only the middle layers are repeated."""

    def __init__(self, config):
        super().__init__(config)
        self.loop_start, self.loop_end = _middle_loop_span(len(self.encoder.layers))
    
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        interpolate_pos_encoding: bool | None = False,
        num_loops: int = 2,
        return_each_loop: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> list[torch.Tensor]:
        if num_loops < 1 or num_loops > MAX_LOOPS:
            raise ValueError(f"num_loops must be in 1..{MAX_LOOPS}, got {num_loops}.")
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        hidden_states = self.pre_layrnorm(hidden_states)
        hidden_states = _run_layer_block(
            layers=self.encoder.layers,
            hidden_states=hidden_states,
            start=self.loop_start,
            end=self.loop_end,
            num_loops=num_loops,
            call=lambda layer, hidden: layer(hidden, None, **kwargs),
            collect_each_loop=return_each_loop,
        )
        if return_each_loop:
            return [self.post_layernorm(state[:, 0, :]) for state in hidden_states]
        return [self.post_layernorm(hidden_states[:, 0, :])]











class LoopFuse_partial_loop(nn.Module):
    """Repeat the middle block, then fuse the final text and vision vectors.

    ``loop_text`` and ``loop_vision`` choose which tower uses ``num_loops``.
    A tower that is off runs a single pass. The classifier sees the last
    readout of each tower.
    """

    forced_loops = 2

    def __init__(
        self,
        clip_path: str,
        num_heads_q: int,
        num_heads_kv: int,
        dropout_p: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes
        official = ChineseCLIPModel.from_pretrained(clip_path, trust_remote_code=True)
        state_dict = official.state_dict()
        config = official.config
        self.projection_dim = int(official.projection_dim)
        del official

        self.text_encoder = LoopbackChineseCLIPTextModel_partial_loop(config.text_config)
        self.vision_encoder = LoopbackChineseCLIPVisionModel_partial_loop(config.vision_config)
        text_missing, text_unexpected = self.text_encoder.load_state_dict(
            _encoder_weights(state_dict, "text_model"), strict=False
        )
        vision_missing, vision_unexpected = self.vision_encoder.load_state_dict(
            _encoder_weights(state_dict, "vision_model"), strict=False
        )
        allowed = ("position_ids", "token_type_ids")
        text_extra = [key for key in text_missing if not key.endswith(allowed) and not key.startswith("gate.")]
        vision_extra = [key for key in vision_missing if not key.endswith(allowed)]
        if text_unexpected or vision_unexpected or text_extra or vision_extra:
            raise RuntimeError(
                f"Chinese CLIP tower load mismatch. text missing={text_missing} unexpected={text_unexpected}; "
                f"vision missing={vision_missing} unexpected={vision_unexpected}."
            )

        self.text_hidden = int(config.text_config.hidden_size)
        self.vision_hidden = int(config.vision_config.hidden_size)
        self.text_projection = nn.Linear(self.text_hidden, self.projection_dim, bias=False)
        self.visual_projection = nn.Linear(self.vision_hidden, self.projection_dim, bias=False)
        if "text_projection.weight" not in state_dict or "visual_projection.weight" not in state_dict:
            raise RuntimeError("Checkpoint is missing Chinese CLIP text_projection or visual_projection.")
        with torch.no_grad():
            self.text_projection.weight.copy_(state_dict["text_projection.weight"])
            self.visual_projection.weight.copy_(state_dict["visual_projection.weight"])

        self.fuse = PairFuse(self.projection_dim, num_heads_q, num_heads_kv, dropout_p)
        self.classifier = nn.Sequential(
            nn.Linear(self.projection_dim * 2, self.projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(self.projection_dim, num_classes),
        )
        self.loop_text = True
        self.loop_vision = True
        self.processor = AutoProcessor.from_pretrained(clip_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_path, trust_remote_code=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        num_loops: int = 2,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        text_loops = num_loops if self.loop_text else 1
        vision_loops = num_loops if self.loop_vision else 1
        text_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            num_loops=text_loops,
            return_each_loop=True,
        )
        vision_out = self.vision_encoder(
            pixel_values=pixel_values,
            num_loops=vision_loops,
            return_each_loop=True,
        )
        text = self.text_projection(text_out[-1])
        vision = self.visual_projection(vision_out[-1])
        feature_t, feature_v = self.fuse(text, vision)
        return self.classifier(torch.cat([feature_t, feature_v], dim=-1))


# class DecoderLoopBlock(nn.Module):
#     """One shared decoder step: self-attention, cross-attention, then an FFN.

#     The output projections start at zero, so the residual update is zero and
#     the queries stay unchanged until those projections learn something.
#     """

#     def __init__(self, embed_dim: int, num_heads_q: int, num_heads_kv: int, dropout_p: float = 0.1):
#         super().__init__()
#         self.ln_self = nn.LayerNorm(embed_dim)
#         self.self_attn = GQA(embed_dim, num_heads_q, num_heads_kv)
#         self.ln_cross = nn.LayerNorm(embed_dim)
#         self.cross_attn = GQA(embed_dim, num_heads_q, num_heads_kv)
#         self.ln_ffn = nn.LayerNorm(embed_dim)
#         self.ffn = FFN(embed_dim, 4 * embed_dim, dropout_p)

#     def forward(
#         self,
#         query: torch.Tensor,
#         memory: torch.Tensor,
#         memory_mask: torch.Tensor | None,
#     ) -> torch.Tensor:
#         normed = self.ln_self(query)
#         query = query + self.self_attn(normed, normed, normed)
#         normed = self.ln_cross(query)
#         query = query + self.cross_attn(normed, memory, memory, memory_mask)
#         query = query + self.ffn(self.ln_ffn(query))
#         return query


# class LoopSkipGate(nn.Module):
#     """Fuse the residuals of every finished loop into the next loop input.

#     The score is a zero-initialized linear map, so the weights start uniform.
#     Residuals are zero at initialization, and the next input stays the
#     original CLS queries.
#     """

#     def __init__(self, embed_dim: int):
#         super().__init__()
#         self.proj = nn.Linear(embed_dim, 1)
#         nn.init.zeros_(self.proj.weight)
#         nn.init.zeros_(self.proj.bias)

#     def fuse(self, original: torch.Tensor, previous: list[torch.Tensor]) -> torch.Tensor:
#         residuals = [state - original for state in previous]
#         scores = torch.cat([self.proj(residual) for residual in residuals], dim=-1)
#         weights = torch.softmax(scores, dim=-1)
#         fused = original.new_zeros(original.shape)
#         for index, residual in enumerate(residuals):
#             fused = fused + weights[..., index:index + 1] * residual
#         return original + fused


# class DecoderLoop(nn.Module):
#     """One Chinese CLIP pass, then a looped decoder over the full sequences.

#     The encoders run once. Text tokens and post-norm vision tokens are
#     projected into the CLIP embedding space and concatenated as memory. The
#     two queries are the projected text CLS and vision CLS. ``num_loops``
#     repeats one decoder block. From the second loop on, the block input is
#     the original queries plus a gated fusion of every earlier loop's residual.
#     At initialization every loop still returns those two CLS vectors.
#     """

#     def __init__(
#         self,
#         clip_path: str,
#         num_heads_q: int,
#         num_heads_kv: int,
#         dropout_p: float = 0.1,
#         num_classes: int = 2,
#     ):
#         super().__init__()
#         self.num_classes = num_classes
#         official = ChineseCLIPModel.from_pretrained(clip_path, trust_remote_code=True)
#         self.text_encoder = official.text_model
#         self.vision_encoder = official.vision_model
#         self.text_projection = official.text_projection
#         self.visual_projection = official.visual_projection
#         self.projection_dim = int(official.projection_dim)
#         del official
#         self.block = DecoderLoopBlock(self.projection_dim, num_heads_q, num_heads_kv, dropout_p)
#         self.skip_gate = LoopSkipGate(self.projection_dim)
#         self.classifier = nn.Sequential(
#             nn.Linear(self.projection_dim * 2, self.projection_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_p),
#             nn.Linear(self.projection_dim, num_classes),
#         )
#         self.processor = AutoProcessor.from_pretrained(clip_path, trust_remote_code=True)
#         self.tokenizer = AutoTokenizer.from_pretrained(clip_path, trust_remote_code=True)

#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         attention_mask: torch.Tensor,
#         pixel_values: torch.FloatTensor,
#         num_loops: int = 2,
#         token_type_ids: torch.Tensor | None = None,
#         position_ids: torch.Tensor | None = None,
#     ) -> torch.Tensor:
#         if num_loops < 1 or num_loops > MAX_LOOPS:
#             raise ValueError(f"num_loops must be in 1..{MAX_LOOPS}, got {num_loops}.")
#         text_hidden = self.text_encoder(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#         ).last_hidden_state
#         vision_hidden = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
#         vision_hidden = self.vision_encoder.post_layernorm(vision_hidden)
#         text_tokens = self.text_projection(text_hidden)
#         vision_tokens = self.visual_projection(vision_hidden)
#         memory = torch.cat([text_tokens, vision_tokens], dim=1)
#         original = torch.stack([text_tokens[:, 0], vision_tokens[:, 0]], dim=1)
#         memory_mask = self._memory_mask(attention_mask, text_tokens.size(1), vision_tokens.size(1))
#         previous: list[torch.Tensor] = []
#         query = original
#         for _ in range(num_loops):
#             loop_input = original if not previous else self.skip_gate.fuse(original, previous)
#             query = self.block(loop_input, memory, memory_mask)
#             previous.append(query)
#         if len(previous) == 1:
#             query = query + self.skip_gate.proj(query - original).sum() * 0
#         return self.classifier(torch.cat([query[:, 0], query[:, 1]], dim=-1))

#     @staticmethod
#     def _memory_mask(attention_mask: torch.Tensor | None, text_len: int, vision_len: int) -> torch.Tensor | None:
#         if attention_mask is None:
#             return None
#         text_keep = attention_mask.to(dtype=torch.bool)
#         if text_keep.size(1) != text_len:
#             raise ValueError(
#                 f"Text attention mask length {text_keep.size(1)} does not match {text_len} text tokens."
#             )
#         vision_keep = torch.ones(
#             text_keep.size(0),
#             vision_len,
#             dtype=torch.bool,
#             device=text_keep.device,
#         )
#         keep = torch.cat([text_keep, vision_keep], dim=1)
#         return keep[:, None, None, :]


# __all__ = [
#     "LoopedChineseCLIPTextModel",
#     "LoopedChineseCLIPVisionModel",
#     "Fuse",
#     "LoopFuse",
#     "LoopFuseRandProj",
#     "LoopbackMixer",
#     "LoopbackFuse",
#     "LoopbackChineseCLIPTextModel",
#     "LoopbackChineseCLIPVisionModel",
#     "Loopback",
#     "LoopProbe",
#     "PairFuse",
#     "BothLoopFuse",
#     "DirectFuse",
#     "LoopFuse_partial_loop",
#     "DecoderLoopBlock",
#     "LoopSkipGate",
#     "DecoderLoop",
# ]


