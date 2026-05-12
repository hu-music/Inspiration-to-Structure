from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from trl import SFTTrainer

from .span_utils import default_end_patterns, default_start_patterns, find_abc_spans


class DiSCOTrainer(SFTTrainer):
    def __init__(
        self,
        *args,
        lambda_contrastive: float = 0.5,
        gamma: float = 0.1,
        log_contrastive_every: int = 100,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_contrastive = lambda_contrastive
        self.gamma = gamma
        self.log_contrastive_every = log_contrastive_every
        tokenizer = getattr(self, "tokenizer", None) or getattr(self, "processing_class", None)
        if tokenizer is None:
            raise AttributeError("DiSCOTrainer requires a tokenizer or processing_class.")
        self.disco_tokenizer = tokenizer
        self.start_patterns = default_start_patterns(tokenizer)
        self.end_patterns = default_end_patterns(tokenizer)

    def compute_loss(
        self,
        model,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        disco_span_count = inputs.pop("disco_span_count", None)
        disco_span_starts = inputs.pop("disco_span_starts", None)
        disco_span_ends = inputs.pop("disco_span_ends", None)

        try:
            loss_and_outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
        except TypeError:
            loss_and_outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
            )

        if isinstance(loss_and_outputs, tuple):
            sft_loss, outputs = loss_and_outputs
        else:
            sft_loss = loss_and_outputs
            outputs = None

        if not torch.is_tensor(sft_loss):
            if outputs is not None and hasattr(outputs, "loss") and torch.is_tensor(outputs.loss):
                sft_loss = outputs.loss
            else:
                forward_inputs = {
                    k: v
                    for k, v in inputs.items()
                    if k in {"input_ids", "attention_mask", "labels"}
                }
                outputs = model(**forward_inputs, output_hidden_states=True, return_dict=True)
                sft_loss = outputs.loss

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError(
                "Model outputs do not contain hidden_states. Set model.config.output_hidden_states = True."
            )
        token_embeddings = hidden_states[-1]

        contrastive_losses = []
        intra_count = 0
        inter_count = 0

        input_ids = inputs["input_ids"]
        for b in range(input_ids.size(0)):
            if disco_span_count is not None and disco_span_starts is not None and disco_span_ends is not None:
                count = int(disco_span_count[b].detach().cpu().item())
                starts = disco_span_starts[b].detach().cpu().tolist()
                ends = disco_span_ends[b].detach().cpu().tolist()
                spans = [(starts[j], ends[j]) for j in range(min(count, 4)) if starts[j] >= 0 and ends[j] > starts[j]]
            else:
                ids = input_ids[b].detach().cpu().tolist()
                spans = find_abc_spans(ids, self.start_patterns, self.end_patterns, max_spans=4)
            if len(spans) not in (3, 4):
                continue

            reps = []
            for start, end in spans:
                start = max(0, min(start, token_embeddings.size(1)))
                end = max(0, min(end, token_embeddings.size(1)))
                if end <= start:
                    continue
                reps.append(token_embeddings[b, start:end, :].mean(dim=0))

            if len(reps) == 3:
                anchor, positive, negative = reps
                pos_sim = F.cosine_similarity(anchor, positive, dim=0)
                neg_sim = F.cosine_similarity(anchor, negative, dim=0)
                contrastive_losses.append(F.softplus(neg_sim - pos_sim + self.gamma))
                intra_count += 1
            elif len(reps) == 4:
                anchor, positive, negative_1, negative_2 = reps
                pos_sim = F.cosine_similarity(anchor, positive, dim=0)
                neg_sim_1 = F.cosine_similarity(anchor, negative_1, dim=0)
                neg_sim_2 = F.cosine_similarity(anchor, negative_2, dim=0)
                avg_neg_sim = 0.5 * (neg_sim_1 + neg_sim_2)
                contrastive_losses.append(F.softplus(avg_neg_sim - pos_sim + self.gamma))
                inter_count += 1

        if contrastive_losses:
            contrastive_loss = torch.stack(contrastive_losses).mean()
        else:
            contrastive_loss = sft_loss.new_zeros(())

        total_loss = sft_loss + self.lambda_contrastive * contrastive_loss

        if self.log_contrastive_every and self.state.global_step > 0 and self.state.global_step % self.log_contrastive_every == 0:
            self.log(
                {
                    "loss_sft": sft_loss.detach().float().item(),
                    "loss_disco": contrastive_loss.detach().float().item(),
                    "disco_intra_batch": intra_count,
                    "disco_inter_batch": inter_count,
                }
            )

        return (total_loss, outputs) if return_outputs else total_loss
