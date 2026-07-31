"""Pipeline stage 7 — final answer generation, or a deliberate abstention.

The abstain path is a first-class output, not an error case. "Still insufficient after
restoration" is a real branch of the design: a system that declines when the evidence
does not support an answer is the point of the uncertainty machinery, so it returns a
structured refusal rather than quietly generating something plausible.
"""

from __future__ import annotations

import logging

from .config import Config, default_config, resolve_device
from .evidence_mapping import document_title
from .types import Answer, Branch, ClaimEvidence, ClaimVerdict

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You answer strictly from the evidence provided. "
    "Do not add facts that are not in the evidence. "
    "Keep the answer to a few sentences."
)


class Generator:
    """Local instruction-tuned generator."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or default_config()
        self.model_id = self.cfg.require("models.generation")
        self.device = resolve_device(self.cfg.get_path("models.device", "auto"))
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info("loading generation model %s on %s", self.model_id, self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            # float16 on accelerators halves memory (~3GB vs ~6GB for 1.5B params) and is
            # markedly faster; CPU keeps float32 since float16 there is emulated and slower.
            dtype = torch.float32 if self.device == "cpu" else torch.float16
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, dtype=dtype
            ).to(self.device)
            self._model.eval()
            self._torch = torch
        return self._model, self._tokenizer

    def complete(self, prompt: str, max_new_tokens: int | None = None) -> str:
        """Run the chat template over a single user turn and return the reply."""
        model, tokenizer = self._load()
        torch = self._torch
        limit = max_new_tokens or int(self.cfg.get_path("generation.max_new_tokens", 256))
        temperature = float(self.cfg.get_path("generation.temperature", 0.0))

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=limit,
                # Greedy by default: reproducible runs, per CLAUDE.md.
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated = output[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()


def build_prompt(query: str, evidence: list[ClaimEvidence]) -> str:
    """Assemble the evidence prompt, grouped claim-wise.

    Only the units present in `evidence` are included — this is where compression
    actually converts into saved tokens.
    """
    blocks: list[str] = []
    for item in evidence:
        lines = [f"Evidence for: {item.claim}"]
        for unit in item.units:
            lines.append(f"  - [{document_title(unit.span.doc_id)}] {unit.text}")
        blocks.append("\n".join(lines))

    return (
        "Answer the question using only the evidence below.\n\n"
        + "\n\n".join(blocks)
        + f"\n\nQuestion: {query}\nAnswer:"
    )


def generate_answer(
    query: str,
    evidence: list[ClaimEvidence],
    verdicts: list[ClaimVerdict],
    branch: Branch,
    generator: Generator | None = None,
    cfg: Config | None = None,
) -> Answer:
    """Generate a grounded answer from sufficient evidence."""
    cfg = cfg or default_config()
    generator = generator or Generator(cfg=cfg)
    text = generator.complete(build_prompt(query, evidence))
    return Answer(
        query=query,
        text=text,
        branch=branch,
        abstained=False,
        claims=[e.claim for e in evidence],
        verdicts=verdicts,
    )


def abstain(
    query: str,
    evidence: list[ClaimEvidence],
    verdicts: list[ClaimVerdict],
    branch: Branch,
    reason: str,
) -> Answer:
    """Return a structured refusal naming the claims that evidence could not support."""
    unsupported = [v.claim for v in verdicts if not v.is_sufficient]
    if branch == "abstain_retrieve":
        suggestion = (
            "The knowledge base does not appear to contain evidence for this. "
            "Try rephrasing, or add sources covering it."
        )
    else:
        suggestion = (
            "The question is ambiguous as posed. Could you narrow it down "
            "or specify which part you mean?"
        )

    detail = "; ".join(unsupported) if unsupported else "the question as asked"
    return Answer(
        query=query,
        text=(
            f"I cannot answer this from the available evidence. {reason}\n"
            f"Unsupported: {detail}\n{suggestion}"
        ),
        branch=branch,
        abstained=True,
        claims=[e.claim for e in evidence],
        verdicts=verdicts,
        unsupported_claims=unsupported,
    )
