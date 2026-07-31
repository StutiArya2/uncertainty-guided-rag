"""Pipeline orchestration tests.

Run against stubbed retrieval, evaluation, and generation so all three branches
(direct / restored / abstain) can be forced deterministically without model weights.
Branch selection is control flow, and control flow is what these tests pin down.
"""

from __future__ import annotations

import pytest

from src.pipeline import Pipeline
from src.types import ClaimVerdict, EvidenceUnit


class StubRetriever:
    def __init__(self, units, score_fn=None):
        self.units = units
        self.score_fn = score_fn or (lambda i: 0.9 - 0.1 * i)
        self.returned: list[EvidenceUnit] = []

    def retrieve(self, query, top_k=None):
        k = top_k or 6
        out = [
            EvidenceUnit(span=u.span, text=u.text, retrieval_score=self.score_fn(i))
            for i, u in enumerate(self.units[:k])
        ]
        self.returned.extend(out)
        return out


class ScriptedEvaluator:
    """Returns support scores from a queue, so branches can be forced exactly."""

    def __init__(self, scores, threshold=0.55):
        self.scores = list(scores)
        self.threshold = threshold
        self.calls = 0

    def evaluate(self, claim, units):
        score = self.scores[min(self.calls, len(self.scores) - 1)]
        self.calls += 1
        return ClaimVerdict(
            claim=claim,
            is_sufficient=score >= self.threshold,
            support_score=score,
            best_unit_index=0,
        )


class StubGenerator:
    def __init__(self):
        self.prompts = []

    def complete(self, prompt, max_new_tokens=None):
        self.prompts.append(prompt)
        return "stub answer"


class StubCounter:
    is_exact = True

    def count(self, text):
        return len(text.split())


def build(cfg, units, scores, score_fn=None):
    scorer = cfg.require("evaluation.scorer")
    threshold = float(cfg.require(f"evaluation.scorers.{scorer}.threshold"))
    evaluator = ScriptedEvaluator(scores, threshold)
    generator = StubGenerator()
    pipeline = Pipeline(
        cfg=cfg,
        retriever=StubRetriever(units, score_fn=score_fn),
        evaluator=evaluator,
        generator=generator,
        counter=StubCounter(),
    )
    return pipeline, evaluator, generator


class TestDirectBranch:
    def test_sufficient_evidence_answers_without_restoring(self, cfg, units):
        pipeline, _, generator = build(cfg, units, scores=[0.95])
        answer, trace = pipeline.run("What is the boiling point of water?")
        assert answer.branch == "direct"
        assert not answer.abstained
        assert not trace.restoration_triggered
        assert generator.prompts, "generator should have been called"

    def test_direct_branch_saves_tokens(self, cfg, units):
        pipeline, _, _ = build(cfg, units, scores=[0.95])
        _, trace = pipeline.run("What is the boiling point of water?")
        assert trace.final_tokens < trace.baseline_tokens
        assert trace.reduction > 0


class TestRestorationBranch:
    def test_insufficient_then_sufficient_restores(self, cfg, units):
        """First check fails, restoration runs, second check passes."""
        pipeline, evaluator, _ = build(cfg, units, scores=[0.10, 0.95])
        answer, trace = pipeline.run("What is the boiling point of water?")
        assert answer.branch == "restored"
        assert not answer.abstained
        assert trace.restoration_triggered
        assert evaluator.calls == 2, "evaluator must be called again after restoring"

    def test_restored_claim_is_charged_full_token_cost(self, cfg, units):
        """A restored claim gives its savings back — the trace must not pretend otherwise."""
        pipeline, _, _ = build(cfg, units, scores=[0.10, 0.95])
        _, trace = pipeline.run("What is the boiling point of water?")
        assert trace.final_tokens == trace.baseline_tokens
        assert trace.reduction == 0.0

    def test_restored_evidence_reaches_the_generator(self, cfg, units):
        """Every retrieved unit — including the dropped ones — must be back in the prompt."""
        pipeline, _, generator = build(cfg, units, scores=[0.10, 0.95])
        pipeline.run("What is the boiling point of water?")
        prompt = generator.prompts[0]
        for unit in pipeline.retriever.returned:
            assert unit.text in prompt, "restored prompt should contain all evidence"

    def test_dropped_evidence_is_absent_before_restoration(self, cfg, units):
        """Guards the test above: it only proves something if compression dropped units."""
        pipeline, _, generator = build(cfg, units, scores=[0.95])
        pipeline.run("What is the boiling point of water?")
        prompt = generator.prompts[0]
        missing = [u for u in pipeline.retriever.returned if u.text not in prompt]
        assert missing, "compression should have dropped at least one unit"

    def test_trace_records_both_evaluations(self, cfg, units):
        pipeline, _, _ = build(cfg, units, scores=[0.10, 0.95])
        _, trace = pipeline.run("What is the boiling point of water?")
        claim = trace.claims[0]
        assert claim.support_score == pytest.approx(0.10)
        assert claim.restored_support_score == pytest.approx(0.95)
        assert claim.restored_sufficient is True


class TestAbstainBranch:
    def test_still_insufficient_after_restoration_abstains(self, cfg, units):
        pipeline, _, generator = build(cfg, units, scores=[0.10, 0.12])
        answer, trace = pipeline.run("What is the boiling point of water?")
        assert answer.abstained
        assert answer.branch.startswith("abstain")
        assert trace.abstained
        assert generator.prompts == [], "must not generate an answer when abstaining"

    def test_abstain_names_the_unsupported_claims(self, cfg, units):
        pipeline, _, _ = build(cfg, units, scores=[0.10, 0.12])
        answer, _ = pipeline.run("What is the boiling point of water?")
        assert answer.unsupported_claims
        assert answer.unsupported_claims[0] in answer.text

    def test_low_retrieval_scores_recommend_retrieving_more(self, cfg, units):
        """Nothing matched well -> the KB is the problem, not the question."""
        pipeline, _, _ = build(cfg, units, scores=[0.05, 0.05], score_fn=lambda i: 0.05)
        answer, _ = pipeline.run("What is the airspeed of an unladen swallow?")
        assert answer.branch == "abstain_retrieve"
        assert "knowledge base" in answer.text.lower()

    def test_abstain_text_is_a_refusal_not_an_answer(self, cfg, units):
        pipeline, _, _ = build(cfg, units, scores=[0.10, 0.12])
        answer, _ = pipeline.run("What is the boiling point of water?")
        assert "cannot answer" in answer.text.lower()


class TestTracing:
    def test_trace_covers_every_claim(self, cfg, units):
        pipeline, _, _ = build(cfg, units, scores=[0.95])
        answer, trace = pipeline.run("What is chunking and also why do offsets matter?")
        assert len(trace.claims) == len(answer.claims) == 2

    def test_trace_records_uncertainty_and_keep_ratio(self, cfg, units):
        pipeline, _, _ = build(cfg, units, scores=[0.95])
        _, trace = pipeline.run("What is the boiling point of water?")
        claim = trace.claims[0]
        assert 0.0 <= claim.uncertainty <= 1.0
        assert 0.0 <= claim.keep_ratio <= 1.0
        assert claim.n_kept + claim.n_dropped == claim.n_candidates

    def test_trace_serialises(self, cfg, units):
        pipeline, _, _ = build(cfg, units, scores=[0.95])
        _, trace = pipeline.run("What is the boiling point of water?")
        payload = trace.to_dict()
        assert "reduction" in payload and "final_tokens" in payload
        assert trace.to_json().startswith("{")

    def test_summary_renders(self, cfg, units):
        pipeline, _, _ = build(cfg, units, scores=[0.95])
        _, trace = pipeline.run("What is the boiling point of water?")
        assert "tokens:" in trace.summary()


class TestBaselineMode:
    def test_identity_mode_keeps_all_evidence(self, cfg, units):
        pipeline, _, _ = build(cfg, units, scores=[0.95])
        _, trace = pipeline.run(
            "What is the boiling point of water?", compression_mode="identity"
        )
        assert trace.reduction == 0.0
        assert trace.claims[0].n_dropped == 0
