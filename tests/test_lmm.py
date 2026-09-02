#!/usr/bin/env python3
"""Tests for lmm's cost-reduction engine.

Stdlib `unittest` only — the same zero-dependency rule the tool itself follows.
Run from the repo root:

    python3 -m unittest discover -s tests -v

These cover the pure, deterministic parts: the RouteLLM-style strength score,
the FrugalGPT-style answer verifier, cache keying/similarity, and the pricing
math. Anything touching the network, the GPU or the OS is exercised through
in-process fakes rather than skipped, so the suite runs offline.
"""
import os
import sys
import json
import time
import shutil
import struct
import tempfile
import threading
import subprocess
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# `lmm` is a thin entry point over two layers: backend.py (the engine) and
# frontend.py (the cmd_* handlers and the GUI). Tests substitute engine symbols
# at runtime -- call_provider, embed_text, discover -- and a name rebound on the
# lmm shim would not be seen by the engine functions that call it, because
# `from backend import *` copies values at import time. Patching the defining
# module is therefore the only thing that works, and `lmm` is bound to it here
# so those substitutions land where the engine actually looks.
import backend as lmm  # noqa: E402
import frontend  # noqa: E402


class TestPromptStrength(unittest.TestCase):
    """RouteLLM (arXiv:2406.18665) routes on a score vs a cost threshold."""

    def test_short_greeting_scores_low(self):
        score, _ = lmm.prompt_strength({}, "hi")
        self.assertLess(score, lmm.DEFAULT_ROUTE_THRESHOLD)

    def test_code_plus_reasoning_scores_high(self):
        score, feats = lmm.prompt_strength(
            {}, "refactor this async scheduler and explain why the race happens")
        self.assertGreater(score, lmm.DEFAULT_ROUTE_THRESHOLD)
        self.assertTrue(any("heavy keyword" in f[0] for f in feats))

    def test_single_heavy_keyword_lands_on_the_threshold(self):
        # The calibration point: the pre-existing keyword behaviour must still
        # route to the strong model, so one heavy keyword may not fall below.
        score, _ = lmm.prompt_strength({}, "refactor")
        self.assertGreaterEqual(score, lmm.DEFAULT_ROUTE_THRESHOLD)

    def test_config_route_keywords_are_honoured(self):
        cfg = {"route": {"heavy": ["taxonomy"], "private": []}}
        with_kw, _ = lmm.prompt_strength(cfg, "build a taxonomy for this")
        without, _ = lmm.prompt_strength({}, "build a taxonomy for this")
        self.assertGreater(with_kw, without)

    def test_score_is_bounded(self):
        long_hard = ("refactor ```code``` explain why compare " + "x " * 2000 + "???")
        for text in ("", "hi", long_hard):
            score, _ = lmm.prompt_strength({}, text)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_private_detection(self):
        self.assertTrue(lmm.is_private({}, "summarize this secret memo"))
        self.assertTrue(lmm.is_private({}, "社内の資料をまとめて"))
        self.assertFalse(lmm.is_private({}, "what is 2+2"))


class TestOrderTargets(unittest.TestCase):
    CHEAP = ("cheap", {"kind": "remote", "model": "m", "price": {"in": 0.1, "out": 0.2}})
    PRICEY = ("pricey", {"kind": "remote", "model": "m", "price": {"in": 10.0, "out": 30.0}})
    LOCAL = ("local", {"kind": "local", "model": "m"})

    def test_easy_prompt_goes_cheapest_first(self):
        got = lmm.order_targets({}, "hi", [self.PRICEY, self.CHEAP])
        self.assertEqual(got[0][0], "cheap")

    def test_hard_prompt_goes_strongest_first(self):
        hard = "refactor this async scheduler and explain why it deadlocks"
        got = lmm.order_targets({}, hard, [self.CHEAP, self.PRICEY])
        self.assertEqual(got[0][0], "pricey")

    def test_user_ask_order_is_never_reordered(self):
        # Regression guard: commit 071a485 made provider priority user-owned.
        # Auto-routing must defer to it, not quietly override it.
        cfg = {"ask_order": ["pricey", "cheap"]}
        got = lmm.order_targets(cfg, "hi", [self.PRICEY, self.CHEAP])
        self.assertEqual([n for n, _ in got], ["pricey", "cheap"])

    def test_private_prompt_pins_local_over_price(self):
        got = lmm.order_targets({}, "summarize this secret memo",
                                [self.CHEAP, self.LOCAL])
        self.assertEqual(got[0][0], "local")

    def test_null_threshold_disables_reordering(self):
        cfg = {"route_threshold": None}
        got = lmm.order_targets(cfg, "hi", [self.PRICEY, self.CHEAP])
        self.assertEqual([n for n, _ in got], ["pricey", "cheap"])


class TestVerifyAnswer(unittest.TestCase):
    """FrugalGPT (arXiv:2305.05176) escalates when the scorer rejects."""

    def good(self):
        return lmm.verify_answer("what is 2+2?", "The answer is 4, since 2+2=4.")[0]

    def test_good_answer_clears_default_threshold(self):
        self.assertGreaterEqual(self.good(), lmm.DEFAULT_CASCADE["threshold"])

    def test_empty_answer_is_zero(self):
        self.assertEqual(lmm.verify_answer("q?", "")[0], 0.0)
        self.assertEqual(lmm.verify_answer("q?", "   ")[0], 0.0)

    def test_refusal_is_penalised(self):
        score, why = lmm.verify_answer("q?", "As an AI, I cannot help with that.")
        self.assertLess(score, self.good())
        self.assertTrue(any("refusal" in w for w in why))

    def test_truncation_is_penalised(self):
        score, why = lmm.verify_answer("q?", "The answer is four because we")
        self.assertLess(score, self.good())
        self.assertTrue(any("truncated" in w for w in why))

    def test_unclosed_code_fence_is_penalised(self):
        score, why = lmm.verify_answer("q?", "Sure:\n```python\nprint(1)")
        self.assertLess(score, self.good())
        self.assertTrue(any("code fence" in w for w in why))

    def test_repetition_is_penalised(self):
        score, why = lmm.verify_answer("q?", "yes it is.\nyes it is.\nyes it is.")
        self.assertLess(score, self.good())
        self.assertTrue(any("repetition" in w for w in why))

    def test_missing_code_block_when_code_requested(self):
        score, why = lmm.verify_answer(
            "write a def solve() function for this.",
            "You should just loop over the list and return it.")
        self.assertTrue(any("code was asked for" in w for w in why))
        self.assertLess(score, 1.0)

    def test_score_stays_in_range(self):
        awful = "As an AI I cannot, maybe.\nAs an AI I cannot, maybe.\n" * 3 + "```"
        score, _ = lmm.verify_answer("write a def f() please", awful)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestPricing(unittest.TestCase):
    def setUp(self):
        self.pricing = lmm.merged_pricing({})

    def test_local_is_always_free(self):
        rate = lmm.price_for({"kind": "local"}, "anything", self.pricing)
        self.assertEqual(rate["in"], 0.0)
        self.assertEqual(rate["out"], 0.0)

    def test_inline_price_wins(self):
        rate = lmm.price_for({"kind": "remote", "price": {"in": 1.5, "out": 4.5}},
                             "whatever", self.pricing)
        self.assertEqual(rate["out"], 4.5)

    def test_named_rate_key(self):
        rate = lmm.price_for({"kind": "remote", "price": "deepseek-chat"},
                             "x", self.pricing)
        self.assertEqual(rate, self.pricing["deepseek-chat"])

    def test_unknown_model_falls_back_to_default(self):
        rate = lmm.price_for({"kind": "remote"}, "totally-unknown-xyz", self.pricing)
        self.assertEqual(rate, self.pricing["default"])

    def test_usage_cost_math(self):
        rate = {"in": 2.0, "out": 10.0, "cw": 0.0, "cr": 0.5}
        usd = lmm.usage_cost({"prompt_tokens": 1_000_000,
                              "completion_tokens": 500_000}, rate)
        self.assertAlmostEqual(usd, 2.0 + 5.0, places=6)

    def test_cached_prompt_tokens_are_billed_at_the_read_rate(self):
        rate = {"in": 2.0, "out": 10.0, "cw": 0.0, "cr": 0.5}
        usd = lmm.usage_cost({"prompt_tokens": 1_000_000, "completion_tokens": 0,
                              "prompt_tokens_details": {"cached_tokens": 1_000_000}},
                             rate)
        self.assertAlmostEqual(usd, 0.5, places=6)

    def test_missing_usage_is_zero_not_a_crash(self):
        self.assertEqual(lmm.usage_cost(None, {"in": 5.0, "out": 5.0}), 0.0)


class TestCache(unittest.TestCase):
    def test_key_ignores_trailing_whitespace(self):
        a = lmm.cache_key([{"role": "user", "content": "hello  \n\n"}], "m")
        b = lmm.cache_key([{"role": "user", "content": "hello"}], "m")
        self.assertEqual(a, b)

    def test_key_preserves_indentation(self):
        a = lmm.cache_key([{"role": "user", "content": "def f():\n    return 1"}], "m")
        b = lmm.cache_key([{"role": "user", "content": "def f():\nreturn 1"}], "m")
        self.assertNotEqual(a, b)

    def test_key_separates_models(self):
        msgs = [{"role": "user", "content": "hi"}]
        self.assertNotEqual(lmm.cache_key(msgs, "a"), lmm.cache_key(msgs, "b"))

    def test_key_covers_the_whole_conversation(self):
        base = [{"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"}]
        other = [{"role": "system", "content": "be verbose"},
                 {"role": "user", "content": "hi"}]
        self.assertNotEqual(lmm.cache_key(base, "m"), lmm.cache_key(other, "m"))

    def test_cosine(self):
        self.assertAlmostEqual(lmm.cosine([1, 0], [1, 0]), 1.0, places=6)
        self.assertAlmostEqual(lmm.cosine([1, 0], [0, 1]), 0.0, places=6)
        self.assertAlmostEqual(lmm.cosine([1, 1], [2, 2]), 1.0, places=6)
        self.assertEqual(lmm.cosine([], [1]), 0.0)          # mismatched lengths
        self.assertEqual(lmm.cosine([0, 0], [0, 0]), 0.0)   # no zero-division

    def test_store_and_exact_lookup_roundtrip(self):
        msgs = [{"role": "user", "content": "what is 2+2"}]
        result = {"choices": [{"message": {"content": "4"}}]}
        with temp_state():
            lmm.cache_store({}, msgs, "m", result, usd=0.01, temperature=0.0)
            entry, how, sim, _cand = lmm.cache_lookup({}, msgs, "m")
            self.assertEqual(how, "exact")
            self.assertEqual(sim, 1.0)
            self.assertEqual(entry["result"], result)

    def test_high_temperature_is_not_cached(self):
        msgs = [{"role": "user", "content": "surprise me"}]
        with temp_state():
            lmm.cache_store({}, msgs, "m", {"ok": 1}, usd=0.0, temperature=0.9)
            entry, how, _, _cand = lmm.cache_lookup({}, msgs, "m")
            self.assertIsNone(entry)
            self.assertIsNone(how)

    def test_expired_entries_are_ignored(self):
        msgs = [{"role": "user", "content": "stale"}]
        with temp_state():
            lmm.cache_store({}, msgs, "m", {"ok": 1}, temperature=0.0)
            # rewrite the entry as if it were written two weeks ago
            with open(lmm.CACHE_LOG, encoding="utf-8") as fh:
                entries = [json.loads(l) for l in fh]
            entries[0]["at"] = time.time() - 15 * 24 * 3600
            with open(lmm.CACHE_LOG, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(entries[0]) + "\n")
            entry, how, _, _cand = lmm.cache_lookup({}, msgs, "m")
            self.assertIsNone(entry)

    def test_disabled_cache_never_hits(self):
        msgs = [{"role": "user", "content": "x"}]
        cfg = {"cache": {"enabled": False}}
        with temp_state():
            lmm.cache_store({}, msgs, "m", {"ok": 1}, temperature=0.0)
            entry, how, _, _cand = lmm.cache_lookup(cfg, msgs, "m")
            self.assertIsNone(entry)


class TestNoFabricatedNumbers(unittest.TestCase):
    """`lmm cost` reports what was measured. A figure derived from an invented
    baseline has no place beside one that was actually billed."""

    def test_cost_report_has_no_cross_provider_estimate(self):
        with temp_state():
            lmm.log_usage({"provider": "p", "kind": "remote", "in": 10,
                           "out": 10, "usd": 1.0, "cache": "miss"})
            report = lmm.cost_report({}, None)
        self.assertNotIn("CROSS-PROVIDER", report)
        self.assertNotIn("illustrative", report)

    def test_the_report_still_shows_what_was_measured(self):
        with temp_state():
            lmm.log_usage({"provider": "p", "kind": "remote", "in": 10,
                           "out": 10, "usd": 1.0, "cache": "miss"})
            report = lmm.cost_report({}, None)
        self.assertIn("HUB MEASURED TOTAL", report)

    def test_breaker_config_is_actually_honoured(self):
        # The key was advertised by `lmm examples` and silently ignored:
        # HUB_BREAKER was built from defaults at import time.
        saved = (lmm.HUB_BREAKER.threshold, lmm.HUB_BREAKER.cooldown_s)
        try:
            brk = lmm.merged_breaker({"breaker": {"threshold": 7,
                                                  "cooldown_s": 90}})
            self.assertEqual(brk["threshold"], 7)
            self.assertEqual(brk["cooldown_s"], 90)
            self.assertTrue(brk["enabled"])          # default preserved
            lmm.HUB_BREAKER.threshold = brk["threshold"]
            lmm.HUB_BREAKER.cooldown_s = brk["cooldown_s"]
            for _ in range(6):
                lmm.HUB_BREAKER.record_failure("p", now=1000.0)
            self.assertEqual(lmm.HUB_BREAKER.state("p", now=1000.0), "closed")
            lmm.HUB_BREAKER.record_failure("p", now=1000.0)   # 7th
            self.assertEqual(lmm.HUB_BREAKER.state("p", now=1000.0), "open")
        finally:
            lmm.HUB_BREAKER.threshold, lmm.HUB_BREAKER.cooldown_s = saved
            lmm.HUB_BREAKER.record_success("p")


class TestToolCalls(unittest.TestCase):
    """A tool-calling reply leaves `content` null and puts its payload in
    `tool_calls`. Treating that as an empty answer turned the cascade from a
    cost reducer into a cost multiplier on exactly the agent traffic a hub
    proxies most."""

    CALL = {"id": "c1", "type": "function",
            "function": {"name": "get_weather",
                         "arguments": '{"location":"Tokyo"}'}}

    def response(self, calls=None, content=None):
        return {"choices": [{"message": {"role": "assistant", "content": content,
                                         "tool_calls": calls},
                             "finish_reason": "tool_calls" if calls else "stop"}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 20}}

    def test_a_valid_tool_call_scores_as_a_complete_answer(self):
        score, why = lmm.verify_tool_calls([self.CALL])
        self.assertGreaterEqual(score, lmm.DEFAULT_CASCADE["threshold"])
        self.assertEqual(why, [])

    def test_verify_answer_dispatches_on_tool_calls(self):
        score, _ = lmm.verify_answer("weather?", None, tool_calls=[self.CALL])
        self.assertGreaterEqual(score, lmm.DEFAULT_CASCADE["threshold"])

    def test_unparseable_arguments_still_escalate(self):
        # The tool-call analogue of an unclosed code fence, and the documented
        # failure mode of small models emitting structured output.
        bad = {"function": {"name": "get_weather", "arguments": '{"location":"Tok'}}
        score, why = lmm.verify_tool_calls([bad])
        self.assertLess(score, lmm.DEFAULT_CASCADE["threshold"])
        self.assertTrue(any("JSON" in w for w in why))

    def test_a_nameless_tool_call_is_a_total_failure(self):
        self.assertEqual(lmm.verify_tool_calls([{"function": {"arguments": "{}"}}])[0],
                         0.0)

    def test_a_no_argument_tool_is_legitimate(self):
        score, why = lmm.verify_tool_calls([{"function": {"name": "now",
                                                          "arguments": ""}}])
        self.assertGreaterEqual(score, lmm.DEFAULT_CASCADE["threshold"])
        self.assertEqual(why, [])

    def test_empty_tool_calls_is_not_an_answer(self):
        self.assertEqual(lmm.verify_tool_calls([])[0], 0.0)

    def test_helpers_tolerate_junk(self):
        self.assertEqual(lmm.tool_calls_of({}), [])
        self.assertEqual(lmm.tool_calls_of({"choices": []}), [])
        self.assertEqual(lmm.message_of(None), {})
        self.assertEqual(lmm.tool_calls_text(None), "")

    def test_cascade_accepts_a_tool_call_at_the_cheapest_rung(self):
        calls = []

        def fake(prov, prompt, temperature=0.7, extra=None, **kw):
            calls.append(prov.get("model"))
            return self.response([self.CALL])
        saved = lmm.call_provider
        lmm.call_provider = fake
        targets = [("cheap", {"kind": "remote", "model": "cheap",
                              "price": {"in": 0.1, "out": 0.2}}),
                   ("strong", {"kind": "remote", "model": "strong",
                               "price": {"in": 15.0, "out": 75.0}})]
        try:
            with temp_state():
                res, trace = lmm.hub_complete({}, "weather in Tokyo?", targets,
                                              {"cascade": True, "cache": False})
        finally:
            lmm.call_provider = saved
        self.assertEqual(calls, ["cheap"])          # no needless escalation
        self.assertTrue(lmm.tool_calls_of(res))     # the tool call is returned
        self.assertTrue(any("accept" in t for t in trace))

    def test_a_response_with_neither_content_nor_tool_calls_is_still_rejected(self):
        saved = lmm.call_provider
        lmm.call_provider = lambda p, m, temperature=0.7, extra=None, **kw: {
            "choices": [{"message": {"role": "assistant"}}]}
        targets = [("p", {"kind": "remote", "model": "p"})]
        try:
            with temp_state():
                res, _ = lmm.hub_complete({}, "hi", targets,
                                          {"cascade": False, "cache": False})
        finally:
            lmm.call_provider = saved
        self.assertIn("error", res)

    def test_chunk_tool_text_extracts_streamed_arguments(self):
        chunk = {"choices": [{"delta": {"tool_calls": [
            {"function": {"name": "f", "arguments": '{"a":1}'}}]}}]}
        self.assertEqual(lmm.chunk_tool_text(chunk), 'f\n{"a":1}')
        self.assertEqual(lmm.chunk_tool_text({"choices": [{"delta": {}}]}), "")
        self.assertEqual(lmm.chunk_tool_text({}), "")

    def test_a_streamed_tool_call_is_not_metered_as_zero_output(self):
        def stream(prov, prompt, temperature=0.7, extra=None, **kw):
            frames = [
                {"choices": [{"delta": {"role": "assistant"}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "function": {"name": "get_weather",
                                              "arguments": ""}}]}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "function": {
                        "arguments": '{"location":"Tokyo"}'}}]}}]},
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            ]
            for f in frames:
                yield ("data: " + json.dumps(f) + "\n").encode(), f
            yield None, None
        saved = lmm.call_provider_stream
        lmm.call_provider_stream = stream
        targets = [("p", {"kind": "remote", "model": "p",
                          "price": {"in": 1.0, "out": 2.0}})]
        try:
            with temp_state():
                b"".join(lmm.hub_stream({}, "weather?", targets, {"cache": False}))
                events = lmm.read_usage()
        finally:
            lmm.call_provider_stream = saved
        self.assertGreater(events[0]["out"], 0)     # was 0 before
        self.assertTrue(events[0]["estimated"])
        self.assertIsNotNone(events[0]["ttft_ms"])  # was None before


class TestVerifiedCache(unittest.TestCase):
    """vCache (arXiv:2502.03771): one static similarity threshold cannot bound
    the false-hit rate, so an entry has to EARN the right to answer."""

    def test_wilson_is_conservative_on_small_samples(self):
        # The naive estimate says 2/2 == 100% correct. Certifying an entry on
        # two lucky draws is exactly the overconfidence this guards against.
        self.assertLess(lmm.wilson_lower_bound(2, 2), 0.8)
        self.assertEqual(lmm.wilson_lower_bound(0, 0), 0.0)

    def test_wilson_tightens_as_evidence_accumulates(self):
        a = lmm.wilson_lower_bound(9, 10)
        b = lmm.wilson_lower_bound(90, 100)
        c = lmm.wilson_lower_bound(900, 1000)
        self.assertLess(a, b)
        self.assertLess(b, c)
        self.assertLess(c, 0.9)          # never exceeds the observed rate

    def test_wilson_bounds_are_in_range(self):
        for k, n in ((0, 5), (5, 5), (3, 7), (1, 100)):
            v = lmm.wilson_lower_bound(k, n)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_evidence_only_counts_observations_at_or_below_this_similarity(self):
        # An observation made at sim 0.99 says nothing reassuring about serving
        # at 0.90, so it must not be counted toward it.
        entry = {"obs": [[0.99, 1], [0.95, 1], [0.85, 0]]}
        trials, ok = lmm.entry_evidence(entry, 0.90)
        self.assertEqual((trials, ok), (1, 0))       # only the 0.85 one
        trials, ok = lmm.entry_evidence(entry, 0.99)
        self.assertEqual((trials, ok), (3, 2))

    def test_static_mode_is_the_default_and_unchanged(self):
        conf = lmm.merged_cache({})
        self.assertIsNone(conf["max_error_rate"])
        ok, why = lmm.certified({}, 0.96, conf)
        self.assertTrue(ok)
        self.assertIn("static", why)
        self.assertFalse(lmm.certified({}, 0.90, conf)[0])

    def test_uncertified_entry_is_refused_until_it_has_evidence(self):
        conf = lmm.merged_cache({"cache": {"max_error_rate": 0.05}})
        entry = {"obs": [[0.9, 1]]}                  # one sample
        ok, why = lmm.certified(entry, 0.9, conf)
        self.assertFalse(ok)
        self.assertIn("observations", why)

    def test_entry_certifies_once_enough_agreement_accumulates(self):
        conf = lmm.merged_cache({"cache": {"max_error_rate": 0.30,
                                           "min_observations": 3}})
        entry = {"obs": [[0.9, 1]] * 40}
        ok, why = lmm.certified(entry, 0.9, conf)
        self.assertTrue(ok, why)

    def test_disagreements_keep_an_entry_uncertified(self):
        conf = lmm.merged_cache({"cache": {"max_error_rate": 0.05,
                                           "min_observations": 3}})
        entry = {"obs": [[0.9, 1], [0.9, 0], [0.9, 1], [0.9, 0], [0.9, 1]]}
        ok, why = lmm.certified(entry, 0.9, conf)
        self.assertFalse(ok)
        self.assertIn("lower bound", why)

    def test_a_stricter_error_bound_certifies_less(self):
        entry = {"obs": [[0.9, 1]] * 20}
        loose = lmm.merged_cache({"cache": {"max_error_rate": 0.30}})
        tight = lmm.merged_cache({"cache": {"max_error_rate": 0.001}})
        self.assertTrue(lmm.certified(entry, 0.9, loose)[0])
        self.assertFalse(lmm.certified(entry, 0.9, tight)[0])

    def test_answers_agree_short_circuits_on_identical_text(self):
        # No embedder needed, and none should be called.
        saved = lmm.embed_text
        lmm.embed_text = lambda *a, **k: self.fail("should not embed")
        try:
            self.assertTrue(lmm.answers_agree("same", "same", lmm.merged_cache({})))
        finally:
            lmm.embed_text = saved

    def test_answers_agree_is_none_without_an_embedder(self):
        # No label is not the same as a wrong label; an unavailable embedder
        # must not be recorded as a disagreement.
        saved = lmm.embed_text
        lmm.embed_text = lambda *a, **k: None
        try:
            self.assertIsNone(lmm.answers_agree("a", "b", lmm.merged_cache({})))
        finally:
            lmm.embed_text = saved

    def test_answers_agree_uses_the_answer_match_threshold(self):
        saved = lmm.embed_text
        lmm.embed_text = lambda text, model: [1.0, 0.0] if text == "a" else [0.9, 0.44]
        try:
            conf = lmm.merged_cache({"cache": {"answer_match": 0.80}})
            self.assertTrue(lmm.answers_agree("a", "b", conf))
            conf = lmm.merged_cache({"cache": {"answer_match": 0.999}})
            self.assertFalse(lmm.answers_agree("a", "b", conf))
        finally:
            lmm.embed_text = saved

    def test_observations_persist_onto_the_entry(self):
        msgs = [{"role": "user", "content": "q"}]
        with temp_state():
            lmm.cache_store({}, msgs, "m", {"ok": 1}, temperature=0.0)
            key = lmm.cache_key(msgs, "m")
            lmm.record_observation({}, key, 0.93, True)
            lmm.record_observation({}, key, 0.91, False)
            entries = lmm.cache_entries(lmm.DEFAULT_CACHE)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["obs"], [[0.93, 1], [0.91, 0]])

    def test_lookup_returns_an_explore_candidate_instead_of_a_hit(self):
        # The near neighbour must never come back as a servable entry.
        msgs_a = [{"role": "user", "content": "what is the capital of France"}]
        msgs_b = [{"role": "user", "content": "capital of France?"}]
        cfg = {"cache": {"semantic": True, "max_error_rate": 0.05,
                         "similarity": 0.5}}
        saved = lmm.embed_text
        lmm.embed_text = lambda text, model: [1.0, 0.05] if "capital" in text else [0.0, 1.0]
        try:
            with temp_state():
                lmm.cache_store(cfg, msgs_a, "m", {"ok": 1}, temperature=0.0)
                entry, how, sim, cand = lmm.cache_lookup(cfg, msgs_b, "m")
        finally:
            lmm.embed_text = saved
        self.assertIsNone(entry)                 # not servable
        self.assertEqual(how, "explore")
        self.assertIsNotNone(cand)               # but available to label
        self.assertGreater(sim, 0.5)

    def test_label_exploration_records_agreement(self):
        msgs = [{"role": "user", "content": "q"}]
        answer = {"choices": [{"message": {"content": "the answer is 4"}}]}
        with temp_state():
            lmm.cache_store({}, msgs, "m", answer, temperature=0.0)
            cand = lmm.cache_entries(lmm.DEFAULT_CACHE)[0]
            lmm.label_exploration({}, cand, 0.93, answer)   # identical text
            entries = lmm.cache_entries(lmm.DEFAULT_CACHE)
        self.assertEqual(entries[0]["obs"], [[0.93, 1]])

    def test_label_exploration_records_disagreement(self):
        msgs = [{"role": "user", "content": "q"}]
        cached = {"choices": [{"message": {"content": "four"}}]}
        fresh = {"choices": [{"message": {"content": "seventeen"}}]}
        saved = lmm.embed_text
        lmm.embed_text = lambda text, model: [1.0, 0.0] if "four" in text else [0.0, 1.0]
        try:
            with temp_state():
                lmm.cache_store({}, msgs, "m", cached, temperature=0.0)
                cand = lmm.cache_entries(lmm.DEFAULT_CACHE)[0]
                lmm.label_exploration({}, cand, 0.93, fresh)
                entries = lmm.cache_entries(lmm.DEFAULT_CACHE)
        finally:
            lmm.embed_text = saved
        self.assertEqual(entries[0]["obs"], [[0.93, 0]])

    def test_label_exploration_is_a_noop_without_a_candidate(self):
        with temp_state():
            lmm.label_exploration({}, None, 0.9, {"x": 1})   # must not raise

    def drive(self, cfg, fresh, rounds=30):
        """Run the explore/label loop and report the round it certified on."""
        A = [{"role": "user", "content": "capital of France"}]
        B = [{"role": "user", "content": "France capital?"}]
        cached = {"choices": [{"message": {"content": "Paris."}}]}
        lmm.cache_store(cfg, A, "m", cached, usd=0.02, temperature=0.0)
        for r in range(1, rounds + 1):
            entry, how, sim, cand = lmm.cache_lookup(cfg, B, "m")
            if entry:
                return r
            if cand:
                lmm.label_exploration(cfg, cand, sim, fresh)
        return None

    def test_an_interchangeable_neighbour_earns_certification(self):
        cfg = {"cache": {"semantic": True, "max_error_rate": 0.20,
                         "min_observations": 3, "similarity": 0.80}}
        same = {"choices": [{"message": {"content": "Paris."}}]}
        saved = lmm.embed_text
        lmm.embed_text = lambda t, m: [1.0, 0.06]
        try:
            with temp_state():
                got = self.drive(cfg, same)
        finally:
            lmm.embed_text = saved
        self.assertIsNotNone(got, "a consistently-agreeing entry never certified")
        self.assertGreater(got, 3)          # not on the first lucky samples

    def test_a_neighbour_whose_answers_differ_is_never_certified(self):
        # THE point of vCache. Similarity is high enough that a static
        # threshold would have served this from the very first request — and
        # been wrong every single time.
        cfg = {"cache": {"semantic": True, "max_error_rate": 0.20,
                         "min_observations": 3, "similarity": 0.80}}
        different = {"choices": [{"message": {"content": "Something else."}}]}
        saved = lmm.embed_text

        def emb(t, m):
            if "Paris" in t:
                return [1.0, 0.0]
            if "else" in t:
                return [0.0, 1.0]
            return [1.0, 0.06]
        lmm.embed_text = emb
        try:
            with temp_state():
                got = self.drive(cfg, different)
                # and the static threshold would have served it immediately
                static = dict(cfg["cache"])
                static["max_error_rate"] = None
                ok, _ = lmm.certified({}, 0.99, lmm.merged_cache({"cache": static}))
        finally:
            lmm.embed_text = saved
        self.assertIsNone(got, "certified an entry whose answers disagreed")
        self.assertTrue(ok, "static mode would indeed have served it")


class TestMessagesHandling(unittest.TestCase):
    def test_as_messages_wraps_a_string(self):
        self.assertEqual(lmm.as_messages("hi"),
                         [{"role": "user", "content": "hi"}])

    def test_as_messages_passes_arrays_through(self):
        msgs = [{"role": "system", "content": "a"}, {"role": "user", "content": "b"}]
        self.assertEqual(lmm.as_messages(msgs), msgs)

    def test_messages_text_joins_every_turn(self):
        msgs = [{"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"}]
        self.assertEqual(lmm.messages_text(msgs), "be terse\nhi")

    def test_messages_text_handles_content_parts(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "part1"},
                                             {"type": "image_url", "image_url": {}}]}]
        self.assertEqual(lmm.messages_text(msgs), "part1")


class TestHubComplete(unittest.TestCase):
    """The unified path: cache -> route -> cascade -> meter -> store."""

    def setUp(self):
        self.calls = []
        self._real = lmm.call_provider

    def tearDown(self):
        lmm.call_provider = self._real

    def fake(self, answers):
        """Patch call_provider to serve canned answers per provider model."""
        def _call(prov, prompt, temperature=0.7, extra=None):
            self.calls.append(prov.get("model"))
            text = answers.get(prov.get("model"), "ok fine answer here.")
            return {"choices": [{"message": {"content": text}}],
                    "usage": {"prompt_tokens": 1000, "completion_tokens": 1000}}
        lmm.call_provider = _call

    def targets(self):
        return [("weak", {"kind": "remote", "model": "weak",
                          "price": {"in": 0.1, "out": 0.1}}),
                ("strong", {"kind": "remote", "model": "strong",
                            "price": {"in": 10.0, "out": 30.0}})]

    def test_cascade_stops_at_the_cheap_rung_when_the_answer_is_good(self):
        self.fake({"weak": "The answer is 4, because 2 plus 2 equals 4."})
        with temp_state():
            res, trace = lmm.hub_complete({}, "what is 2+2", self.targets(),
                                          {"cascade": True, "cache": False})
        self.assertEqual(self.calls, ["weak"])
        self.assertIn("4", res["choices"][0]["message"]["content"])
        self.assertTrue(any("accept" in t for t in trace))

    def test_cascade_escalates_when_the_cheap_rung_refuses(self):
        self.fake({"weak": "As an AI, I cannot answer that.",
                   "strong": "The answer is 4, because 2 plus 2 equals 4."})
        with temp_state():
            res, trace = lmm.hub_complete({}, "what is 2+2", self.targets(),
                                          {"cascade": True, "cache": False})
        self.assertEqual(self.calls, ["weak", "strong"])
        self.assertIn("4", res["choices"][0]["message"]["content"])

    def test_cascade_returns_the_best_scoring_answer_when_none_clear(self):
        self.fake({"weak": "", "strong": "As an AI, I cannot, maybe"})
        with temp_state():
            res, trace = lmm.hub_complete({}, "what is 2+2", self.targets(),
                                          {"cascade": True, "cache": False})
        self.assertTrue(any("returning best" in t for t in trace))
        self.assertIn("choices", res)

    def test_a_private_prompt_never_escalates_to_the_cloud(self):
        # A low score is not a reason to ship a secret off the machine.
        self.fake({"local": "As an AI, I cannot answer that."})
        targets = [("local", {"kind": "local", "model": "local"})] + self.targets()
        with temp_state():
            lmm.hub_complete({}, "summarize this secret memo", targets,
                             {"cascade": True, "cache": False})
        self.assertEqual(self.calls, ["local"])

    def test_private_prompt_with_no_local_provider_is_refused(self):
        """The old behaviour warned in the trace and sent the prompt anyway —
        and the trace prints AFTER the request returns, so the warning was a
        eulogy. A pin that yields is not a pin: no local provider means the
        request is refused, and nothing is sent anywhere."""
        self.fake({})
        with temp_state():
            res, trace = lmm.hub_complete({}, "summarize this secret memo",
                                          self.targets(),
                                          {"cascade": False, "cache": False})
        self.assertIn("refusing", res.get("error", ""))
        self.assertEqual(self.calls, [], "the private prompt was sent out")

    def test_no_warning_when_a_local_provider_exists(self):
        self.fake({})
        targets = [("local", {"kind": "local", "model": "local"})] + self.targets()
        with temp_state():
            _res, trace = lmm.hub_complete({}, "summarize this secret memo",
                                           targets,
                                           {"cascade": False, "cache": False})
        self.assertFalse(any(t.startswith("[warn]") for t in trace))

    def test_cascade_rungs_are_cheapest_first(self):
        rungs = lmm.cascade_rungs({}, list(reversed(self.targets())))
        self.assertEqual([n for n, _ in rungs], ["weak", "strong"])

    def test_cascade_rungs_respect_max_rungs(self):
        cfg = {"cascade": {"max_rungs": 1}}
        rungs = lmm.cascade_rungs(cfg, self.targets())
        self.assertEqual(len(rungs), 1)

    def test_explicit_rungs_win_over_price_order(self):
        cfg = {"cascade": {"rungs": ["strong", "weak"]}}
        rungs = lmm.cascade_rungs(cfg, self.targets())
        self.assertEqual([n for n, _ in rungs], ["strong", "weak"])

    def test_non_cascade_uses_one_provider_only(self):
        self.fake({})
        with temp_state():
            lmm.hub_complete({}, "hi", self.targets(),
                             {"cascade": False, "cache": False})
        self.assertEqual(len(self.calls), 1)

    def test_easy_prompt_reaches_the_cheap_provider_first(self):
        self.fake({})
        with temp_state():
            lmm.hub_complete({}, "hi", self.targets(),
                             {"cascade": False, "cache": False})
        self.assertEqual(self.calls, ["weak"])

    def test_second_identical_call_is_served_from_cache(self):
        self.fake({"weak": "The answer is 4, because 2 plus 2 equals 4."})
        opts = {"cascade": False, "cache": True, "extra": {"temperature": 0.0}}
        with temp_state():
            lmm.hub_complete({}, "what is 2+2", self.targets(), opts)
            self.assertEqual(len(self.calls), 1)
            res, trace = lmm.hub_complete({}, "what is 2+2", self.targets(), opts)
        self.assertEqual(len(self.calls), 1)          # no second upstream call
        self.assertTrue(any("cache" in t for t in trace))
        self.assertIn("4", res["choices"][0]["message"]["content"])

    def test_cache_works_without_an_explicit_temperature(self):
        # lmm's own 0.7 fallback must not be mistaken for the caller asking
        # for variety, or the cache would never store anything by default.
        self.fake({"weak": "The answer is 4, because 2 plus 2 equals 4."})
        opts = {"cascade": False, "cache": True}
        with temp_state():
            lmm.hub_complete({}, "what is 2+2", self.targets(), opts)
            _res, trace = lmm.hub_complete({}, "what is 2+2", self.targets(), opts)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(any("cache" in t for t in trace))

    def test_explicit_high_temperature_is_not_cached(self):
        self.fake({"weak": "The answer is 4, because 2 plus 2 equals 4."})
        opts = {"cascade": False, "cache": True, "extra": {"temperature": 1.0}}
        with temp_state():
            lmm.hub_complete({}, "surprise me", self.targets(), opts)
            lmm.hub_complete({}, "surprise me", self.targets(), opts)
        self.assertEqual(len(self.calls), 2)

    def test_a_cascade_answer_is_cached_under_the_request_key(self):
        # The answer arrives from rung 1 but the next identical question must
        # still hit, so the key identifies the request, not the responder.
        self.fake({"weak": "As an AI, I cannot answer that.",
                   "strong": "The answer is 4, because 2 plus 2 equals 4."})
        opts = {"cascade": True, "cache": True}
        with temp_state():
            lmm.hub_complete({}, "what is 2+2", self.targets(), opts)
            self.assertEqual(self.calls, ["weak", "strong"])
            res, trace = lmm.hub_complete({}, "what is 2+2", self.targets(), opts)
        self.assertEqual(self.calls, ["weak", "strong"])   # no new upstream calls
        self.assertIn("4", res["choices"][0]["message"]["content"])
        self.assertTrue(any("cache" in t for t in trace))

    def test_every_call_is_metered(self):
        self.fake({})
        with temp_state():
            lmm.hub_complete({}, "hi", self.targets(),
                             {"cascade": False, "cache": False})
            events = lmm.read_usage()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["in"], 1000)
        self.assertEqual(events[0]["out"], 1000)
        # weak provider: 1000/1e6*0.1 + 1000/1e6*0.1
        self.assertAlmostEqual(events[0]["usd"], 0.0002, places=6)

    def test_provider_error_falls_through_to_the_next(self):
        def _call(prov, prompt, temperature=0.7, extra=None):
            self.calls.append(prov.get("model"))
            if prov.get("model") == "weak":
                return {"error": "401 unauthorized"}
            return {"choices": [{"message": {"content": "fine answer from strong."}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10}}
        lmm.call_provider = _call
        with temp_state():
            res, _ = lmm.hub_complete({}, "hi", self.targets(),
                                      {"cascade": False, "cache": False})
        self.assertEqual(self.calls, ["weak", "strong"])
        self.assertIn("strong", res["choices"][0]["message"]["content"])

    def test_all_providers_failing_surfaces_an_error(self):
        lmm.call_provider = lambda p, m, temperature=0.7, extra=None: {"error": "down"}
        with temp_state():
            res, _ = lmm.hub_complete({}, "hi", self.targets(),
                                      {"cascade": False, "cache": False})
        self.assertIn("error", res)


class TestConfigMerging(unittest.TestCase):
    def test_cascade_defaults_are_overlaid_not_replaced(self):
        got = lmm.merged_cascade({"cascade": {"threshold": 0.9}})
        self.assertEqual(got["threshold"], 0.9)
        self.assertEqual(got["max_rungs"], lmm.DEFAULT_CASCADE["max_rungs"])

    def test_cache_defaults_are_overlaid_not_replaced(self):
        got = lmm.merged_cache({"cache": {"semantic": True}})
        self.assertTrue(got["semantic"])
        self.assertEqual(got["similarity"], lmm.DEFAULT_CACHE["similarity"])

    def test_semantic_cache_is_off_by_default(self):
        # vCache (arXiv:2502.03771): a static similarity threshold cannot bound
        # false hits, so fuzzy matching must be a deliberate choice.
        self.assertFalse(lmm.merged_cache({})["semantic"])

    def test_provider_price_is_optional(self):
        provs = lmm.merged_providers({"providers": {"p": {"model": "m"}}})
        self.assertIsNone(provs["p"]["price"])
        self.assertEqual(provs["p"]["kind"], "remote")

    def test_examples_output_is_valid_json(self):
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            frontend.cmd_examples()
        json.loads(buf.getvalue())


class TestVramFit(unittest.TestCase):
    """The KV-cache formula, checked against published figures.

    kv = 2 * layers * kv_heads * head_dim * ctx * bytes_per_element
    """

    LLAMA3_8B = {"params": 8.03e9, "layers": 32, "kv_heads": 8, "head_dim": 128,
                 "ctx_max": 131072, "quant": "q4_k_m"}
    # Llama-2 7B predates GQA: 32 query heads AND 32 KV heads.
    LLAMA2_7B = {"params": 6.74e9, "layers": 32, "kv_heads": 32, "head_dim": 128,
                 "ctx_max": 4096, "quant": "f16"}

    def test_llama3_8b_kv_at_32k_matches_published_4gib(self):
        est = lmm.estimate_vram(self.LLAMA3_8B, 32768, kv_type="f16")
        self.assertAlmostEqual(est["kv_gib"], 4.0, places=3)

    def test_llama2_7b_kv_at_4k_matches_published_2gib(self):
        est = lmm.estimate_vram(self.LLAMA2_7B, 4096, kv_type="f16")
        self.assertAlmostEqual(est["kv_gib"], 2.0, places=3)

    def test_gqa_head_count_matters_fourfold(self):
        # Llama 3.1 8B has 32 query heads but 8 KV heads; using the query count
        # would overstate the KV cache by 4x. This is the easiest thing to get
        # wrong in the formula, so it is pinned.
        mha = dict(self.LLAMA3_8B, kv_heads=32)
        a = lmm.estimate_vram(self.LLAMA3_8B, 8192)["kv_gib"]
        b = lmm.estimate_vram(mha, 8192)["kv_gib"]
        self.assertAlmostEqual(b / a, 4.0, places=6)

    def test_kv_quantization_halves_and_quarters(self):
        f16 = lmm.estimate_vram(self.LLAMA3_8B, 32768, kv_type="f16")["kv_gib"]
        q8 = lmm.estimate_vram(self.LLAMA3_8B, 32768, kv_type="q8_0")["kv_gib"]
        q4 = lmm.estimate_vram(self.LLAMA3_8B, 32768, kv_type="q4_0")["kv_gib"]
        self.assertAlmostEqual(q8, f16 / 2, places=6)
        self.assertAlmostEqual(q4, f16 / 4, places=6)

    def test_kv_scales_linearly_with_context(self):
        a = lmm.estimate_vram(self.LLAMA3_8B, 4096)["kv_gib"]
        b = lmm.estimate_vram(self.LLAMA3_8B, 8192)["kv_gib"]
        self.assertAlmostEqual(b, a * 2, places=6)

    def test_weights_track_bits_per_weight(self):
        q4 = lmm.estimate_vram(self.LLAMA3_8B, 0, quant="q4_k_m")["weights_gib"]
        q8 = lmm.estimate_vram(self.LLAMA3_8B, 0, quant="q8_0")["weights_gib"]
        f16 = lmm.estimate_vram(self.LLAMA3_8B, 0, quant="f16")["weights_gib"]
        self.assertLess(q4, q8)
        self.assertLess(q8, f16)
        # 8B at q4_k_m lands near the ~4.6 GiB real GGUF file size
        self.assertGreater(q4, 4.0)
        self.assertLess(q4, 5.2)

    def test_total_is_weights_plus_kv_plus_overhead(self):
        est = lmm.estimate_vram(self.LLAMA3_8B, 8192)
        self.assertAlmostEqual(
            est["total_gib"],
            est["weights_gib"] + est["kv_gib"] + est["overhead_gib"], places=9)

    def test_max_context_round_trips(self):
        budget = 24.0
        ctx = lmm.max_context_for(self.LLAMA3_8B, budget)
        self.assertGreater(ctx, 0)
        self.assertLessEqual(lmm.estimate_vram(self.LLAMA3_8B, ctx)["total_gib"],
                             budget + 1e-6)
        # one token past the limit must not fit
        self.assertGreater(
            lmm.estimate_vram(self.LLAMA3_8B, ctx + 2)["total_gib"], budget)

    def test_max_context_is_zero_when_weights_alone_overflow(self):
        self.assertEqual(lmm.max_context_for(self.LLAMA3_8B, 1.0), 0)

    def test_kv_quantization_buys_context(self):
        f16 = lmm.max_context_for(self.LLAMA3_8B, 8.0, kv_type="f16")
        q4 = lmm.max_context_for(self.LLAMA3_8B, 8.0, kv_type="q4_0")
        self.assertAlmostEqual(q4 / f16, 4.0, delta=0.01)

    def test_quant_bpw_lookup(self):
        self.assertEqual(lmm.quant_bpw("Q4_K_M"), lmm.QUANT_BPW["q4_k_m"])
        self.assertEqual(lmm.quant_bpw("F16"), 16.0)
        # unknown quant falls back to Ollama's default pull, not a crash
        self.assertEqual(lmm.quant_bpw("wat"), lmm.QUANT_BPW["q4_k_m"])
        self.assertEqual(lmm.quant_bpw(None), lmm.QUANT_BPW["q4_k_m"])

    def test_parse_params(self):
        self.assertAlmostEqual(lmm.parse_params("8.0B"), 8.0e9)
        self.assertAlmostEqual(lmm.parse_params("134.52M"), 134.52e6)
        self.assertAlmostEqual(lmm.parse_params("7b"), 7.0e9)
        self.assertEqual(lmm.parse_params("garbage"), 0.0)
        self.assertEqual(lmm.parse_params(None), 0.0)

    def test_params_from_name_ignores_version_numbers(self):
        self.assertAlmostEqual(lmm.params_from_name("qwen2.5-coder:7b"), 7.0e9)
        self.assertAlmostEqual(lmm.params_from_name("llama3.1:70b"), 70.0e9)
        self.assertEqual(lmm.params_from_name("mistral"), 0.0)


def gguf_string(v):
    b = v.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def gguf_kv_u32(key, val):
    return gguf_string(key) + struct.pack("<II", lmm.GGUF_UINT32, val)


def gguf_kv_str(key, val):
    return gguf_string(key) + struct.pack("<I", lmm.GGUF_STRING) + gguf_string(val)


def write_gguf(path, arch="llama", layers=32, heads=32, kv_heads=8,
               embed=4096, ctx=131072, key_length=None, version=3,
               endian="<", pad_to_bpw=4.85, tensor_dims=(4096, 1960000),
               big_endian_magic=False, drop_kv=()):
    """Write a minimal but spec-valid GGUF with one tensor, for offline tests.

    Only the metadata `fit` reads is emitted. `tensor_dims` sets the parameter
    count (product of dims), and the file is padded so size/params lands near
    `pad_to_bpw`, simulating a realistic measured bits-per-weight.

    Defaults reproduce Llama-3-8B: 8.03B parameters at ~4.85 bpw, i.e. a 4.5 GB
    file. The padding is written by seeking rather than writing zeros, so the
    file is SPARSE — the size is real (which is what read_gguf measures) but it
    occupies almost no disk. lmm only ever reads the header, so it never
    touches the hole.
    """
    e = endian
    def s(v):
        b = v.encode("utf-8")
        return struct.pack(e + "Q", len(b)) + b
    def kvu(k, v):
        return s(k) + struct.pack(e + "II", lmm.GGUF_UINT32, v)
    def kvs(k, v):
        return s(k) + struct.pack(e + "I", lmm.GGUF_STRING) + s(v)

    pairs = []
    pairs.append(("arch", kvs("general.architecture", arch)))
    pairs.append(("layers", kvu("%s.block_count" % arch, layers)))
    pairs.append(("heads", kvu("%s.attention.head_count" % arch, heads)))
    if kv_heads is not None:
        pairs.append(("kv_heads",
                      kvu("%s.attention.head_count_kv" % arch, kv_heads)))
    pairs.append(("embed", kvu("%s.embedding_length" % arch, embed)))
    pairs.append(("ctx", kvu("%s.context_length" % arch, ctx)))
    if key_length is not None:
        pairs.append(("key_length",
                      kvu("%s.attention.key_length" % arch, key_length)))
    meta = b"".join(body for name, body in pairs if name not in drop_kv)
    n_kv = sum(1 for name, _ in pairs if name not in drop_kv)

    dims = tensor_dims
    tensor = (s("token_embd.weight") + struct.pack(e + "I", len(dims))
              + b"".join(struct.pack(e + "Q", d) for d in dims)
              + struct.pack(e + "I", 12) + struct.pack(e + "Q", 0))
    magic = lmm.GGUF_MAGIC[::-1] if big_endian_magic else lmm.GGUF_MAGIC
    header = magic + struct.pack(e + "IQQ", version, 1, n_kv)
    blob = header + meta + tensor
    params = 1
    for d in dims:
        params *= d
    target = int(params * pad_to_bpw / 8) if pad_to_bpw else len(blob)
    with open(path, "wb") as fh:
        fh.write(blob)
        if target > len(blob):
            fh.seek(target - 1)      # sparse: real size, no disk
            fh.write(b"\0")
    return path


class TestGguf(unittest.TestCase):
    """Reading the model geometry straight from a .gguf file — the path that
    lets LM Studio / llama.cpp / KoboldCPP users run `fit` with no runtime.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="lmm-gguf-")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def path(self, name="m.gguf", **kw):
        return write_gguf(os.path.join(self.dir, name), **kw)

    def test_reads_the_attention_geometry(self):
        got = lmm.read_gguf(self.path())
        self.assertNotIn("error", got)
        self.assertEqual(got["layers"], 32)
        self.assertEqual(got["kv_heads"], 8)
        self.assertEqual(got["head_dim"], 128)
        self.assertEqual(got["ctx_max"], 131072)
        self.assertEqual(got["arch"], "llama")
        self.assertEqual(got["source"], "gguf")

    def test_parameter_count_is_summed_from_the_tensor_table(self):
        # Exact, not parsed from a name like "7b" — that is the whole point of
        # reading the file rather than guessing from a tag.
        got = lmm.read_gguf(self.path())
        self.assertEqual(got["params"], 4096 * 1960000)     # 8.03B, exact
        got = lmm.read_gguf(self.path(name="b.gguf", tensor_dims=(1024, 64, 2)))
        self.assertEqual(got["params"], 1024 * 64 * 2)      # 3-D tensor

    def test_kv_heads_fall_back_to_head_count_without_gqa(self):
        # A pre-GQA model omits head_count_kv; it must default to head_count,
        # not to zero (which would make the KV cache vanish).
        got = lmm.read_gguf(self.path(kv_heads=None))
        self.assertEqual(got["kv_heads"], got["head_dim"] and 32)

    def test_head_dim_prefers_key_length_over_embed_div_heads(self):
        got = lmm.read_gguf(self.path(key_length=96, embed=4096, heads=32))
        self.assertEqual(got["head_dim"], 96)      # not 4096/32 == 128

    def test_measured_bits_per_weight_matches_the_padding(self):
        got = lmm.read_gguf(self.path(pad_to_bpw=4.85))
        self.assertAlmostEqual(got["measured_bpw"], 4.85, delta=0.05)
        self.assertEqual(got["quant"], "~q4_k_m")

    def test_weights_are_exact_from_the_file(self):
        p = self.path(pad_to_bpw=5.5)
        got = lmm.read_gguf(p)
        self.assertAlmostEqual(got["weights_gib"],
                               os.path.getsize(p) / lmm.GIB, places=6)
        est = lmm.estimate_vram(got, 0)
        self.assertAlmostEqual(est["weights_gib"], got["weights_gib"], places=6)

    def test_quant_whatif_ignores_the_measured_weights(self):
        # Asking "what if this were q8_0" must recompute from params x bpw, not
        # reuse the file's actual size.
        got = lmm.read_gguf(self.path(pad_to_bpw=4.85))
        as_is = lmm.estimate_vram(got, 0)["weights_gib"]
        whatif = lmm.estimate_vram(got, 0, quant="q8_0")["weights_gib"]
        self.assertGreater(whatif, as_is)

    def test_published_kv_figure_holds_for_a_gguf_source(self):
        # The whole KV formula is pinned to Llama-3-8B @32K fp16 = 4.0 GiB;
        # reading the spec from a file must not change that.
        got = lmm.read_gguf(self.path())
        self.assertAlmostEqual(lmm.estimate_vram(got, 32768)["kv_gib"], 4.0,
                               places=3)

    def test_big_endian_magic_is_accepted(self):
        got = lmm.read_gguf(self.path(endian=">", big_endian_magic=True))
        self.assertNotIn("error", got)
        self.assertEqual(got["layers"], 32)

    def test_bad_magic_is_an_error_not_an_exception(self):
        p = os.path.join(self.dir, "x.gguf")
        with open(p, "wb") as fh:
            fh.write(b"XXXX" + b"\0" * 64)
        self.assertIn("error", lmm.read_gguf(p))

    def test_truncated_file_is_an_error(self):
        p = os.path.join(self.dir, "t.gguf")
        with open(p, "wb") as fh:
            fh.write(lmm.GGUF_MAGIC + struct.pack("<I", 3)[:2])
        self.assertIn("error", lmm.read_gguf(p))

    def test_missing_file_is_an_error(self):
        self.assertIn("error", lmm.read_gguf(os.path.join(self.dir, "nope.gguf")))

    def test_gguf_v1_is_rejected_with_a_clear_reason(self):
        got = lmm.read_gguf(self.path(version=1))
        self.assertIn("error", got)
        self.assertIn("v1", got["error"])

    def test_missing_geometry_is_an_error(self):
        got = lmm.read_gguf(self.path(drop_kv=("layers",)))
        self.assertIn("error", got)

    def test_looks_like_gguf_does_not_catch_ollama_tags(self):
        self.assertFalse(lmm.looks_like_gguf("llama3.1:8b"))
        self.assertFalse(lmm.looks_like_gguf("qwen2.5-coder:7b"))
        self.assertTrue(lmm.looks_like_gguf("model.gguf"))
        self.assertTrue(lmm.looks_like_gguf("/models/x.GGUF"))

    def test_quant_label_from_bpw(self):
        self.assertEqual(lmm.quant_label_from_bpw(4.85), "~q4_k_m")
        self.assertEqual(lmm.quant_label_from_bpw(16.0), "~f16")
        self.assertEqual(lmm.quant_label_from_bpw(0), "")


class TestSSEFraming(unittest.TestCase):
    """SSE events are terminated by a BLANK LINE. Get that wrong and every
    client concatenates the whole stream into one event."""

    def test_frame_ends_with_a_blank_line(self):
        f = lmm.sse_frame({"a": 1})
        self.assertTrue(f.startswith(b"data: "))
        self.assertTrue(f.endswith(b"\n\n"))

    def test_frame_payload_is_single_line_json(self):
        f = lmm.sse_frame({"a": "x\ny"})
        body = f[len(b"data: "):-2]
        self.assertNotIn(b"\n", body)          # newline must be escaped in JSON
        self.assertEqual(json.loads(body.decode())["a"], "x\ny")

    def test_relay_restores_the_terminator(self):
        # http_stream_sse consumes the blank separator while parsing, so a
        # relayed line arrives with a single newline and must be re-terminated.
        self.assertEqual(lmm.sse_relay(b'data: {"a":1}\n'), b'data: {"a":1}\n\n')
        self.assertEqual(lmm.sse_relay(b'data: {"a":1}\r\n'), b'data: {"a":1}\n\n')
        self.assertEqual(lmm.sse_relay(b'data: {"a":1}'), b'data: {"a":1}\n\n')

    def test_relay_does_not_double_terminate(self):
        self.assertEqual(lmm.sse_relay(b'data: {"a":1}\n\n'), b'data: {"a":1}\n\n')

    def test_done_sentinel(self):
        self.assertEqual(lmm.SSE_DONE, b"data: [DONE]\n\n")

    def test_chunk_text_extracts_the_delta(self):
        self.assertEqual(lmm.chunk_text(
            {"choices": [{"delta": {"content": "hi"}}]}), "hi")

    def test_chunk_text_tolerates_every_empty_shape(self):
        for bad in ({}, {"choices": []}, {"choices": [{}]},
                    {"choices": [{"delta": {}}]},
                    {"choices": [{"delta": {"role": "assistant"}}]},
                    {"choices": [{"delta": {"content": None}}]}):
            self.assertEqual(lmm.chunk_text(bad), "")


class TestSynthStream(unittest.TestCase):
    """A cached or cascaded answer still has to reach the client as a stream."""

    def frames(self, text, **kw):
        return lmm.synth_stream(text, "m", **kw)

    def parsed(self, frames):
        out = []
        for f in frames:
            body = f[len(b"data: "):].strip()
            if body != b"[DONE]":
                out.append(json.loads(body.decode()))
        return out

    def test_ends_with_done(self):
        self.assertEqual(self.frames("hello")[-1], lmm.SSE_DONE)

    def test_first_chunk_carries_the_role(self):
        first = self.parsed(self.frames("hello"))[0]
        self.assertEqual(first["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual(first["object"], "chat.completion.chunk")

    def test_content_reassembles_exactly(self):
        text = "The answer is 4. " * 20
        got = "".join(lmm.chunk_text(c) for c in self.parsed(self.frames(text)))
        self.assertEqual(got, text)

    def test_last_content_chunk_has_finish_reason(self):
        chunks = self.parsed(self.frames("hello"))
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_usage_chunk_when_usage_is_known(self):
        chunks = self.parsed(self.frames("hello", usage={"completion_tokens": 5}))
        self.assertEqual(chunks[-1]["usage"]["completion_tokens"], 5)
        self.assertEqual(chunks[-1]["choices"], [])   # usage chunk carries none

    def test_empty_text_still_produces_a_valid_stream(self):
        frames = self.frames("")
        self.assertEqual(frames[-1], lmm.SSE_DONE)
        self.assertTrue(all(f.endswith(b"\n\n") for f in frames))


class TestEstimateTokens(unittest.TestCase):
    def test_ascii_is_about_four_chars_per_token(self):
        self.assertAlmostEqual(lmm.estimate_tokens("a" * 400), 100, delta=2)

    def test_cjk_counts_closer_to_one_per_char(self):
        self.assertGreater(lmm.estimate_tokens("日本語" * 10), 25)

    def test_empty_is_zero_and_short_is_at_least_one(self):
        self.assertEqual(lmm.estimate_tokens(""), 0)
        self.assertEqual(lmm.estimate_tokens(None), 0)
        self.assertGreaterEqual(lmm.estimate_tokens("a"), 1)


class TestHubStream(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self._real = lmm.call_provider_stream
        self._real_call = lmm.call_provider

    def tearDown(self):
        lmm.call_provider_stream = self._real
        lmm.call_provider = self._real_call

    def fake_stream(self, words, usage=True, fail_at=None):
        def _s(prov, prompt, temperature=0.7, extra=None):
            self.calls.append(prov.get("model"))
            yield (b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
                   {"choices": [{"delta": {"role": "assistant"}}]})
            for i, w in enumerate(words):
                if fail_at is not None and i == fail_at:
                    yield None, {"error": "upstream exploded"}
                    return
                obj = {"choices": [{"delta": {"content": w}}]}
                yield ("data: " + json.dumps(obj) + "\n").encode(), obj
            if usage:
                u = {"choices": [], "usage": {"prompt_tokens": 10,
                                              "completion_tokens": len(words)}}
                yield ("data: " + json.dumps(u) + "\n").encode(), u
            yield None, None
        lmm.call_provider_stream = _s

    def targets(self):
        return [("p", {"kind": "remote", "model": "p",
                       "price": {"in": 1.0, "out": 2.0}})]

    def collect(self, gen):
        return b"".join(gen)

    def test_passthrough_relays_terminated_frames_and_ends_with_done(self):
        self.fake_stream(["a", "b"])
        with temp_state():
            out = self.collect(lmm.hub_stream({}, "hi", self.targets(),
                                              {"cache": False}))
        self.assertTrue(out.endswith(lmm.SSE_DONE))
        for block in out.split(b"\n\n")[:-1]:
            self.assertTrue(block.startswith(b"data: "))

    def test_passthrough_meters_with_real_usage_and_ttft(self):
        self.fake_stream(["a", "b", "c"])
        with temp_state():
            self.collect(lmm.hub_stream({}, "hi", self.targets(), {"cache": False}))
            events = lmm.read_usage()
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]["stream"])
        self.assertFalse(events[0]["estimated"])
        self.assertFalse(events[0]["partial"])
        self.assertEqual(events[0]["out"], 3)
        self.assertIsNotNone(events[0]["ttft_ms"])

    def test_missing_usage_chunk_is_estimated_and_flagged(self):
        self.fake_stream(["hello "] * 4, usage=False)
        with temp_state():
            self.collect(lmm.hub_stream({}, "hi", self.targets(), {"cache": False}))
            events = lmm.read_usage()
        self.assertTrue(events[0]["estimated"])
        self.assertGreater(events[0]["out"], 0)

    def test_client_hangup_still_meters_as_partial(self):
        # Tokens were generated and billed upstream even though nobody read
        # them; recording nothing would understate real spend.
        self.fake_stream(["a", "b", "c", "d", "e"])
        with temp_state():
            gen = lmm.hub_stream({}, "hi", self.targets(), {"cache": False})
            next(gen)
            next(gen)
            gen.close()                      # client hangs up
            events = lmm.read_usage()
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]["partial"])
        self.assertFalse(events[0]["accepted"])

    def test_partial_stream_is_never_cached(self):
        self.fake_stream(["a", "b", "c", "d", "e"])
        with temp_state():
            gen = lmm.hub_stream({}, "hi", self.targets(), {"cache": True})
            next(gen)
            next(gen)
            gen.close()
            entry, how, _, _cand = lmm.cache_lookup({}, lmm.as_messages("hi"), "p")
        self.assertIsNone(entry)

    def test_cache_hit_is_replayed_as_a_synthetic_stream(self):
        self.fake_stream(["The answer is 4."])
        opts = {"cache": True}
        with temp_state():
            self.collect(lmm.hub_stream({}, "hi", self.targets(), opts))
            self.assertEqual(len(self.calls), 1)
            out = self.collect(lmm.hub_stream({}, "hi", self.targets(), opts))
            events = lmm.read_usage()
        self.assertEqual(len(self.calls), 1)          # no second upstream call
        self.assertTrue(out.endswith(lmm.SSE_DONE))
        self.assertIn(b"The answer is 4.", out)
        self.assertEqual(events[-1]["provider"], "cache")

    def test_usage_chunk_is_withheld_unless_the_client_asked(self):
        # lmm always requests include_usage upstream so it can meter, but must
        # not hand the client a chunk it never asked for.
        self.fake_stream(["a", "b"])
        with temp_state():
            out = self.collect(lmm.hub_stream({}, "hi", self.targets(),
                                              {"cache": False}))
            events = lmm.read_usage()
        self.assertNotIn(b'"usage"', out)
        self.assertEqual(events[0]["out"], 2)      # still metered exactly

    def test_usage_chunk_is_relayed_when_the_client_asked(self):
        self.fake_stream(["a", "b"])
        with temp_state():
            out = self.collect(lmm.hub_stream({}, "hi", self.targets(),
                                              {"cache": False, "client_usage": True}))
        self.assertIn(b'"usage"', out)

    def test_upstream_failure_before_any_byte_fails_over(self):
        def _s(prov, prompt, temperature=0.7, extra=None):
            self.calls.append(prov.get("model"))
            if prov.get("model") == "bad":
                yield None, {"error": "401"}
                return
            obj = {"choices": [{"delta": {"content": "ok"}}]}
            yield ("data: " + json.dumps(obj) + "\n").encode(), obj
            yield None, None
        lmm.call_provider_stream = _s
        targets = [("bad", {"kind": "remote", "model": "bad"}),
                   ("good", {"kind": "remote", "model": "good"})]
        with temp_state():
            out = self.collect(lmm.hub_stream({}, "hi", targets, {"cache": False}))
        self.assertEqual(self.calls, ["bad", "good"])
        self.assertIn(b"ok", out)

    def test_mid_stream_failure_reports_instead_of_failing_over(self):
        # Bytes are already on the wire; a retry would duplicate output.
        self.fake_stream(["a", "b", "c"], fail_at=1)
        targets = self.targets() + [("p2", {"kind": "remote", "model": "p2"})]
        with temp_state():
            out = self.collect(lmm.hub_stream({}, "hi", targets, {"cache": False}))
        self.assertIn(b"mid-stream", out)
        self.assertTrue(out.endswith(lmm.SSE_DONE))
        self.assertEqual(self.calls, ["p"])           # no retry on p2

    def test_cascade_buffers_then_replays(self):
        # The verifier needs the whole answer, so streaming is synthesised.
        def _call(prov, prompt, temperature=0.7, extra=None):
            self.calls.append(prov.get("model"))
            return {"choices": [{"message": {"content": "The answer is 4, indeed."}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 5}}
        lmm.call_provider = _call
        lmm.call_provider_stream = None       # must not be used at all
        with temp_state():
            out = self.collect(lmm.hub_stream({}, "hi", self.targets(),
                                              {"cascade": True, "cache": False}))
        self.assertIn(b"The answer is 4", out)
        self.assertTrue(out.endswith(lmm.SSE_DONE))

    def test_no_targets_yields_an_error_frame_not_a_crash(self):
        with temp_state():
            out = self.collect(lmm.hub_stream({}, "hi", [], {"cache": False}))
        self.assertIn(b"no provider available", out)
        self.assertTrue(out.endswith(lmm.SSE_DONE))


class TestStreamingRegressions(unittest.TestCase):
    """Defects found by probing the first streaming implementation. Each of
    these reproduced before its fix."""

    def setUp(self):
        self.calls = []
        self.lookups = []
        self.orders = []
        self._stream = lmm.call_provider_stream
        self._call = lmm.call_provider
        self._lookup = lmm.cache_lookup
        self._order = lmm.order_targets

    def tearDown(self):
        lmm.call_provider_stream = self._stream
        lmm.call_provider = self._call
        lmm.cache_lookup = self._lookup
        lmm.order_targets = self._order

    def targets(self):
        return [("p", {"kind": "remote", "model": "p",
                       "price": {"in": 1.0, "out": 2.0}}),
                ("q", {"kind": "remote", "model": "q",
                       "price": {"in": 2.0, "out": 4.0}})]

    def count_calls(self):
        def _lookup(cfg, messages, model, extra=None):
            self.lookups.append(model)
            return self._lookup(cfg, messages, model, extra)

        def _order(cfg, prompt, targets):
            self.orders.append(1)
            return self._order(cfg, prompt, targets)
        lmm.cache_lookup = _lookup
        lmm.order_targets = _order

    def cascade_fake(self):
        def _call(prov, prompt, temperature=0.7, extra=None):
            self.calls.append(prov.get("model"))
            return {"choices": [{"message": {"content": "The answer is 4, truly."}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 100}}
        lmm.call_provider = _call

    # --- defect 1 -------------------------------------------------------
    def test_provider_ignoring_stream_true_is_an_error_not_silence(self):
        # A provider that answers a streamed request with a plain JSON body
        # produced zero frames, zero metering and no failover — just silence.
        import io

        class FakeResp(io.BytesIO):
            def close(self):
                pass
        real_open = lmm.__dict__.get("_urlopen_patch")
        import urllib.request
        saved = urllib.request.urlopen
        urllib.request.urlopen = lambda *a, **k: FakeResp(
            b'{"choices":[{"message":{"content":"not a stream"}}]}')
        try:
            frames = list(lmm.http_stream_sse("http://x/v1/chat/completions",
                                              {}, ""))
        finally:
            urllib.request.urlopen = saved
        self.assertEqual(len(frames), 1)
        self.assertIsNotNone(frames[0][1])
        self.assertIn("no SSE frames", frames[0][1]["error"])

    def test_a_provider_ignoring_stream_now_fails_over(self):
        def _s(prov, prompt, temperature=0.7, extra=None):
            self.calls.append(prov.get("model"))
            if prov.get("model") == "p":
                yield None, {"error": "upstream returned no SSE frames"}
                return
            obj = {"choices": [{"delta": {"content": "ok"}}]}
            yield ("data: " + json.dumps(obj) + "\n").encode(), obj
            yield None, None
        lmm.call_provider_stream = _s
        with temp_state():
            out = b"".join(lmm.hub_stream({}, "hi", self.targets(),
                                          {"cache": False}))
        self.assertEqual(self.calls, ["p", "q"])
        self.assertIn(b"ok", out)

    # --- defect 2 -------------------------------------------------------
    def test_buffered_cascade_is_metered_as_a_stream(self):
        self.cascade_fake()
        lmm.call_provider_stream = None
        with temp_state():
            b"".join(lmm.hub_stream({}, "hi", self.targets(),
                                    {"cascade": True, "cache": False}))
            events = lmm.read_usage()
        self.assertTrue(events[0]["stream"])
        self.assertTrue(events[0]["buffered"])
        self.assertIsNotNone(events[0]["ttft_ms"])

    def test_non_streamed_calls_are_still_marked_not_streamed(self):
        self.cascade_fake()
        with temp_state():
            lmm.hub_complete({}, "hi", self.targets(),
                             {"cascade": False, "cache": False})
            events = lmm.read_usage()
        self.assertFalse(events[0]["stream"])
        self.assertFalse(events[0]["buffered"])

    # --- defects 3 and 4 ------------------------------------------------
    def test_streamed_cascade_looks_up_the_cache_once(self):
        # Two lookups meant two embeddings and two near-miss log rows, which
        # doubled the very statistic `lmm cache` reports for tuning.
        self.cascade_fake()
        lmm.call_provider_stream = None
        self.count_calls()
        with temp_state():
            b"".join(lmm.hub_stream({}, "hi", self.targets(),
                                    {"cascade": True, "cache": True}))
        self.assertEqual(len(self.lookups), 1)

    def test_streamed_cascade_orders_targets_once(self):
        self.cascade_fake()
        lmm.call_provider_stream = None
        self.count_calls()
        with temp_state():
            b"".join(lmm.hub_stream({}, "hi", self.targets(),
                                    {"cascade": True, "cache": False}))
        self.assertEqual(len(self.orders), 1)

    def test_streamed_cascade_still_caches_with_the_real_spend(self):
        # hub_stream now owns the store; if it recorded $0 the next hit would
        # report saving nothing.
        self.cascade_fake()
        lmm.call_provider_stream = None
        with temp_state():
            b"".join(lmm.hub_stream({}, "hi", self.targets(),
                                    {"cascade": True, "cache": True}))
            entry, how, _, _cand = lmm.cache_lookup({}, lmm.as_messages("hi"), "p")
            self.assertEqual(how, "exact")
            self.assertGreater(entry["usd"], 0.0)
            out = b"".join(lmm.hub_stream({}, "hi", self.targets(),
                                          {"cascade": True, "cache": True}))
            events = lmm.read_usage()
        self.assertEqual(len(self.calls), 1)          # served from cache
        self.assertIn(b"The answer is 4", out)
        self.assertGreater(events[-1]["saved_usd"], 0.0)

    # --- defect 5 -------------------------------------------------------
    def test_source_is_not_downgraded_to_ask_by_the_delegate(self):
        self.cascade_fake()
        lmm.call_provider_stream = None
        with temp_state():
            b"".join(lmm.hub_stream({}, "hi", self.targets(),
                                    {"cascade": True, "cache": False,
                                     "source": "hub"}))
            events = lmm.read_usage()
        self.assertEqual(events[0]["source"], "hub")

    # --- defect 6 -------------------------------------------------------
    def test_cost_report_separates_estimates_from_measurements(self):
        pricing = lmm.merged_pricing({})
        with temp_state():
            lmm.log_usage({"provider": "p", "kind": "remote", "in": 100,
                           "out": 100, "usd": 1.0, "estimated": True,
                           "stream": True, "ttft_ms": 120})
            lmm.log_usage({"provider": "p", "kind": "remote", "in": 100,
                           "out": 100, "usd": 2.0, "stream": True,
                           "ttft_ms": 80})
            block = "\n".join(lmm.hub_cost_block({}, pricing))
        self.assertIn("HUB MEASURED TOTAL  $3.0000", block)
        self.assertIn("ESTIMATED", block)
        self.assertIn("$1.0000", block)
        self.assertIn("STREAM TTFT", block)

    def test_cost_report_names_partial_spend(self):
        pricing = lmm.merged_pricing({})
        with temp_state():
            lmm.log_usage({"provider": "p", "kind": "remote", "in": 10,
                           "out": 10, "usd": 0.5, "partial": True,
                           "stream": True})
            block = "\n".join(lmm.hub_cost_block({}, pricing))
        self.assertIn("PARTIAL", block)
        self.assertIn("abandoned", block)

    def test_buffered_cascade_ttft_is_excluded_from_the_stream_ttft_series(self):
        pricing = lmm.merged_pricing({})
        with temp_state():
            lmm.log_usage({"provider": "p", "kind": "remote", "in": 1, "out": 1,
                           "usd": 0.1, "stream": True, "buffered": True,
                           "ttft_ms": 9999})
            block = "\n".join(lmm.hub_cost_block({}, pricing))
        self.assertNotIn("STREAM TTFT", block)

    def test_near_miss_rows_are_not_counted_as_calls(self):
        pricing = lmm.merged_pricing({})
        with temp_state():
            lmm.log_usage({"provider": "cache", "kind": "local", "in": 0,
                           "out": 0, "usd": 0.0, "cache": "near-miss",
                           "similarity": 0.9})
            block = "\n".join(lmm.hub_cost_block({}, pricing))
        # A near-miss is a cache probe, not a billable call: it must not create
        # a phantom provider row, and an all-near-miss log is not "usage".
        self.assertNotIn("HUB MEASURED TOTAL", block)
        self.assertIn("No metered hub calls yet", block)

    def test_percentile(self):
        xs = list(range(1, 101))
        self.assertEqual(lmm.percentile(xs, 50), 50)
        self.assertEqual(lmm.percentile(xs, 90), 90)
        self.assertEqual(lmm.percentile(xs, 100), 100)
        self.assertEqual(lmm.percentile([], 90), 0.0)
        self.assertEqual(lmm.percentile([7], 90), 7)


class TestBenchMetrics(unittest.TestCase):
    """TTFT / TPOT / e2e per the standard serving decomposition."""

    def tearDown(self):
        if hasattr(self, "_real"):
            lmm.call_provider_stream = self._real

    def fake(self, n_words, delay=0.01, usage=True):
        self._real = lmm.call_provider_stream

        def _s(prov, prompt, temperature=0.7, extra=None):
            time.sleep(delay)                     # prefill
            for i in range(n_words):
                if i:
                    time.sleep(delay)             # decode, per token
                yield None, {"choices": [{"delta": {"content": "x"}}]}
            if usage:
                yield None, {"choices": [], "usage": {"prompt_tokens": 3,
                                                      "completion_tokens": n_words}}
            yield None, None
        lmm.call_provider_stream = _s

    def test_identity_e2e_equals_ttft_plus_tpot_times_tokens_minus_one(self):
        self.fake(5)
        r = lmm.bench_once({"model": "m", "base_url": "x"}, "hi")
        self.assertNotIn("error", r)
        self.assertAlmostEqual(
            r["e2e_ms"], r["ttft_ms"] + r["tpot_ms"] * (r["out_tokens"] - 1),
            places=6)

    def test_ttft_reflects_prefill_delay(self):
        self.fake(4, delay=0.05)
        r = lmm.bench_once({"model": "m", "base_url": "x"}, "hi")
        self.assertGreater(r["ttft_ms"], 40)

    def test_throughput_is_tokens_over_e2e(self):
        self.fake(6)
        r = lmm.bench_once({"model": "m", "base_url": "x"}, "hi")
        self.assertAlmostEqual(r["tok_per_s"], r["out_tokens"] / (r["e2e_ms"] / 1000.0),
                               places=6)

    def test_single_token_response_has_no_tpot(self):
        self.fake(1)
        r = lmm.bench_once({"model": "m", "base_url": "x"}, "hi")
        self.assertEqual(r["tpot_ms"], 0.0)

    def test_estimated_flag_when_provider_omits_usage(self):
        self.fake(4, usage=False)
        r = lmm.bench_once({"model": "m", "base_url": "x"}, "hi")
        self.assertTrue(r["estimated"])

    def test_error_is_surfaced(self):
        self._real = lmm.call_provider_stream

        def _s(prov, prompt, temperature=0.7, extra=None):
            yield None, {"error": "boom"}
        lmm.call_provider_stream = _s
        self.assertIn("error", lmm.bench_once({"model": "m"}, "hi"))

    def test_median(self):
        self.assertEqual(lmm.median([3, 1, 2]), 2)
        self.assertEqual(lmm.median([4, 1, 2, 3]), 2.5)
        self.assertEqual(lmm.median([]), 0.0)


class TestDiscovery(unittest.TestCase):
    """The registry lists 16 runtimes; discovery has to actually cover them."""

    def setUp(self):
        self._probe = lmm.probe_port
        self._models = lmm.probe_models
        self._procs = lmm.proc_count
        lmm.probe_models = lambda base, timeout=1.0: []
        lmm.proc_count = lambda names: 0

    def tearDown(self):
        lmm.probe_port = self._probe
        lmm.probe_models = self._models
        lmm.proc_count = self._procs

    def test_every_registry_runtime_has_a_label(self):
        for name, spec in lmm.RUNTIME_REGISTRY.items():
            self.assertTrue(spec.get("label"), name)

    def test_every_registry_runtime_is_detectable(self):
        lmm.probe_port = lambda port, host="127.0.0.1", timeout=0.25: False
        for name in lmm.RUNTIME_REGISTRY:
            got = lmm.detect_runtime(name)
            self.assertIn("running", got)
            self.assertEqual(got["key"], name)

    def test_discover_covers_the_whole_registry(self):
        lmm.probe_port = lambda port, host="127.0.0.1", timeout=0.25: False
        keys = {it.get("key") for it in lmm.discover({}, with_models=False)}
        self.assertTrue(set(lmm.RUNTIME_REGISTRY).issubset(keys))

    def test_an_open_port_marks_a_runtime_as_serving(self):
        lmm.probe_port = lambda port, host="127.0.0.1", timeout=0.25: port == 1337
        got = lmm.detect_runtime("jan")
        self.assertTrue(got["serving"])
        self.assertTrue(got["running"])
        self.assertIn("1337", got["endpoint"])
        self.assertNotIn("closed", got["endpoint"])

    def test_a_closed_port_is_labelled_closed(self):
        lmm.probe_port = lambda port, host="127.0.0.1", timeout=0.25: False
        self.assertIn("closed", lmm.detect_runtime("jan")["endpoint"])

    def test_shared_default_port_needs_a_matching_process(self):
        # llama.cpp's server and Open WebUI both default to 8080, so an open
        # socket alone cannot say which one is there.
        lmm.probe_port = lambda port, host="127.0.0.1", timeout=0.25: port == 8080
        self.assertFalse(lmm.detect_runtime("llamacpp")["serving"])
        lmm.proc_count = lambda names: 1
        self.assertTrue(lmm.detect_runtime("llamacpp")["serving"])

    def test_unambiguous_port_does_not_need_a_process(self):
        lmm.probe_port = lambda port, host="127.0.0.1", timeout=0.25: port == 11434
        self.assertTrue(lmm.detect_runtime("ollama")["serving"])

    def test_desktop_only_apps_have_no_invented_port(self):
        # Guessing an endpoint for an app with no documented local API would
        # produce confident wrong output.
        for name in ("chatgpt", "cursor", "perplexity", "devin", "chatbox"):
            self.assertIsNone(lmm.RUNTIME_REGISTRY[name]["port"], name)

    def test_documented_ports_are_the_published_defaults(self):
        expect = {"ollama": 11434, "lmstudio": 1234, "jan": 1337, "gpt4all": 4891,
                  "anythingllm": 3001, "koboldcpp": 5001, "vllm": 8000,
                  "llamacpp": 8080, "openwebui": 8080}
        for name, port in expect.items():
            self.assertEqual(lmm.RUNTIME_REGISTRY[name]["port"], port, name)

    def test_probe_port_on_a_closed_port_is_false_and_fast(self):
        # port 1 is reserved and never listening
        self.assertFalse(self._probe(1, timeout=0.2))
        self.assertFalse(self._probe(None))

    def test_probe_models_tolerates_a_dead_endpoint(self):
        self.assertEqual(self._models("http://127.0.0.1:1/v1", timeout=0.2), [])


class TestMeasuredTokens(unittest.TestCase):
    """`walk` yields one usage block per record; descending past one
    double-counts every total in `lmm cost`."""

    def walk_totals(self, record):
        # exercise the nested walker through the public path
        import tempfile as tf
        d = tf.mkdtemp(prefix="lmm-claude-")
        proj = os.path.join(d, "projects", "p")
        os.makedirs(proj)
        path = os.path.join(proj, "s.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        saved = lmm.CLAUDE_PROJECTS
        lmm.CLAUDE_PROJECTS = os.path.join(d, "projects")
        try:
            return lmm.measured_tokens()
        finally:
            lmm.CLAUDE_PROJECTS = saved
            os.remove(path)
            os.rmdir(proj)
            os.rmdir(os.path.join(d, "projects"))
            os.rmdir(d)

    def test_nested_usage_is_counted_once(self):
        rec = {"type": "assistant", "model": "claude-sonnet-x",
               "message": {"model": "claude-sonnet-x",
                           "usage": {"input_tokens": 10, "output_tokens": 20}}}
        got = self.walk_totals(rec)
        fam = got["by_family"]["sonnet"]
        self.assertEqual(fam["in"], 10)
        self.assertEqual(fam["out"], 20)

    def test_duplicated_usage_at_two_levels_is_not_doubled(self):
        u = {"input_tokens": 10, "output_tokens": 20}
        rec = {"model": "claude-opus-x", "usage": dict(u),
               "message": {"usage": dict(u)}}
        got = self.walk_totals(rec)
        fam = got["by_family"]["opus"]
        self.assertEqual(fam["in"], 10)
        self.assertEqual(fam["out"], 20)


class TestHubSecurity(unittest.TestCase):
    """The hub holds the user's API keys; reachability is the entire boundary."""

    def test_loopback_needs_nothing(self):
        for host in ("127.0.0.1", "localhost", "::1", "127.1.2.3"):
            allowed, lines = lmm.hub_bind_check(host, lmm.merged_hub({}))
            self.assertTrue(allowed, host)
            self.assertEqual(lines, [])

    def test_an_empty_host_is_not_loopback_it_is_every_interface(self):
        """Socratic consequence of "reachability is the entire security
        boundary": whatever hub_bind_check calls loopback must actually be
        unreachable from the network. Measured: HTTPServer(("", 0)) listens
        on 0.0.0.0 and accepts a connection on the LAN address. "" was in
        LOOPBACK_HOSTS, so `--host ""` bound everywhere with no token and no
        warning."""
        import socket
        import http.server
        srv = http.server.HTTPServer(("", 0), http.server.BaseHTTPRequestHandler)
        try:
            self.assertEqual(srv.server_address[0], "0.0.0.0")   # the fact
        finally:
            srv.server_close()
        self.assertFalse(lmm.is_loopback(""))                    # the rule
        allowed, lines = lmm.hub_bind_check("", lmm.merged_hub({}),
                                            lambda: "SUGGESTED")
        self.assertFalse(allowed)
        self.assertIn("refusing", "\n".join(lines))

    def test_non_loopback_without_token_is_refused(self):
        allowed, lines = lmm.hub_bind_check("0.0.0.0", lmm.merged_hub({}),
                                            lambda: "SUGGESTED")
        self.assertFalse(allowed)
        text = "\n".join(lines)
        self.assertIn("refusing", text)
        self.assertIn("SUGGESTED", text)      # tells the user exactly what to do

    def test_non_loopback_with_token_is_allowed(self):
        hub = lmm.merged_hub({"hub": {"token": "t"}})
        allowed, lines = lmm.hub_bind_check("0.0.0.0", hub)
        self.assertTrue(allowed)
        self.assertIn("token required", "\n".join(lines))

    def test_allow_remote_is_allowed_but_warned(self):
        hub = lmm.merged_hub({"hub": {"allow_remote": True}})
        allowed, lines = lmm.hub_bind_check("0.0.0.0", hub)
        self.assertTrue(allowed)
        self.assertIn("WARNING", "\n".join(lines))

    def test_loopback_with_token_reminds_clients(self):
        hub = lmm.merged_hub({"hub": {"token": "t"}})
        allowed, lines = lmm.hub_bind_check("127.0.0.1", hub)
        self.assertTrue(allowed)
        self.assertIn("Authorization: Bearer", "\n".join(lines))


class TestUntrustedConfig(unittest.TestCase):
    """A config in the working directory is whatever repo you cd'd into, so it
    must not be able to run shell commands (`extra_runtimes[].models_cmd`)."""

    def setUp(self):
        self._path, self._trusted = lmm.CONFIG_PATH, lmm.CONFIG_TRUSTED
        self._ran = []
        self._run = lmm.run
        lmm.run = lambda cmd: self._ran.append(cmd)

    def tearDown(self):
        lmm.CONFIG_PATH, lmm.CONFIG_TRUSTED = self._path, self._trusted
        lmm.run = self._run

    def cfg(self):
        return {"extra_runtimes": [{"name": "X", "procs": [],
                                    "installed_paths": [],
                                    "models_cmd": "echo pwned"}]}

    def test_untrusted_config_does_not_run_models_cmd(self):
        import io
        import contextlib
        lmm.CONFIG_PATH, lmm.CONFIG_TRUSTED = "lmm.config.json", False
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            lmm.detect_extra(self.cfg())
        self.assertNotIn("echo pwned", self._ran)    # models_cmd never ran
        self.assertIn("ignoring models_cmd", err.getvalue())   # and it says why

    def test_trusted_config_runs_models_cmd(self):
        lmm.CONFIG_PATH, lmm.CONFIG_TRUSTED = "/home/u/.lmm/config.json", True
        lmm.detect_extra(self.cfg())
        self.assertIn("echo pwned", self._ran)


class TestBackoff(unittest.TestCase):
    """Full-jitter backoff (AWS, Marc Brooker)."""

    def test_delay_is_within_the_capped_exponential(self):
        # rand injected at the extremes bounds the range.
        for attempt in range(6):
            hi = lmm.backoff_delay(attempt, 250, 8000, rand=1.0)
            lo = lmm.backoff_delay(attempt, 250, 8000, rand=0.0)
            self.assertEqual(lo, 0.0)                      # full jitter reaches 0
            self.assertLessEqual(hi, 8000 / 1000.0)        # never exceeds the cap
            self.assertAlmostEqual(hi, min(8000, 250 * 2 ** attempt) / 1000.0)

    def test_grows_until_the_cap(self):
        a = lmm.backoff_delay(0, 250, 8000, rand=1.0)
        b = lmm.backoff_delay(1, 250, 8000, rand=1.0)
        self.assertGreater(b, a)


class TestRetryAfter(unittest.TestCase):
    def test_integer_seconds(self):
        self.assertEqual(lmm.parse_retry_after("5"), 5.0)
        self.assertEqual(lmm.parse_retry_after("0"), 0.0)

    def test_http_date_in_the_future_is_positive(self):
        import email.utils, datetime as dt
        future = email.utils.format_datetime(
            dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=30))
        self.assertGreater(lmm.parse_retry_after(future), 20)

    def test_garbage_and_empty_are_none(self):
        self.assertIsNone(lmm.parse_retry_after("soon"))
        self.assertIsNone(lmm.parse_retry_after(""))
        self.assertIsNone(lmm.parse_retry_after(None))


class TestErrorClassification(unittest.TestCase):
    def make(self, code, retry_after=None):
        import urllib.error, io
        hdrs = {}
        if retry_after is not None:
            hdrs["Retry-After"] = retry_after
        return urllib.error.HTTPError("http://x", code, "msg",
                                      hdrs, io.BytesIO(b""))

    def test_429_and_5xx_are_retriable(self):
        for code in (429, 500, 502, 503):
            self.assertTrue(lmm.classify_http_error(self.make(code))["retriable"], code)

    def test_4xx_client_errors_are_not_retriable(self):
        for code in (400, 401, 403, 404, 422):
            self.assertFalse(lmm.classify_http_error(self.make(code))["retriable"], code)

    def test_connection_errors_are_retriable(self):
        got = lmm.classify_http_error(OSError("connection refused"))
        self.assertTrue(got["retriable"])
        self.assertIsNone(got["status"])

    def test_retry_after_is_parsed_from_the_header(self):
        got = lmm.classify_http_error(self.make(429, "7"))
        self.assertEqual(got["retry_after"], 7.0)


class TestCallWithRetry(unittest.TestCase):
    def setUp(self):
        self._call = lmm.call_provider
        self.slept = []

    def tearDown(self):
        lmm.call_provider = self._call

    def sleep(self, d):
        self.slept.append(d)

    def sequence(self, results):
        it = iter(results)
        self.n = 0

        def _call(prov, prompt, temperature=0.7, extra=None):
            self.n += 1
            return next(it)
        lmm.call_provider = _call

    def test_transient_error_then_success(self):
        self.sequence([{"error": "503", "retriable": True},
                       {"choices": [{"message": {"content": "ok"}}]}])
        res = lmm.call_with_retry({}, "hi",
                                  retry={"attempts": 3, "base_ms": 1, "cap_ms": 2},
                                  sleep=self.sleep)
        self.assertEqual(res["choices"][0]["message"]["content"], "ok")
        self.assertEqual(self.n, 2)
        self.assertEqual(len(self.slept), 1)

    def test_permanent_error_is_not_retried(self):
        self.sequence([{"error": "401", "retriable": False}])
        res = lmm.call_with_retry({}, "hi",
                                  retry={"attempts": 3, "base_ms": 1, "cap_ms": 2},
                                  sleep=self.sleep)
        self.assertIn("error", res)
        self.assertEqual(self.n, 1)                # tried once, gave up

    def test_bare_error_from_a_fake_is_not_retried(self):
        # No `retriable` key -> the pre-retry behaviour, so old fakes still work.
        self.sequence([{"error": "boom"}])
        lmm.call_with_retry({}, "hi",
                            retry={"attempts": 3, "base_ms": 1, "cap_ms": 2},
                            sleep=self.sleep)
        self.assertEqual(self.n, 1)

    def test_attempts_are_bounded(self):
        self.sequence([{"error": "503", "retriable": True}] * 9)
        lmm.call_with_retry({}, "hi",
                            retry={"attempts": 3, "base_ms": 1, "cap_ms": 2},
                            sleep=self.sleep)
        self.assertEqual(self.n, 3)                # never more than `attempts`

    def test_retry_after_overrides_backoff_and_is_capped(self):
        self.sequence([{"error": "429", "retriable": True, "retry_after": 999},
                       {"choices": [{"message": {"content": "ok"}}]}])
        lmm.call_with_retry({}, "hi",
                            retry={"attempts": 2, "base_ms": 1, "cap_ms": 3000},
                            sleep=self.sleep)
        self.assertEqual(self.slept, [3.0])        # capped at cap_ms, not 999s


class TestCircuitBreaker(unittest.TestCase):
    def test_opens_after_threshold_consecutive_failures(self):
        cb = lmm.CircuitBreaker(threshold=3, cooldown_s=30)
        self.assertTrue(cb.available("p", now=0))
        cb.record_failure("p", now=0)
        cb.record_failure("p", now=0)
        self.assertTrue(cb.available("p", now=0))   # 2 < 3, still closed
        cb.record_failure("p", now=0)
        self.assertFalse(cb.available("p", now=0))  # tripped

    def test_half_open_after_cooldown(self):
        cb = lmm.CircuitBreaker(threshold=1, cooldown_s=30)
        cb.record_failure("p", now=100)
        self.assertEqual(cb.state("p", now=110), "open")
        self.assertEqual(cb.state("p", now=131), "half-open")
        self.assertTrue(cb.available("p", now=131))

    def test_success_closes_the_circuit(self):
        cb = lmm.CircuitBreaker(threshold=1, cooldown_s=30)
        cb.record_failure("p", now=0)
        self.assertFalse(cb.available("p", now=0))
        cb.record_success("p")
        self.assertTrue(cb.available("p", now=0))
        self.assertEqual(cb.state("p", now=0), "closed")

    def test_success_resets_the_failure_count(self):
        cb = lmm.CircuitBreaker(threshold=3, cooldown_s=30)
        cb.record_failure("p", now=0)
        cb.record_failure("p", now=0)
        cb.record_success("p")                      # count back to 0
        cb.record_failure("p", now=0)
        cb.record_failure("p", now=0)
        self.assertTrue(cb.available("p", now=0))   # only 2 since the reset


class TestBreakerInHub(unittest.TestCase):
    def setUp(self):
        self._call = lmm.call_provider

    def tearDown(self):
        lmm.call_provider = self._call

    def targets(self):
        return [("dead", {"kind": "remote", "model": "d", "price": {"in": 1, "out": 2}}),
                ("good", {"kind": "remote", "model": "g", "price": {"in": 1, "out": 2}})]

    def test_open_provider_is_skipped(self):
        seen = []

        def _call(prov, prompt, temperature=0.7, extra=None):
            seen.append(prov.get("model"))
            return {"choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        lmm.call_provider = _call
        cb = lmm.CircuitBreaker(threshold=1, cooldown_s=999)
        cb.record_failure("dead", now=time.time())   # force it open
        with temp_state():
            lmm.hub_complete({"ask_order": ["dead", "good"]}, "hi", self.targets(),
                             {"cache": False, "breaker": cb})
        self.assertEqual(seen, ["g"])                # dead was skipped

    def test_a_provider_failure_is_recorded(self):
        def _call(prov, prompt, temperature=0.7, extra=None):
            if prov.get("model") == "d":
                return {"error": "503", "retriable": False}
            return {"choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        lmm.call_provider = _call
        cb = lmm.CircuitBreaker(threshold=1, cooldown_s=999)
        with temp_state():
            lmm.hub_complete({"ask_order": ["dead", "good"]}, "hi", self.targets(),
                             {"cache": False, "breaker": cb})
        self.assertFalse(cb.available("dead", now=time.time()))
        self.assertTrue(cb.available("good", now=time.time()))


class TestHubCostStats(unittest.TestCase):
    """The one aggregation the text report, dashboard and status all share."""

    def seed(self):
        lmm.log_usage({"provider": "p", "kind": "remote", "in": 100, "out": 200,
                       "usd": 1.0, "cache": "miss", "stream": True, "ttft_ms": 80})
        lmm.log_usage({"provider": "p", "kind": "remote", "in": 100, "out": 200,
                       "usd": 2.0, "cache": "miss", "estimated": True})
        lmm.log_usage({"provider": "loc", "kind": "local", "in": 50, "out": 50,
                       "usd": 0.0, "cache": "miss"})
        lmm.log_usage({"provider": "cache", "kind": "local", "in": 0, "out": 0,
                       "usd": 0.0, "cache": "exact", "saved_usd": 1.5})
        lmm.log_usage({"provider": "cache", "kind": "local", "in": 0, "out": 0,
                       "usd": 0.0, "cache": "near-miss", "similarity": 0.9})

    def test_aggregation(self):
        with temp_state():
            self.seed()
            st = lmm.hub_cost_stats()
        self.assertEqual(st["calls"], 3)             # near-miss/hit are not calls
        self.assertAlmostEqual(st["measured"], 3.0)
        self.assertAlmostEqual(st["est_usd"], 2.0)
        self.assertEqual(st["hits"]["exact"], 1)
        self.assertAlmostEqual(st["saved_cache"], 1.5)
        self.assertEqual(st["local_calls"], 1)
        self.assertEqual(st["ttfts"], [80.0])

    def test_empty_log_yields_zeroes(self):
        with temp_state():
            st = lmm.hub_cost_stats()
        self.assertEqual(st["calls"], 0)
        self.assertEqual(st["measured"], 0.0)

    def test_cost_summary_is_one_line_and_honest(self):
        # The GUI shows this verbatim; it must never be a paragraph, and it
        # must not surface any hypothetical figure.
        with temp_state():
            self.seed()
            line = lmm.cost_summary({})
        self.assertNotIn("\n", line)
        self.assertIn("hub $", line)
        self.assertIn("cache saved", line)
        self.assertNotIn("~$", line)                 # no illustrative estimates

    def test_cost_summary_with_no_data_is_a_short_hint(self):
        with temp_state():
            line = lmm.cost_summary({})
        self.assertNotIn("\n", line)
        self.assertIn("no measured spend", line)

    def test_dash_cards_render_from_the_same_stats(self):
        with temp_state():
            self.seed()
            cards = frontend.dash_cards({})
        labels = [c[0] for c in cards]
        self.assertIn("Hub spend", labels)
        self.assertIn("Cache savings", labels)
        self.assertIn("Stream TTFT", labels)
        self.assertIn("Not clean measurements", labels)
        for label, value, note in cards:
            self.assertIsInstance(value, str)
            self.assertIsInstance(note, str)

    def test_build_dash_produces_html_with_serving_column(self):
        saved = lmm.discover
        lmm.discover = lambda cfg, with_models=True: [
            {"name": "X", "key": "x", "type": "local", "paid": False,
             "running": True, "serving": True, "procs": 1,
             "models": ["m1"], "endpoint": "http://localhost:1/v1",
             "installed": True}]
        try:
            with temp_state():
                h = frontend.build_dash({})
        finally:
            lmm.discover = saved
        self.assertTrue(h.startswith("<!doctype"))
        self.assertIn("<th>Serving</th>", h)
        self.assertIn("m1", h)


class TestRouteUnified(unittest.TestCase):
    """`lmm route` must give ONE answer — the head of the order the ask engine
    actually uses — not an independent heuristic that can contradict it."""

    def setUp(self):
        self._ollama = lmm.local_ollama_provider
        lmm.local_ollama_provider = lambda: None     # no implicit Ollama

    def tearDown(self):
        lmm.local_ollama_provider = self._ollama

    def cfg(self):
        return {"ask_order": ["openai"],
                "providers": {"openai": {"api_key": "k", "model": "gpt-4o",
                                         "kind": "remote"}}}

    def test_recommendation_is_the_head_of_the_real_order(self):
        cfg = lmm.load_config() if False else self.cfg()
        rec = lmm.route_task(cfg, "summarize this document")
        targets = lmm.resolve_ask_targets(cfg, "summarize this document", None)
        head = lmm.order_targets(cfg, "summarize this document", targets)[0][0]
        self.assertTrue(rec.startswith(head), rec)   # cannot contradict

    def test_ask_order_is_named_as_the_reason(self):
        rec = lmm.route_task(self.cfg(), "hello")
        self.assertIn("ask_order", rec)

    def test_no_providers_says_so(self):
        rec = lmm.route_task({}, "hello")
        self.assertIn("no provider available", rec)

    def test_private_with_no_local_says_start_one(self):
        rec = lmm.route_task({}, "summarize this secret memo")
        self.assertIn("private", rec)
        self.assertIn("lmm serve", rec)


class TestPruneSeen(unittest.TestCase):
    """Windows recycles HWNDs; an ever-growing 'seen' set eventually swallows
    brand-new windows that reuse an old handle."""

    def test_absent_handles_are_forgotten_after_the_grace(self):
        seen = {100: 1, 200: 1}
        frontend.prune_seen(seen, live={200}, tick=4, grace=2)
        self.assertNotIn(100, seen)                  # gone 3 ticks: forgotten
        self.assertIn(200, seen)

    def test_recently_absent_handles_survive_the_grace(self):
        seen = {100: 3}
        frontend.prune_seen(seen, live=set(), tick=4, grace=2)
        self.assertIn(100, seen)                     # only 1 tick gone

    def test_live_handles_are_never_pruned(self):
        seen = {100: 0}
        frontend.prune_seen(seen, live={100}, tick=99, grace=2)
        self.assertIn(100, seen)


class TestPresentationSurfaces(unittest.TestCase):
    def test_extra_runtimes_share_the_detect_runtime_shape(self):
        # Consumers iterate discover() and must not care which detector made
        # an item; extra runtimes lacked `key` and `serving`.
        saved = lmm.proc_count
        lmm.proc_count = lambda names: 1
        try:
            got = lmm.detect_extra({"extra_runtimes": [
                {"name": "My Agent", "procs": ["x"], "installed_paths": []}]})
        finally:
            lmm.proc_count = saved
        self.assertEqual(got[0]["key"], "my-agent")
        self.assertTrue(got[0]["serving"])
        for field in ("name", "key", "type", "paid", "running", "serving",
                      "procs", "models", "endpoint", "installed"):
            self.assertIn(field, got[0])

    def test_cmd_cache_reports_the_effective_config(self):
        import io
        import contextlib
        cfg = {"cache": {"similarity": 0.8, "ttl_hours": 1}}
        buf = io.StringIO()
        with temp_state():
            with contextlib.redirect_stdout(buf):
                frontend.cmd_cache(cfg)
        out = buf.getvalue()
        self.assertIn("similarity=0.8", out)
        self.assertIn("ttl_hours=1", out)
        self.assertNotIn("0.95", out)                # not the default

    def test_cmd_models_lists_every_runtime_with_models(self):
        import io
        import contextlib
        saved = lmm.discover
        lmm.discover = lambda cfg, with_models=True: [
            {"name": "LM Studio", "models": ["phi-4"]},
            {"name": "Ollama", "models": []},
            {"name": "Jan", "models": ["jan-nano"]}]
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                frontend.cmd_models({})
        finally:
            lmm.discover = saved
        out = buf.getvalue()
        self.assertIn("LM Studio", out)
        self.assertIn("phi-4", out)
        self.assertIn("jan-nano", out)
        self.assertNotIn("Ollama:", out)             # nothing to list there

    def test_version_constant_is_wired(self):
        self.assertRegex(lmm.VERSION, r"^\d+\.\d+\.\d+$")
        self.assertNotEqual(lmm.VERSION, "1.0.0")
        # bumped for GGUF reading, verified cache and the tool-call fix
        self.assertEqual(lmm.VERSION, "1.2.0")


class TestCliSmoke(unittest.TestCase):
    """Every command, in a real subprocess, with an isolated HOME.

    This is the check that was run by hand after every change in this repo's
    history — which is exactly why `lmm hide` could be a silent no-op and
    `lmm cli` could exit 2 for as long as they did. A check nobody runs
    automatically catches nothing.
    """

    COMMANDS = [
        ["--version"], ["discover"], ["discover", "--json"],
        ["status"], ["models"], ["cost"], ["cost", "--days", "7"],
        ["route", "summarize this"], ["route", "--explain", "hi"],
        ["fit", "--vram", "8"], ["bench"], ["cache"], ["examples"],
        ["hide", "claude"], ["ask"],
    ]

    @classmethod
    def setUpClass(cls):
        cls.home = tempfile.mkdtemp(prefix="lmm-smoke-home-")
        cls.root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.home, ignore_errors=True)

    def run_cmd(self, args, home=None):
        import subprocess
        env = dict(os.environ)
        env["HOME"] = home or self.home
        env.pop("OLLAMA_HOST", None)
        return subprocess.run(
            [sys.executable, os.path.join(self.root, "lmm.py")] + args,
            capture_output=True, timeout=120, env=env)

    # Commands that ACT on a provider or a model have nothing to act on with
    # zero configuration, and a CLI must say so through its exit status --
    # `lmm ask` used to exit 0 with its failure on stdout, so a script could
    # not tell an answer from an outage. These may exit non-zero; they may
    # not crash.
    NEEDS_A_TARGET = ("ask", "bench", "fit")

    def test_every_command_succeeds_with_no_config(self):
        # The tool's core promise is that it works with zero configuration:
        # every command runs, none traceback, and the ones that have
        # something to show with no config at all exit 0.
        for args in self.COMMANDS:
            with self.subTest(cmd=" ".join(args)):
                r = self.run_cmd(args)
                err = r.stderr.decode()
                self.assertNotIn("Traceback", err, "lmm %s crashed\n%s"
                                 % (" ".join(args), err[:400]))
                if args[0] in self.NEEDS_A_TARGET:
                    self.assertIn(r.returncode, (1, 2),
                                  "lmm %s has nothing to act on and must say "
                                  "so via its exit status, got %d"
                                  % (" ".join(args), r.returncode))
                else:
                    self.assertEqual(r.returncode, 0,
                                     "lmm %s exited %d\n%s" % (
                                         " ".join(args), r.returncode,
                                         err[:400]))

    def test_every_command_says_something(self):
        # Silence is how `lmm hide` hid a missing dispatch branch for months.
        for args in self.COMMANDS:
            with self.subTest(cmd=" ".join(args)):
                r = self.run_cmd(args)
                self.assertTrue((r.stdout + r.stderr).strip(),
                                "lmm %s printed nothing at all" % " ".join(args))

    def test_examples_output_is_loadable_config(self):
        r = self.run_cmd(["examples"])
        cfg = json.loads(r.stdout.decode())
        self.assertIn("providers", cfg)

    def test_examples_matches_the_committed_sample(self):
        # config.example.json is generated from `lmm examples`; if they drift,
        # the documented shape stops matching the real one.
        r = self.run_cmd(["examples"])
        with open(os.path.join(self.root, "config.example.json"),
                  encoding="utf-8") as fh:
            self.assertEqual(json.loads(r.stdout.decode()), json.load(fh))

    def test_version_is_the_constant(self):
        r = self.run_cmd(["--version"])
        self.assertIn(lmm.VERSION, r.stdout.decode())

    def test_bare_lmm_shows_status_when_no_gui_is_possible(self):
        # The default command is the GUI. On every server, container and ssh
        # session that means no tkinter or no display — and the tool used to
        # print one line about a toolkit the user never asked for, show
        # nothing, and exit. First contact must show STATUS.
        import subprocess
        env = dict(os.environ)
        env["HOME"] = self.home
        env.pop("DISPLAY", None)              # simulate headless even with tk
        r = subprocess.run([sys.executable, os.path.join(self.root, "lmm.py")],
                           capture_output=True, timeout=120, env=env)
        out = r.stdout.decode()
        self.assertEqual(r.returncode, 0, r.stderr.decode()[:400])
        self.assertNotIn("Traceback", r.stderr.decode())
        # it fell back to the text status rather than showing nothing
        self.assertIn("GPU:", out)
        self.assertIn("hub:", out)

    def test_a_closed_pipe_is_not_an_error(self):
        # `lmm discover | head -1` used to end in a BrokenPipeError traceback.
        # A reader closing early is how pipes work, not a failure.
        import subprocess
        env = dict(os.environ)
        env["HOME"] = self.home
        p = subprocess.Popen(
            [sys.executable, os.path.join(self.root, "lmm.py"), "discover"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        p.stdout.readline()              # take one line, then hang up
        p.stdout.close()
        p.wait(timeout=120)
        err = p.stderr.read().decode()
        p.stderr.close()
        self.assertNotIn("Traceback", err)
        self.assertNotIn("BrokenPipeError", err)

    def test_an_unknown_command_fails_loudly(self):
        r = self.run_cmd(["definitely-not-a-command"])
        self.assertNotEqual(r.returncode, 0)

    def test_a_working_directory_config_cannot_run_shell_commands(self):
        # Regression for the RCE: `lmm` reads ./lmm.config.json, and
        # extra_runtimes[].models_cmd is a shell command, so any repo you cd
        # into could run code the moment you typed `lmm`.
        import subprocess
        d = tempfile.mkdtemp(prefix="lmm-smoke-cwd-")
        marker = os.path.join(d, "PWNED")
        try:
            with open(os.path.join(d, "lmm.config.json"), "w",
                      encoding="utf-8") as fh:
                json.dump({"extra_runtimes": [{
                    "name": "Innocent", "procs": [], "installed_paths": [],
                    "models_cmd": "touch %s" % marker}]}, fh)
            env = dict(os.environ)
            env["HOME"] = self.home
            subprocess.run([sys.executable, os.path.join(self.root, "lmm.py"),
                            "discover"], capture_output=True, timeout=120,
                           cwd=d, env=env)
            self.assertFalse(os.path.exists(marker),
                             "a working-directory config executed shell")
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestConcurrency(unittest.TestCase):
    """The hub serves requests on concurrent threads, and both state files are
    maintained by read-modify-replace rewrites. Unlocked, an append racing a
    rewrite is erased by the os.replace — measured at 21.9% of all metering
    events lost under 8 writers before the locks went in. A tool that sells
    measurement must not lose a fifth of its measurements under load."""

    THREADS, PER = 8, 250

    def test_no_metering_event_is_lost_under_concurrent_compaction(self):
        import threading
        with temp_state():
            saved = lmm.USAGE_MAX_BYTES
            lmm.USAGE_MAX_BYTES = 4000        # force frequent compaction
            try:
                def worker(t):
                    for _ in range(self.PER):
                        lmm.log_usage({"provider": "p%d" % t, "kind": "remote",
                                       "in": 10, "out": 10, "usd": 0.001,
                                       "cache": "miss"})
                ts = [threading.Thread(target=worker, args=(t,))
                      for t in range(self.THREADS)]
                for t in ts:
                    t.start()
                for t in ts:
                    t.join()
                st = lmm.hub_cost_stats()
            finally:
                lmm.USAGE_MAX_BYTES = saved
        expected = self.THREADS * self.PER
        self.assertEqual(st["calls"], expected)
        self.assertAlmostEqual(st["measured"], expected * 0.001, places=9)

    def test_no_cache_entry_is_lost_to_a_concurrent_prune(self):
        import threading
        with temp_state():
            def worker(t):
                for i in range(40):
                    msgs = [{"role": "user", "content": "q-%d-%d" % (t, i)}]
                    lmm.cache_store({}, msgs, "m", {"ok": 1}, temperature=0.0)
            ts = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            n = len(lmm.cache_entries(lmm.DEFAULT_CACHE))
        self.assertEqual(n, 320)

    def test_concurrent_observations_all_land(self):
        # record_observation is a read-modify-replace of the whole cache file;
        # two racing observers used to overwrite each other's labels — the
        # evidence the verified cache certifies entries on.
        import threading
        with temp_state():
            msgs = [{"role": "user", "content": "q"}]
            lmm.cache_store({}, msgs, "m", {"ok": 1}, temperature=0.0)
            key = lmm.cache_key(msgs, "m")

            def observe(i):
                lmm.record_observation({}, key, 0.9, i % 2 == 0)
            ts = [threading.Thread(target=observe, args=(i,)) for i in range(30)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            entry = lmm.cache_entries(lmm.DEFAULT_CACHE)[0]
        self.assertEqual(len(entry["obs"]), 30)


class TestUsageCompaction(unittest.TestCase):
    """usage.jsonl is append-only and every reader parses the whole file — the
    GUI on a 5-second timer. Unbounded, the metering that exists to SHOW spend
    makes showing spend linearly slower. Compaction folds old events into one
    rollup line whose totals must be EXACT, or `lmm cost` changes its answer
    just because the log got tidied."""

    def seed(self, n=60):
        import random
        rng = random.Random(7)
        for _ in range(n):
            r = rng.random()
            if r < 0.2:
                lmm.log_usage({"provider": "cache", "kind": "local", "in": 0,
                               "out": 0, "usd": 0.0,
                               "cache": "exact" if r < 0.12 else "semantic",
                               "saved_usd": 0.001})
            elif r < 0.3:
                lmm.log_usage({"provider": "cache", "kind": "local", "in": 0,
                               "out": 0, "usd": 0.0, "cache": "near-miss",
                               "similarity": 0.9})
            else:
                kind = "local" if r < 0.5 else "remote"
                lmm.log_usage({"provider": "p-" + kind, "kind": kind,
                               "in": 100, "out": 50,
                               "usd": 0.0 if kind == "local" else 0.002,
                               "cache": "miss", "stream": True, "ttft_ms": 40,
                               "estimated": r < 0.4, "partial": r < 0.35})

    def totals(self, st):
        return (round(st["measured"], 9), st["calls"], st["hits"],
                round(st["saved_cache"], 9), round(st["est_usd"], 9),
                round(st["partial_usd"], 9), st["partial_calls"],
                st["local_calls"], st["local_tokens"],
                {n: (a["calls"], a["in"], a["out"], round(a["usd"], 9))
                 for n, a in sorted(st["providers"].items())})

    def test_totals_are_identical_across_compaction(self):
        with temp_state():
            self.seed()
            before = self.totals(lmm.hub_cost_stats())
            self.assertTrue(compact(keep=10))
            after = self.totals(lmm.hub_cost_stats())
        self.assertEqual(before, after)

    def test_the_tail_stays_raw_and_the_file_shrinks_to_keep_plus_one(self):
        with temp_state():
            self.seed()
            compact(keep=10)
            with open(lmm.USAGE_LOG, encoding="utf-8") as fh:
                lines = [json.loads(l) for l in fh if l.strip()]
        self.assertEqual(len(lines), 11)              # rollup + 10 raw
        self.assertTrue(lines[0]["rollup"])
        self.assertTrue(all(not l.get("rollup") for l in lines[1:]))

    def test_recompaction_converges_to_one_rollup_with_the_same_totals(self):
        with temp_state():
            self.seed()
            before = self.totals(lmm.hub_cost_stats())
            compact(keep=20)
            compact(keep=5)
            after = self.totals(lmm.hub_cost_stats())
            with open(lmm.USAGE_LOG, encoding="utf-8") as fh:
                rollups = [l for l in fh if json.loads(l).get("rollup")]
            with open(lmm.USAGE_LOG, encoding="utf-8") as fh:
                r = json.loads(next(fh))
        self.assertEqual(before, after)
        self.assertEqual(len(rollups), 1)
        # and the fold count survives the merge
        self.assertEqual(r["events"] + 5, 60)

    def test_ttft_series_is_tail_only_by_design(self):
        # Percentiles cannot be merged, so the rollup must not pretend to
        # carry them; the STREAM TTFT line reads from the raw tail.
        with temp_state():
            self.seed()
            compact(keep=10)
            st = lmm.hub_cost_stats()
        self.assertLessEqual(len(st["ttfts"]), 10)

    def test_cmd_cache_counts_hits_through_rollups(self):
        import io
        import contextlib
        with temp_state():
            self.seed()
            before = io.StringIO()
            with contextlib.redirect_stdout(before):
                frontend.cmd_cache({})
            compact(keep=5)
            after = io.StringIO()
            with contextlib.redirect_stdout(after):
                frontend.cmd_cache({})
        line_b = [l for l in before.getvalue().splitlines() if "hits:" in l][0]
        line_a = [l for l in after.getvalue().splitlines() if "hits:" in l][0]
        self.assertEqual(line_b, line_a)

    def test_log_usage_compacts_automatically_past_the_cap(self):
        with temp_state():
            saved = lmm.USAGE_MAX_BYTES
            lmm.USAGE_MAX_BYTES = 2000        # tiny cap for the test
            try:
                self.seed(200)
                with open(lmm.USAGE_LOG, encoding="utf-8") as fh:
                    lines = fh.readlines()
                self.assertLess(os.path.getsize(lmm.USAGE_LOG), 500_000)
                self.assertTrue(any(json.loads(l).get("rollup") for l in lines))
            finally:
                lmm.USAGE_MAX_BYTES = saved

    def test_nothing_to_fold_is_a_clean_no(self):
        with temp_state():
            lmm.log_usage({"provider": "p", "kind": "remote", "in": 1, "out": 1,
                           "usd": 0.1, "cache": "miss"})
            self.assertFalse(compact(keep=10))

    def test_report_names_the_rollup(self):
        with temp_state():
            self.seed()
            compact(keep=5)
            block = "\n".join(lmm.hub_cost_block({}, lmm.merged_pricing({})))
        self.assertIn("includes a rollup of older events", block)


class TestGuiLogic(unittest.TestCase):
    """tkinter is not installed everywhere, and that was the standing excuse
    for this layer having no coverage while it drifted. The logic never needed
    Tk — it was just trapped inside a closure."""

    ITEMS = [
        {"name": "Ollama", "type": "local", "paid": False, "running": True,
         "serving": True, "procs": 2, "models": ["qwen2.5:7b", "llama3.1:8b"],
         "endpoint": "http://localhost:11434/v1"},
        {"name": "Claude Code", "type": "remote", "paid": True, "running": False,
         "serving": False, "procs": 0, "models": [],
         "endpoint": "api.anthropic.com"},
    ]

    def test_rows_carry_running_and_serving_separately(self):
        # `running` is "window or port"; `serving` is "the port answered".
        # Collapsing them was what made a GUI app and a headless server look
        # identical in the table.
        rows = frontend.gui_rows(self.ITEMS)
        self.assertEqual(len(rows), 2)
        (vals, tag) = rows[0]
        self.assertEqual(tag, "on")
        self.assertEqual(vals[3], "YES")          # running
        self.assertEqual(vals[4], "YES")          # serving
        self.assertEqual(rows[1][1], "off")
        self.assertEqual(rows[1][0][4], "-")      # not serving

    def test_rows_have_one_cell_per_column(self):
        for vals, _ in frontend.gui_rows(self.ITEMS):
            self.assertEqual(len(vals), 8)        # matches the GUI's `cols`

    def test_paid_and_models_render(self):
        vals = frontend.gui_rows(self.ITEMS)[0][0]
        self.assertEqual(vals[2], "free")
        self.assertIn("qwen2.5:7b", vals[6])
        self.assertEqual(frontend.gui_rows(self.ITEMS)[1][0][2], "PAID")
        self.assertEqual(frontend.gui_rows(self.ITEMS)[1][0][6], "-")

    def test_rows_survive_a_sparse_item(self):
        # detect_extra used to omit keys detect_runtime emits.
        rows = frontend.gui_rows([{"name": "X", "type": "local", "running": False}])
        self.assertEqual(rows[0][0][4], "-")
        self.assertEqual(rows[0][0][5], 0)

    def test_model_choices_are_deduped_and_sorted(self):
        items = self.ITEMS + [{"name": "LM Studio", "type": "local",
                               "running": True, "models": ["llama3.1:8b", "phi-4"]}]
        self.assertEqual(frontend.gui_model_choices(items),
                         ["llama3.1:8b", "phi-4", "qwen2.5:7b"])

    def test_model_choices_empty_when_nothing_is_running(self):
        self.assertEqual(frontend.gui_model_choices([{"name": "X", "models": []}]), [])

    def test_gpu_label_warns_only_when_vram_is_tight(self):
        tight = frontend.gpu_label({"name": "RTX", "used": 900, "total": 1000, "pct": 90})
        roomy = frontend.gpu_label({"name": "RTX", "used": 100, "total": 1000, "pct": 10})
        self.assertIn("\u26a0", tight)
        self.assertNotIn("\u26a0", roomy)
        self.assertIn("RTX", roomy)

    def test_gpu_label_handles_no_gpu(self):
        self.assertEqual(frontend.gpu_label(None), "GPU: n/a")


class TestWatchDecision(unittest.TestCase):
    """The daemon's per-tick decision, without the Windows API. This is the
    part that regressed before — recycled handles being skipped forever."""

    def watchlist(self):
        return frontend.watch_list({})

    def test_watchlist_covers_the_registry(self):
        wl = self.watchlist()
        names = {rt for rt, _ in wl}
        self.assertIn("claude", names)
        self.assertTrue(all(kw == kw.lower() for _, kw in wl))

    def test_watchlist_includes_user_runtimes(self):
        wl = frontend.watch_list({"extra_runtimes": [{"name": "My Agent"}]})
        self.assertIn(("My Agent", "my agent"), wl)

    def test_a_window_is_acted_on_once(self):
        seen, wl = {}, [("claude", "claude")]
        first = frontend.watch_new_windows([(100, "Claude")], seen, 1, wl)
        self.assertEqual([(h, t, r) for h, t, r in first],
                         [(100, "Claude", "claude")])
        again = frontend.watch_new_windows([(100, "Claude")], seen, 2, wl)
        self.assertEqual(again, [])               # not hidden twice

    def test_a_recycled_handle_is_treated_as_new_again(self):
        # Windows reuses HWND values. Before prune_seen, a new window landing
        # on an old handle was skipped forever and never hidden.
        seen, wl = {}, [("claude", "claude")]
        frontend.watch_new_windows([(100, "Claude")], seen, 1, wl)
        frontend.prune_seen(seen, live=set(), tick=5, grace=2)   # window closed
        again = frontend.watch_new_windows([(100, "Claude")], seen, 6, wl)
        self.assertEqual(len(again), 1)

    def test_the_owning_runtime_is_identified(self):
        wl = frontend.watch_list({})
        got = frontend.watch_new_windows([(1, "ChatGPT")], {}, 1, wl)
        self.assertEqual(got[0][2], "chatgpt")

    def test_an_unmatched_title_still_reports_a_placeholder(self):
        got = frontend.watch_new_windows([(1, "Some Other App")], {}, 1,
                                    [("claude", "claude")])
        self.assertEqual(got[0][2], "?")

    def test_seen_is_stamped_for_every_window_not_just_new_ones(self):
        # An existing window must keep its tick refreshed, or prune_seen will
        # forget a window that is still open.
        seen, wl = {}, [("claude", "claude")]
        frontend.watch_new_windows([(100, "Claude")], seen, 1, wl)
        frontend.watch_new_windows([(100, "Claude")], seen, 7, wl)
        self.assertEqual(seen[100], 7)


class StubBackend(object):
    """A real OpenAI-compatible server on localhost, speaking both streamed and
    non-streamed replies.

    The hub is the product's core and had zero test coverage, because testing a
    proxy needs something real to proxy to. This is that something — it also
    echoes back which roles and params it received, so the tests can assert the
    hub forwarded them rather than trusting it did.
    """

    def __init__(self, answer="The answer is 4, because 2 plus 2 equals 4."):
        import http.server
        import socketserver
        import threading
        self.seen = []
        self.seen_get = []
        answer_text = answer
        seen = self.seen
        seen_get = self.seen_get

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                # A real /v1/models, so fetch_models and the hub's model
                # aggregation are tested over real HTTP, not monkeypatches.
                seen_get.append(self.path)
                if self.path.rstrip("/").endswith("/v1/models"):
                    return self._json({"object": "list", "data": [
                        {"id": "stub-model-a", "object": "model"},
                        {"id": "stub-model-b", "object": "model"}]})
                self.send_response(404)
                self.end_headers()

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n).decode() or "{}")
                seen.append(req)
                roles = ",".join(m.get("role", "?") for m in req.get("messages", []))
                text = "%s [roles=%s max_tokens=%s]" % (
                    answer_text, roles, req.get("max_tokens"))
                if req.get("stream"):
                    return self._stream(req, text)
                self._json({
                    "id": "chatcmpl-stub", "object": "chat.completion",
                    "model": req.get("model"),
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": text}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 200},
                })

            def _stream(self, req, text):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                base = {"id": "chatcmpl-stub", "object": "chat.completion.chunk",
                        "created": 0, "model": req.get("model")}

                def frame(o):
                    self.wfile.write(("data: " + json.dumps(o) + "\n\n").encode())
                    self.wfile.flush()
                frame(dict(base, choices=[{"index": 0,
                                           "delta": {"role": "assistant"},
                                           "finish_reason": None}]))
                for w in text.split(" "):
                    frame(dict(base, choices=[{"index": 0,
                                               "delta": {"content": w + " "},
                                               "finish_reason": None}]))
                frame(dict(base, choices=[{"index": 0, "delta": {},
                                           "finish_reason": "stop"}]))
                if (req.get("stream_options") or {}).get("include_usage"):
                    frame(dict(base, choices=[],
                               usage={"prompt_tokens": 100,
                                      "completion_tokens": 200}))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

            def _json(self, obj):
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        self.httpd = S(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        # poll_interval is not a tuning knob here, it is the teardown cost:
        # shutdown() blocks until serve_forever notices, which happens on a
        # poll boundary. The stdlib default of 0.5 s made every stop() cost
        # ~0.48 s, and the suite creates one of these per test — measured at
        # 16 s of a 25 s run, spent entirely on waiting to stop.
        self.thread = threading.Thread(
            target=lambda: self.httpd.serve_forever(poll_interval=0.01),
            daemon=True)
        self.thread.start()

    @property
    def base_url(self):
        return "http://127.0.0.1:%d/v1" % self.port

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class HubServer(object):
    """Run lmm's own hub in a thread, against a stub backend."""

    def __init__(self, cfg, host="127.0.0.1"):
        import threading
        import socket
        s = socket.socket()
        s.bind((host, 0))
        self.port = s.getsockname()[1]
        s.close()
        # quiet=True, NOT redirect_stdout: contextlib.redirect_stdout swaps
        # sys.stdout for the WHOLE PROCESS, and this thread never exits — the
        # fixture was silently swallowing every print in the interpreter for
        # the rest of the run.
        self.thread = threading.Thread(
            target=frontend.cmd_serve_hub, args=(cfg, host, self.port),
            kwargs={"quiet": True}, daemon=True)
        self.thread.start()
        for _ in range(100):                  # wait for the listener
            try:
                socket.create_connection((host, self.port), timeout=0.1).close()
                return
            except OSError:
                time.sleep(0.02)

    def request(self, path, body=None, headers=None, method=None):
        import urllib.request
        import urllib.error
        url = "http://127.0.0.1:%d%s" % (self.port, path)
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method,
                                     headers=headers or {})
        if data:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()


try:
    import openai as _openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class TestTrayWiring(unittest.TestCase):
    """setup_tray existed with zero callers after the merge — the minimize-
    to-tray feature was silently severed. Headless CI cannot click a tray
    icon, so these pin what can be pinned: the platform contract and the
    wiring."""

    def test_non_windows_is_a_no_op(self):
        if os.name == "nt":
            self.skipTest("Windows would actually build the tray")
        self.assertIsNone(frontend.setup_tray(object()))

    def test_launch_gui_wires_the_tray_and_a_safe_close(self):
        """The GUI must call setup_tray, and when there is no tray (the
        return is None) X must destroy the window — withdrawing with no tray
        icon would orphan the process with no way to bring it back."""
        import ast
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "frontend.py")
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        gui = next(n for n in tree.body
                   if isinstance(n, ast.FunctionDef) and n.name == "launch_gui")
        calls = [c.func.id for c in ast.walk(gui)
                 if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)]
        self.assertIn("setup_tray", calls)
        src = ast.get_source_segment(open(path, encoding="utf-8").read(), gui)
        self.assertIn("root.destroy", src)


class TestPerModelRouting(unittest.TestCase):
    """The hub's /v1/models and per-model routing, over plain HTTP (no SDK
    needed), so this coverage exists even where the openai package is absent.
    Master fixed the "stub model ids" bug in fbbc59e; the merge undid it and
    nothing noticed, because resolve_provider_by_model kept existing with
    zero callers — a reader-less writer's mirror image."""

    def setUp(self):
        self.stub = StubBackend()
        self.cfg = {"providers": {"stub": {
            "api_key": "k", "base_url": self.stub.base_url,
            "model": "default-model", "kind": "local"}},
            "ask_order": ["stub"]}

    def tearDown(self):
        self.stub.stop()

    def _get(self, hub, path):
        import urllib.request
        with urllib.request.urlopen(
                "http://127.0.0.1:%d%s" % (hub.port, path), timeout=10) as r:
            return json.loads(r.read().decode())

    def _post(self, hub, payload):
        import urllib.request
        req = urllib.request.Request(
            "http://127.0.0.1:%d/v1/chat/completions" % hub.port,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode())

    def test_models_lists_real_ids_and_the_provider_alias(self):
        with temp_state():
            hub = HubServer(self.cfg)
            ids = [m["id"] for m in self._get(hub, "/v1/models")["data"]]
        self.assertEqual(ids, ["stub-model-a", "stub-model-b", "stub"])

    def test_a_model_id_is_forwarded_verbatim(self):
        with temp_state():
            hub = HubServer(self.cfg)
            res = self._post(hub, {"model": "stub-model-a", "lmm_no_cache": 1,
                                   "messages": [{"role": "user",
                                                 "content": "hi"}]})
        self.assertEqual(res["model"], "stub-model-a")

    def test_a_provider_name_still_routes_to_its_default_model(self):
        with temp_state():
            hub = HubServer(self.cfg)
            res = self._post(hub, {"model": "stub", "lmm_no_cache": 1,
                                   "messages": [{"role": "user",
                                                 "content": "hi"}]})
        self.assertEqual(res["model"], "default-model")

    def test_model_lists_are_cached_not_refetched_per_request(self):
        """Per-request fetch_models measured +80% P50 on model-id requests —
        a full backend round-trip per call. The hub is long-lived, so the
        list is cached for MODELS_CACHE_TTL; N requests cost one GET."""
        with temp_state():
            hub = HubServer(self.cfg)
            for _ in range(5):
                self._post(hub, {"model": "stub-model-a", "lmm_no_cache": 1,
                                 "messages": [{"role": "user",
                                               "content": "hi"}]})
            self._get(hub, "/v1/models")
        gets = [p for p in self.stub.seen_get if "models" in p]
        self.assertEqual(len(gets), 1,
                         "each request re-fetched the model list: %r" % gets)

    def test_the_cache_expires(self):
        prov = self.cfg["providers"]["stub"]
        self.assertEqual(lmm.fetch_models(prov),
                         ["stub-model-a", "stub-model-b"])
        self.assertEqual(lmm.fetch_models(prov),
                         ["stub-model-a", "stub-model-b"])
        self.assertEqual(
            len([p for p in self.stub.seen_get if "models" in p]), 1)
        with lmm._MODELS_CACHE_LOCK:      # rewind the clock: entry now stale
            ts, ids = lmm._MODELS_CACHE[prov["base_url"]]
            lmm._MODELS_CACHE[prov["base_url"]] = (ts - 2 * lmm.MODELS_CACHE_TTL, ids)
        lmm.fetch_models(prov)
        self.assertEqual(
            len([p for p in self.stub.seen_get if "models" in p]), 2,
            "a stale entry was served past its TTL")

    def test_liveness_is_not_probed_per_request(self):
        """Every hub request used to run both implicit-provider detectors —
        an HTTP probe plus subprocess spawns — even when the request named an
        explicit provider. Measured: 2.00 spawns + 1.00 probes per request,
        a ~153 req/s ceiling. The routing-path wrappers are memoised for
        IMPLICIT_CACHE_TTL; N requests may detect at most once each."""
        counts = {"ollama": 0, "lmstudio": 0}
        saved_o, saved_l = lmm.detect_ollama, lmm.detect_lmstudio
        lmm.detect_ollama = lambda *a, **k: (
            counts.__setitem__("ollama", counts["ollama"] + 1),
            {"running": False})[1]
        lmm.detect_lmstudio = lambda *a, **k: (
            counts.__setitem__("lmstudio", counts["lmstudio"] + 1),
            {"running": False})[1]
        try:
            with lmm._IMPLICIT_CACHE_LOCK:
                lmm._IMPLICIT_CACHE.clear()
            with temp_state():
                hub = HubServer(self.cfg)
                for _ in range(8):
                    self._post(hub, {"model": "stub", "lmm_no_cache": 1,
                                     "messages": [{"role": "user",
                                                   "content": "hi"}]})
            self.assertLessEqual(counts["ollama"], 1, counts)
            self.assertLessEqual(counts["lmstudio"], 1, counts)
        finally:
            lmm.detect_ollama, lmm.detect_lmstudio = saved_o, saved_l

    def test_the_liveness_memo_expires(self):
        calls = []
        with lmm._IMPLICIT_CACHE_LOCK:
            lmm._IMPLICIT_CACHE.clear()
        probe = lambda: (calls.append(1), None)[1]
        lmm._memo_implicit("x", probe)
        lmm._memo_implicit("x", probe)
        self.assertEqual(len(calls), 1, "the memo did not hold")
        with lmm._IMPLICIT_CACHE_LOCK:      # rewind the clock: entry stale
            ts, val = lmm._IMPLICIT_CACHE["x"]
            lmm._IMPLICIT_CACHE["x"] = (ts - 2 * lmm.IMPLICIT_CACHE_TTL, val)
        lmm._memo_implicit("x", probe)
        self.assertEqual(len(calls), 2, "a stale entry outlived its TTL")

    def test_an_unknown_model_is_a_clear_400(self):
        import urllib.error
        with temp_state():
            hub = HubServer(self.cfg)
            saved = lmm.local_ollama_provider
            lmm.local_ollama_provider = lambda: None    # no safety net
            try:
                with self.assertRaises(urllib.error.HTTPError) as cm:
                    self._post(hub, {"model": "no-such-model",
                                     "messages": [{"role": "user",
                                                   "content": "hi"}]})
                self.assertEqual(cm.exception.code, 400)
            finally:
                lmm.local_ollama_provider = saved


@unittest.skipUnless(HAS_OPENAI, "openai SDK not installed — the suite stays "
                     "zero-dependency; this class runs only where the real "
                     "client library happens to be available")
class TestOpenAISdkCompat(unittest.TestCase):
    """The README's core claim is "point your apps at the hub". Apps do not
    speak hand-rolled curl — they speak the OpenAI client library, which has
    its own strictness about SSE framing, [DONE], and the usage chunk's empty
    `choices`. Until the real SDK had run against the hub, that claim was
    unproven."""

    @classmethod
    def setUpClass(cls):
        cls.backend = StubBackend()

    @classmethod
    def tearDownClass(cls):
        cls.backend.stop()

    def cfg(self, **hub):
        return {"providers": {"stub": {"api_key": "k",
                                       "base_url": self.backend.base_url,
                                       "model": "m", "kind": "remote"}},
                "hub": hub, "cache": {"enabled": False}}

    def client(self, hub, api_key="dummy"):
        return _openai.OpenAI(base_url="http://127.0.0.1:%d/v1" % hub.port,
                              api_key=api_key)

    def test_non_streaming_roundtrip(self):
        with temp_state():
            hub = HubServer(self.cfg())
            r = self.client(hub).chat.completions.create(
                model="stub", max_tokens=64,
                messages=[{"role": "system", "content": "be terse"},
                          {"role": "user", "content": "hi"}])
        self.assertIn("roles=system,user", r.choices[0].message.content)
        self.assertEqual(r.usage.completion_tokens, 200)

    def test_streaming_assembles_through_the_sdk(self):
        # The SDK yields ZERO chunks, with no error, if the blank-line
        # framing is wrong — the silent failure mode this class exists for.
        with temp_state():
            hub = HubServer(self.cfg())
            parts = []
            for chunk in self.client(hub).chat.completions.create(
                    model="stub", stream=True,
                    messages=[{"role": "user", "content": "count"}]):
                if chunk.choices and chunk.choices[0].delta.content:
                    parts.append(chunk.choices[0].delta.content)
        self.assertGreater(len(parts), 3)
        self.assertIn("The answer is 4", "".join(parts))

    def test_stream_usage_chunk_parses_in_the_sdk(self):
        # That final chunk carries choices: [] — stricter parsers reject it,
        # so it must only appear when the client opted in, and must parse.
        with temp_state():
            hub = HubServer(self.cfg())
            usage = None
            for chunk in self.client(hub).chat.completions.create(
                    model="stub", stream=True,
                    stream_options={"include_usage": True},
                    messages=[{"role": "user", "content": "count"}]):
                if chunk.usage:
                    usage = chunk.usage
        self.assertIsNotNone(usage)
        self.assertEqual(usage.completion_tokens, 200)

    def test_models_list_through_the_sdk(self):
        """Real model ids from the backend, plus the provider name as a
        routable alias. Listing only provider names — as the hub once did —
        meant a client could never pick between two models on one backend."""
        with temp_state():
            hub = HubServer(self.cfg())
            ids = [m.id for m in self.client(hub).models.list()]
        self.assertEqual(ids, ["stub-model-a", "stub-model-b", "stub"])

    def test_a_real_model_id_routes_and_is_forwarded(self):
        """Picking a model from /v1/models must reach the provider that
        serves it AND request that exact model — not the provider's default.
        The stub echoes the model it was asked for, so this is end-to-end."""
        with temp_state():
            hub = HubServer(self.cfg())
            res = self.client(hub).chat.completions.create(
                model="stub-model-b",
                messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(res.model, "stub-model-b")

    def test_wrong_token_is_an_authentication_error(self):
        with temp_state():
            hub = HubServer(self.cfg(token="tok_real"))
            with self.assertRaises(_openai.AuthenticationError):
                self.client(hub, api_key="wrong").chat.completions.create(
                    model="stub", messages=[{"role": "user", "content": "x"}])
            r = self.client(hub, api_key="tok_real").chat.completions.create(
                model="stub", messages=[{"role": "user", "content": "x"}])
        self.assertTrue(r.choices[0].message.content)


class TestHubServer(unittest.TestCase):
    """End-to-end through the real HTTP server. `cmd_serve_hub` is the core of
    the product and had no test at all — every hub bug in this repo's history
    was found by hand, one at a time."""

    @classmethod
    def setUpClass(cls):
        cls.backend = StubBackend()

    @classmethod
    def tearDownClass(cls):
        cls.backend.stop()

    def cfg(self, **hub):
        return {"providers": {"stub": {"api_key": "SECRET-KEY",
                                       "base_url": self.backend.base_url,
                                       "model": "m", "kind": "remote",
                                       "price": {"in": 1.0, "out": 2.0}}},
                "hub": hub, "cache": {"enabled": False}}

    def test_models_endpoint_lists_providers(self):
        with temp_state():
            hub = HubServer(self.cfg())
            status, body = hub.request("/v1/models")
        self.assertEqual(status, 200)
        self.assertIn("stub", body)

    def test_full_conversation_reaches_the_backend(self):
        # The hub used to forward only messages[0], silently dropping the
        # system prompt and every prior turn.
        del self.backend.seen[:]
        with temp_state():
            hub = HubServer(self.cfg())
            status, body = hub.request("/v1/chat/completions", {
                "model": "stub", "max_tokens": 128,
                "messages": [{"role": "system", "content": "be terse"},
                             {"role": "user", "content": "hi"},
                             {"role": "assistant", "content": "hello"},
                             {"role": "user", "content": "and now?"}]})
        self.assertEqual(status, 200)
        self.assertIn("roles=system,user,assistant,user", body)
        self.assertIn("max_tokens=128", body)     # passthrough params too

    def test_streaming_produces_well_formed_sse(self):
        with temp_state():
            hub = HubServer(self.cfg())
            status, body = hub.request("/v1/chat/completions", {
                "model": "stub", "stream": True,
                "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200)
        self.assertTrue(body.endswith("data: [DONE]\n\n"))
        events = [e for e in body.split("\n\n") if e.strip()]
        self.assertGreater(len(events), 2)
        for e in events:
            self.assertTrue(e.startswith("data: "), e[:40])
            payload = e[len("data: "):]
            if payload != "[DONE]":
                json.loads(payload)               # every event is one object

    def test_a_streamed_call_is_metered(self):
        with temp_state():
            hub = HubServer(self.cfg())
            hub.request("/v1/chat/completions", {
                "model": "stub", "stream": True,
                "messages": [{"role": "user", "content": "meter me"}]})
            time.sleep(0.2)                       # let the finally block land
            events = lmm.read_usage()
        self.assertTrue(events)
        self.assertTrue(events[0]["stream"])
        self.assertEqual(events[0]["out"], 200)   # real usage, not estimated

    def test_token_gates_every_endpoint(self):
        with temp_state():
            hub = HubServer(self.cfg(token="tok_abc123"))
            no_tok = hub.request("/v1/models")
            bad = hub.request("/v1/models",
                              headers={"Authorization": "Bearer nope"})
            good = hub.request("/v1/models",
                               headers={"Authorization": "Bearer tok_abc123"})
            post = hub.request("/v1/chat/completions", {
                "model": "stub",
                "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(no_tok[0], 401)
        self.assertEqual(bad[0], 401)
        self.assertEqual(good[0], 200)
        self.assertEqual(post[0], 401)            # POST is gated too

    def test_a_denial_leaks_neither_token_nor_api_key(self):
        with temp_state():
            hub = HubServer(self.cfg(token="tok_abc123"))
            status, body = hub.request("/v1/models")
        self.assertEqual(status, 401)
        self.assertNotIn("tok_abc123", body)
        self.assertNotIn("SECRET-KEY", body)

    def test_unknown_path_is_404(self):
        with temp_state():
            hub = HubServer(self.cfg())
            self.assertEqual(hub.request("/nope")[0], 404)


class TestCommandSurfaces(unittest.TestCase):
    """The commands where drift went unnoticed longest, because nothing here
    was covered: a silent no-op `hide`, a `cli` that exited 2, a GUI cost label
    showing prose."""

    def test_hide_returns_advice_off_windows_without_raising(self):
        msg = lmm.hide_taskbar("claude")
        self.assertIsInstance(msg, str)
        self.assertTrue(msg.strip())

    def test_hide_reports_an_unknown_runtime(self):
        self.assertIn("unknown", lmm.hide_taskbar("definitely-not-a-runtime").lower())

    def test_dash_writes_html_a_browser_could_open(self):
        import io
        import contextlib
        saved_disc, saved_open = lmm.discover, lmm.webbrowser.open
        lmm.discover = lambda cfg, with_models=True: [
            {"name": "Ollama", "key": "ollama", "type": "local", "paid": False,
             "running": True, "serving": True, "procs": 1, "models": ["m"],
             "endpoint": "http://localhost:11434/v1", "installed": True}]
        lmm.webbrowser.open = lambda *a, **k: True
        try:
            with temp_state():
                d = tempfile.mkdtemp(prefix="lmm-dash-")
                saved_home = lmm.HOME
                lmm.HOME = d
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        frontend.cmd_dash({})
                    path = os.path.join(d, ".lmm_dashboard.html")
                    self.assertTrue(os.path.isfile(path))
                    with open(path, encoding="utf-8") as fh:
                        html = fh.read()
                finally:
                    lmm.HOME = saved_home
        finally:
            lmm.discover, lmm.webbrowser.open = saved_disc, saved_open
        self.assertTrue(html.startswith("<!doctype"))
        self.assertIn("<th>Serving</th>", html)
        self.assertIn("Ollama", html)

    def test_serve_builds_an_ollama_pull(self):
        import io
        import contextlib
        ran = []
        saved = lmm.run
        lmm.run = lambda cmd: ran.append(cmd) or type(
            "R", (), {"stdout": "ok", "stderr": "", "returncode": 0})()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                frontend.cmd_serve("qwen2.5-coder:7b")
        finally:
            lmm.run = saved
        self.assertIn("ollama pull qwen2.5-coder:7b", ran)

    def test_serve_without_a_model_prints_usage_and_runs_nothing(self):
        import io
        import contextlib
        ran = []
        saved = lmm.run
        lmm.run = lambda cmd: ran.append(cmd)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                frontend.cmd_serve(None)
        finally:
            lmm.run = saved
        self.assertEqual(ran, [])
        self.assertIn("usage:", buf.getvalue())

    def test_autostart_writes_nothing_off_windows(self):
        # This command registers a login service. The test must confirm the
        # non-Windows path is inert WITHOUT letting it install anything.
        import io
        import contextlib
        ran = []
        saved_run, saved_os = lmm.run, os.name
        lmm.run = lambda cmd: ran.append(cmd)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                frontend.cmd_autostart()
        finally:
            lmm.run = saved_run
        if saved_os != "nt":
            self.assertEqual(ran, [], "autostart shelled out on a non-Windows host")
            self.assertTrue(buf.getvalue().strip())


class TestModuleSplit(unittest.TestCase):
    """`lmm` is three files: an engine, a presentation layer, and a shim.

    The split is only worth having if it holds, so these assert the boundary
    rather than trusting it: the engine must not grow a cmd_* handler, the
    entry point must still expose both halves under one name, and rebinding an
    engine symbol must actually be seen by the engine.
    """

    def test_backend_holds_no_command_handlers(self):
        stray = [n for n in dir(lmm) if n.startswith("cmd_")]
        self.assertEqual(stray, [], "engine grew a CLI handler: %s" % stray)
        self.assertFalse(hasattr(lmm, "main"),
                         "argument parsing belongs to the frontend")

    def test_frontend_holds_the_command_handlers(self):
        for name in ("cmd_ask", "cmd_serve_hub", "cmd_selftest", "cmd_doctor",
                     "cmd_fit", "cmd_cache", "main"):
            self.assertTrue(callable(getattr(frontend, name, None)), name)

    def test_entry_point_reexports_both_layers(self):
        import lmm as entry
        for name in ("hub_complete", "cache_lookup", "read_gguf"):   # engine
            self.assertTrue(hasattr(entry, name), name)
        for name in ("cmd_ask", "launch_gui", "main"):               # frontend
            self.assertTrue(hasattr(entry, name), name)

    def test_rebinding_an_engine_symbol_is_seen_by_the_engine(self):
        """The reason the suite patches `backend`, not the `lmm` shim.

        `from backend import *` copies values at import time, so a name
        rebound on the shim afterwards is invisible to the engine functions
        that call it. Every patch in this file depends on that being true of
        the defining module and not of the shim, so it is asserted once here.
        """
        import lmm as entry
        saved = lmm.embed_text
        try:
            lmm.embed_text = lambda text, model: [1.0, 0.0]
            self.assertEqual(lmm.embed_text("x", "m"), [1.0, 0.0])
            self.assertIsNot(entry.embed_text, lmm.embed_text,
                             "the shim holds its own copy, as star-import does")
        finally:
            lmm.embed_text = saved


class TestCommandWiring(unittest.TestCase):
    """Every registered subcommand must be dispatched, and vice versa.

    `lmm hide` was a silent no-op and `lmm cli` exited 2 because a subparser
    existed with no matching branch, and a branch existed with no subparser.
    Both were invisible to a suite that only tested functions, so the wiring
    is checked structurally, over the whole surface at once.
    """

    def _parsed(self):
        import ast
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "frontend.py")
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        main = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "main")
        registered, dispatched = set(), set()
        for node in ast.walk(main):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_parser"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)):
                registered.add(node.args[0].value)
            if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) \
                    and node.left.id == "cmd":
                for c in node.comparators:
                    if isinstance(c, ast.Constant) and isinstance(c.value, str):
                        dispatched.add(c.value)
        return registered, dispatched

    def test_every_subcommand_is_dispatched(self):
        registered, dispatched = self._parsed()
        self.assertTrue(registered, "no subcommands found")
        self.assertEqual(sorted(registered - dispatched), [],
                         "registered but never dispatched (a silent no-op)")

    def test_every_dispatch_branch_has_a_subcommand(self):
        registered, dispatched = self._parsed()
        self.assertEqual(sorted(dispatched - registered), [],
                         "dispatched but unregistered (argparse exits 2)")

    def test_surface_the_selftest_gate_requires_is_present(self):
        """`lmm selftest` — this repo's CI gate — asserts these exist."""
        registered, _ = self._parsed()
        needed = ["discover", "status", "models", "cost", "route", "serve",
                  "ask", "chat", "config", "log", "selftest", "doctor",
                  "stop", "dash", "gui", "watch", "autostart", "hide",
                  "examples"]
        self.assertEqual([c for c in needed if c not in registered], [])


class TestUnifiedCallProvider(unittest.TestCase):
    """One transport serves the hub and the interactive commands.

    The hub needs caller-supplied parameters forwarded (tools, max_tokens);
    `lmm ask` and `lmm chat` need a full history and a token stream. Those
    grew as two signatures on two branches, so the merged one is pinned here.
    """

    def setUp(self):
        self.stub = StubBackend()
        self.prov = {"api_key": "k", "base_url": self.stub.base_url,
                     "model": "m", "kind": "local"}

    def tearDown(self):
        self.stub.stop()

    def test_messages_win_over_prompt(self):
        lmm.call_provider(self.prov, "ignored",
                          messages=[{"role": "system", "content": "be terse"},
                                    {"role": "user", "content": "hi"}])
        sent = self.stub.seen[-1]["messages"]
        self.assertEqual([m["role"] for m in sent], ["system", "user"])

    def test_bare_prompt_becomes_one_user_turn(self):
        lmm.call_provider(self.prov, "hi")
        self.assertEqual(self.stub.seen[-1]["messages"],
                         [{"role": "user", "content": "hi"}])

    def test_extra_params_are_forwarded(self):
        lmm.call_provider(self.prov, "hi", extra={"max_tokens": 7})
        self.assertEqual(self.stub.seen[-1]["max_tokens"], 7)

    def test_non_streaming_never_asks_upstream_to_stream(self):
        lmm.call_provider(self.prov, "hi")
        self.assertFalse(self.stub.seen[-1].get("stream"))

    def test_stream_returns_text_chunks(self):
        gen = lmm.call_provider(self.prov, "hi", stream=True)
        self.assertNotIsInstance(gen, dict)
        text = "".join(gen)
        self.assertIn("answer is 4", text)
        self.assertTrue(self.stub.seen[-1]["stream"])

    def test_a_provider_with_no_model_reports_instead_of_calling(self):
        res = lmm.call_provider({"base_url": "http://127.0.0.1:1/v1"}, "hi")
        self.assertIn("error", res)


class TestRunTimeout(unittest.TestCase):
    """`lmm pull` downloads for minutes; a status probe must not wait that long."""

    def test_default_is_short(self):
        import inspect
        self.assertEqual(
            inspect.signature(lmm.run).parameters["timeout"].default, 25)

    def test_caller_can_extend_it(self):
        r = lmm.run("exit 0", timeout=120)
        self.assertIsNotNone(r)
        self.assertEqual(r.returncode, 0)


class TestMergedCommands(unittest.TestCase):
    """Behaviour that exists because two branches each solved half of it."""

    def _capture(self, fn, *a, **kw):
        """Both streams, joined: diagnostics go to stderr now (stdout is the
        answer and nothing else), and these tests assert on what is SAID,
        not on which stream said it."""
        import io
        import contextlib
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            fn(*a, **kw)
        return out.getvalue() + err.getvalue()

    def test_discover_save_seeds_ask_order_from_what_is_running(self):
        saved_home, saved_disc = lmm.HOME, lmm.discover
        d = tempfile.mkdtemp(prefix="lmm-home-")
        try:
            lmm.HOME = d
            lmm.discover = lambda cfg, **kw: [
                {"name": "Ollama", "running": True, "paid": False},
                {"name": "Jan", "running": False, "paid": False}]
            out = self._capture(frontend.cmd_discover, {}, False, True)
            with open(os.path.join(d, ".lmm", "config.json")) as f:
                written = json.load(f)
            self.assertEqual(written["ask_order"], ["Ollama"])
            self.assertIn("Ollama", out)
        finally:
            lmm.HOME, lmm.discover = saved_home, saved_disc
            shutil.rmtree(d, ignore_errors=True)

    def test_discover_save_with_nothing_running_writes_nothing(self):
        saved_home, saved_disc = lmm.HOME, lmm.discover
        d = tempfile.mkdtemp(prefix="lmm-home-")
        try:
            lmm.HOME = d
            lmm.discover = lambda cfg, **kw: [
                {"name": "Ollama", "running": False, "paid": False}]
            out = self._capture(frontend.cmd_discover, {}, False, True)
            self.assertFalse(os.path.exists(os.path.join(d, ".lmm", "config.json")),
                             "wrote an empty priority list over the user's config")
            self.assertIn("nothing to save", out)
        finally:
            lmm.HOME, lmm.discover = saved_home, saved_disc
            shutil.rmtree(d, ignore_errors=True)

    def test_models_lists_runtimes_and_configured_providers(self):
        """A cloud provider needs no local process, so discover never sees it."""
        saved_disc, saved_fetch = lmm.discover, frontend.fetch_models
        try:
            lmm.discover = lambda cfg, **kw: [
                {"name": "Ollama", "running": True, "models": ["llama3:8b"]}]
            frontend.fetch_models = lambda prov: ["gpt-4o-mini"]
            out = self._capture(frontend.cmd_models,
                                {"providers": {"openai": {
                                    "api_key": "k", "base_url": "http://x/v1",
                                    "model": "gpt-4o-mini", "kind": "cloud"}}})
            self.assertIn("llama3:8b", out)
            self.assertIn("gpt-4o-mini", out)
        finally:
            lmm.discover, frontend.fetch_models = saved_disc, saved_fetch

    def test_ask_verify_prints_the_backend_that_passed_the_gate(self):
        saved = frontend.route_and_verify
        try:
            frontend.route_and_verify = lambda task, cfg, order: (
                "good", "verified ok", "42")
            out = self._capture(frontend.cmd_ask, "q", None, {}, verify=True)
            self.assertIn("good", out)
            self.assertIn("42", out)
        finally:
            frontend.route_and_verify = saved

    def test_ask_verify_says_so_when_nothing_passes(self):
        saved = frontend.route_and_verify
        try:
            frontend.route_and_verify = lambda task, cfg, order: (
                None, "all replies unfit", None)
            out = self._capture(frontend.cmd_ask, "q", None, {}, verify=True)
            self.assertIn("no backend passed", out)
            self.assertIn("all replies unfit", out)
        finally:
            frontend.route_and_verify = saved

    def test_unknown_provider_names_the_ones_that_exist(self):
        """A typo is not an outage; a dead end should carry the fix."""
        saved = lmm.discover
        try:
            lmm.discover = lambda cfg, **kw: []
            out = self._capture(frontend.cmd_ask, "q", "openia", {})
            self.assertIn("unknown provider 'openia'", out)
            self.assertIn("known providers", out)
        finally:
            lmm.discover = saved


class TestPackaging(unittest.TestCase):
    """An install that cannot start is worse than no install.

    Splitting one file into three broke both installers and the documented
    curl line: they still shipped `lmm.py` alone, so every fresh install died
    on `from backend import *` before printing anything. These derive the file
    list from the repository rather than hardcoding it, so adding a fourth
    module fails here until the installers and the README know about it.
    """

    def setUp(self):
        self.root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.modules = sorted(
            f for f in os.listdir(self.root)
            if f.endswith(".py") and not f.startswith("test"))

    def test_the_repo_has_the_modules_we_think_it_has(self):
        self.assertEqual(self.modules, ["backend.py", "frontend.py", "lmm.py"])

    def _read(self, name):
        with open(os.path.join(self.root, name), encoding="utf-8") as f:
            return f.read()

    def test_installers_ship_every_module(self):
        for installer in ("install.sh", "install.ps1"):
            text = self._read(installer)
            for mod in self.modules:
                self.assertIn(mod, text,
                              "%s does not ship %s" % (installer, mod))

    def test_readme_tells_you_to_fetch_every_module(self):
        readme = self._read("README.md")
        install = readme[readme.index("## Install"):readme.index("## Usage")]
        for mod in self.modules:
            self.assertIn(mod, install,
                          "the install instructions never mention " + mod)

    def test_every_import_resolves_to_the_standard_library(self):
        """"Zero dependencies" is the headline claim, so it gets a test.

        It was only ever checked by a CI step that read `lmm.py`, which after
        the split imports almost nothing — the check would have passed while a
        third-party import sat in `backend.py`. Scanning every module keeps it
        honest, and deriving the local names from the repo means a new module
        is not mistaken for a missing package.
        """
        import ast
        if not hasattr(sys, "stdlib_module_names"):
            self.skipTest("needs Python 3.10+ to introspect the stdlib list")
        local = {m[:-3] for m in self.modules}
        offenders = {}
        for mod in self.modules:
            names = set()
            for node in ast.walk(ast.parse(self._read(mod))):
                if isinstance(node, ast.Import):
                    names.update(a.name.split(".")[0] for a in node.names)
                elif (isinstance(node, ast.ImportFrom) and node.level == 0
                        and node.module):
                    names.add(node.module.split(".")[0])
            extra = sorted(n for n in names
                           if n not in sys.stdlib_module_names and n not in local)
            if extra:
                offenders[mod] = extra
        self.assertEqual(offenders, {}, "non-stdlib imports found")

    def test_no_tracked_file_carries_a_personal_machine_path(self):
        """The pre-push hook hardcoded one contributor's Windows temp
        directory, which made the guard run on exactly one machine on Earth
        and fail everywhere else. Pin the class, not the instance: no
        tracked text file may name a specific user's home directory."""
        import subprocess
        import re
        r = subprocess.run(["git", "ls-files"], cwd=self.root,
                           stdout=subprocess.PIPE, timeout=60)
        pat = re.compile(r"C:[/\\]Users[/\\](?!Public)[^/\\]+"
                         r"|/home/(?!user\b)[a-z][a-z0-9_-]+/"
                         r"|/Users/[a-z][a-z0-9_-]+/")
        offenders = []
        for name in r.stdout.decode().splitlines():
            p = os.path.join(self.root, name)
            try:
                with open(p, encoding="utf-8") as f:
                    text = f.read()
            except (UnicodeDecodeError, OSError):
                continue
            for m in pat.finditer(text):
                offenders.append("%s: %s" % (name, m.group(0)))
        self.assertEqual(offenders, [])

    def test_the_push_gate_proves_the_pushed_revision(self):
        """pre-push must check out and selftest the exact revision being
        pushed — testing lone blobs died with the three-file split, and
        testing the working tree tests the wrong thing. Structural: the hook
        exists, is the only pre-push, and uses a worktree, not blobs."""
        hook = os.path.join(self.root, ".githooks", "pre-push")
        self.assertTrue(os.path.isfile(hook))
        self.assertFalse(os.path.exists(os.path.join(self.root, "hooks")),
                         "a second, unwired hooks/ directory is back")
        with open(hook, encoding="utf-8") as f:
            text = f.read()
        self.assertIn("git worktree add", text)
        self.assertNotIn("git show", text,
                         "the hook is testing lone blobs again")

    def test_nothing_is_defined_with_no_callers(self):
        """A function nobody calls is either dead weight or — twice in this
        repo's history — a SEVERED FEATURE: the code survived, its call site
        did not, and the feature silently stopped existing.

        `setup_tray` lost its caller in a merge and minimize-to-tray simply
        vanished. `resolve_provider_by_model` lost its caller and the hub
        went back to ignoring which model you asked for. `cache_prune` and
        `gui_ask` were plain dead weight. All four were found by running
        this sweep BY HAND, three separate times. Automating the audit is
        the last step of the method it came from: a check performed
        repeatedly by a person is a check that will eventually be skipped.

        Anything genuinely entry-point-like (main, __init__, dunders) is
        exempt; everything else must be reachable from the product itself.
        """
        import ast
        srcs = {}
        for name in self.modules:
            with open(os.path.join(self.root, name), encoding="utf-8") as f:
                srcs[name] = f.read()
        trees = {n: ast.parse(src) for n, src in srcs.items()}

        # Real references only. A regex over source text counts the symbol's
        # own docstring and the comment explaining it, which is how the first
        # two versions of this test passed the setup_tray mutation they
        # existed to catch: unwiring the call left the comment behind, and
        # the comment looked like life support.
        referenced = set()
        for n, tree in trees.items():
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    referenced.add(node.id)
                elif isinstance(node, ast.Attribute):
                    referenced.add(node.attr)

        EXEMPT = {"main"}
        orphans = []
        for name, tree in trees.items():
            for node in tree.body:
                if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    continue
                sym = node.name
                if sym in EXEMPT or sym.startswith("__"):
                    continue
                if sym not in referenced:
                    orphans.append("%s:%s" % (name, sym))
        self.assertEqual(orphans, [],
                         "defined but never called — dead weight, or a "
                         "feature whose call site was severed")

    def test_conditional_skips_decorate_the_class_they_name(self):
        """A decorator sits above a class, so inserting a new class between
        them silently moves it. That happened here: two classes were added
        above `TestOpenAISdkCompat` and inherited its
        `skipUnless(HAS_OPENAI)`. The consequences were invisible locally,
        where the SDK is installed and everything ran — but on any machine
        without it (CI, every contributor) the tray tests were skipped and
        never ran at all, while the SDK tests lost their guard and errored
        with NameError. Only the SDK class may carry that skip.
        """
        import ast
        path = os.path.join(self.root, "tests", "test_lmm.py")
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for dec in node.decorator_list:
                names = {n.id for n in ast.walk(dec)
                         if isinstance(n, ast.Name)}
                if "HAS_OPENAI" not in names:
                    continue
                self.assertIn("Sdk", node.name,
                              "the openai skip drifted onto %s, which has "
                              "nothing to do with the SDK" % node.name)

    def test_no_server_uses_the_default_poll_interval(self):
        """serve_forever's poll_interval decides how long shutdown blocks,
        not how fast the server answers. The stdlib default is 0.5 s, so
        every omission is half a second of latency on stopping — measured
        at 16 s of a 25 s test run, and a visibly sticky Ctrl-C on the hub.
        Pin the class, not the instance: every call must say what it wants.
        """
        import ast
        offenders = []
        for name in self.modules + ["tests/test_lmm.py"]:
            path = os.path.join(self.root, name)
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read())
            called_here = set()
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                if isinstance(fn, ast.Attribute):
                    called_here.add(id(fn))
                    if fn.attr != "serve_forever":
                        continue
                elif getattr(fn, "id", None) != "serve_forever":
                    continue
                kws = {k.arg for k in node.keywords}
                if "poll_interval" not in kws and not node.args:
                    offenders.append("%s:%d (default)" % (name, node.lineno))
            # A bare REFERENCE is the dangerous form: handing
            # `x.serve_forever` to Thread(target=...) means someone else
            # calls it, with every default intact. The first version of this
            # test only inspected calls and therefore passed the mutation it
            # was written to catch — a test that cannot fail is not a test.
            for node in ast.walk(tree):
                if (isinstance(node, ast.Attribute)
                        and node.attr == "serve_forever"
                        and id(node) not in called_here):
                    offenders.append("%s:%d (bare reference)"
                                     % (name, node.lineno))
        self.assertEqual(offenders, [],
                         "serve_forever called with the default poll_interval")

    def test_a_lone_entry_point_explains_itself(self):
        """Copying lmm.py by itself must not end in an import traceback."""
        import subprocess
        d = tempfile.mkdtemp(prefix="lmm-lone-")
        try:
            shutil.copy(os.path.join(self.root, "lmm.py"),
                        os.path.join(d, "lmm.py"))
            r = subprocess.run([sys.executable, os.path.join(d, "lmm.py"),
                                "discover"], cwd=d, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=120)
            err = r.stderr.decode("utf-8", "ignore")
            self.assertEqual(r.returncode, 1, err)
            self.assertNotIn("Traceback", err)
            self.assertIn("backend.py", err)
            self.assertIn("same directory", err)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_the_launcher_tracks_library_upgrades_without_reinstall(self):
        """install.sh claims the launcher "picks up an upgrade to the
        library without being reinstalled itself". Cross-examined for real:
        install, then upgrade ONLY the library copy, and the same launcher
        must report the new version."""
        import subprocess
        if not shutil.which("bash"):
            self.skipTest("bash not available")
        home = tempfile.mkdtemp(prefix="lmm-up-")
        try:
            env = dict(os.environ, HOME=home)
            subprocess.run(["bash", os.path.join(self.root, "install.sh")],
                           env=env, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=120, check=True)
            lib = os.path.join(home, ".local", "share", "lmm", "backend.py")
            with open(lib, encoding="utf-8") as f:
                src = f.read()
            with open(lib, "w", encoding="utf-8") as f:
                f.write(src.replace('VERSION = "', 'VERSION = "9.9.9-', 1))
            launcher = os.path.join(home, ".local", "bin", "lmm")
            v = subprocess.run([launcher, "--version"], env=env,
                               stdout=subprocess.PIPE, timeout=120)
            self.assertIn("9.9.9-", v.stdout.decode(),
                          "the launcher froze a copy instead of tracking "
                          "the library")
        finally:
            shutil.rmtree(home, ignore_errors=True)

    def test_install_sh_produces_a_working_command(self):
        import subprocess
        if not shutil.which("bash"):
            self.skipTest("bash not available")
        home = tempfile.mkdtemp(prefix="lmm-home-")
        try:
            env = dict(os.environ, HOME=home)
            r = subprocess.run(["bash", os.path.join(self.root, "install.sh")],
                               env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=120)
            self.assertEqual(r.returncode, 0,
                             r.stderr.decode("utf-8", "ignore"))
            launcher = os.path.join(home, ".local", "bin", "lmm")
            self.assertTrue(os.path.isfile(launcher), "no launcher installed")
            v = subprocess.run([launcher, "--version"], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               timeout=120)
            out = v.stdout.decode("utf-8", "ignore")
            self.assertEqual(v.returncode, 0,
                             out + v.stderr.decode("utf-8", "ignore"))
            self.assertIn("lmm ", out)
        finally:
            shutil.rmtree(home, ignore_errors=True)


class TestOneOfEach(unittest.TestCase):
    """The merge briefly created two of everything; these pin the collapse.

    Two routers disagreed on exactly the prompts where routing matters
    ("refactor this 500-line module" went to a 3B model), and the second
    observability log had none of the first one's protections. One of each,
    verified — not asserted.
    """

    def test_there_is_exactly_one_name_map(self):
        """NAME_TO_KEY was duplicated inside resolve_ask_targets and
        optimize_ask_order — the second copy's own comment admitted it
        "mirrors" the first. One module constant now; no function may grow
        a private copy again."""
        import ast
        import inspect
        self.assertIn("ollama", lmm.NAME_TO_KEY)
        for fn in (lmm.resolve_ask_targets, lmm.optimize_ask_order):
            tree = ast.parse(inspect.getsource(fn))
            body = next(n for n in tree.body
                        if isinstance(n, ast.FunctionDef))
            dicts = [n for n in ast.walk(body) if isinstance(n, ast.Dict)
                     and any(isinstance(k, ast.Constant)
                             and k.value == "ollama" for k in n.keys)]
            self.assertEqual(dicts, [],
                             "%s regrew a private name map" % fn.__name__)

    def test_there_is_exactly_one_router(self):
        self.assertFalse(hasattr(lmm, "score_and_route"),
                         "a second prompt scorer is back")
        self.assertFalse(hasattr(lmm, "backend_catalog"))

    def test_route_and_verify_orders_by_the_one_router(self):
        """The verify loop must try backends in order_targets order."""
        saved_call, saved_lo = lmm.call_provider, lmm.local_ollama_provider
        tried = []
        try:
            lmm.local_ollama_provider = lambda: None
            def call(prov, prompt, **kw):
                tried.append(prov["model"])
                # long enough to clear verify_reply's under-delivery floor
                # for reasoning tasks (120 chars)
                return {"choices": [{"message": {"content":
                        "The Schrodinger equation governs how the quantum "
                        "state of a physical system evolves over time via "
                        "the Hamiltonian, and it underlies superposition "
                        "and interference throughout quantum mechanics."}}]}
            lmm.call_provider = call
            cfg = {"providers": {
                "a": {"api_key": "x", "base_url": "http://127.0.0.1:1/v1",
                      "model": "m-a", "kind": "local"},
                "b": {"api_key": "x", "base_url": "http://127.0.0.1:2/v1",
                      "model": "m-b", "kind": "local"}}}
            name, reason, reply = lmm.route_and_verify(
                "explain quantum mechanics", cfg, ["b", "a"])
            self.assertEqual(tried, ["m-b"],
                             "ask_order was not honoured by the verify loop")
            self.assertEqual(name, "b")
            self.assertIn("verified ok", reason)
        finally:
            lmm.call_provider = saved_call
            lmm.local_ollama_provider = saved_lo

    def test_trail_events_share_the_protected_log(self):
        """log_hub must land in USAGE_LOG — cap, compaction and lock included.
        hub.log had none of the three; 50k events measured 14.7 MB and grew
        forever, while the protected log held at 3.7 MB under the same load."""
        with temp_state():
            lmm.log_hub({"event": "ask", "provider": "p", "ok": True})
            self.assertTrue(os.path.isfile(lmm.USAGE_LOG))
            events = lmm.read_usage()
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["event"], "ask")

    def test_trail_events_do_not_pollute_cost_totals(self):
        """A trail entry carries a provider but is not a billable call."""
        with temp_state():
            lmm.log_usage({"provider": "openai", "model": "m", "kind": "remote",
                           "in": 100, "out": 200, "usd": 0.01})
            lmm.log_hub({"event": "ask", "provider": "openai", "ok": True,
                         "prompt": "q"})
            lmm.log_hub({"event": "ask_attempt", "provider": "openai",
                         "ok": True, "latency_ms": 5})
            st = lmm.hub_cost_stats()
            self.assertEqual(st["calls"], 1,
                             "trail events were counted as billable calls")
            self.assertAlmostEqual(st["measured"], 0.01)

    def test_measure_performance_joins_metering_and_failures(self):
        """One writer per fact: a success IS the metering event meter_call
        wrote (cache=="miss"); only failures get an ask_attempt entry, since
        a failure has nothing to meter. The reader joins the two."""
        with temp_state():
            lmm.log_usage({"provider": "p", "model": "m", "kind": "local",
                           "in": 10, "out": 20, "usd": 0.0,
                           "cache": "miss", "ms": 10})
            lmm.log_hub({"event": "ask_attempt", "provider": "p", "ok": False,
                         "latency_ms": 30})
            lmm.log_usage({"provider": "p", "model": "m", "kind": "local",
                           "in": 0, "out": 0, "usd": 0.0,
                           "cache": "exact", "saved_usd": 0.01})
            st = lmm.measure_performance()
            self.assertEqual(st["p"]["ok"], 1,
                             "the metered call was not counted as a success")
            self.assertEqual(st["p"]["fail"], 1)
            self.assertEqual(st["p"]["avg_ms"], 20)
            # the cache hit is an answer, not an attempt — neither ok nor fail
            self.assertEqual(st["p"]["ok"] + st["p"]["fail"], 2)

    def test_the_trail_is_bounded(self):
        """The reason the second log had to die: unbounded growth."""
        with temp_state():
            saved = lmm.USAGE_MAX_BYTES
            try:
                lmm.USAGE_MAX_BYTES = 20_000
                for i in range(2000):
                    lmm.log_hub({"event": "ask", "provider": "p", "ok": True,
                                 "prompt": "x" * 40})
                size = os.path.getsize(lmm.USAGE_LOG)
                self.assertLess(size, 60_000,
                                "trail writes bypass the byte cap")
            finally:
                lmm.USAGE_MAX_BYTES = saved

    def test_doctor_probes_each_configured_provider(self):
        """hub-status's one unique capability, folded into doctor."""
        import io
        import contextlib
        stub = StubBackend()
        try:
            cfg = {"providers": {"stub": {
                "api_key": "k", "base_url": stub.base_url,
                "model": "m", "kind": "local"}},
                "ask_order": ["stub"]}
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                try:
                    frontend.cmd_doctor(cfg)
                except SystemExit:
                    pass
            out = buf.getvalue()
            self.assertIn("provider 'stub' answers", out)
        finally:
            stub.stop()


class TestClaimsSurviveElenchus(unittest.TestCase):
    """The product's strongest claims, cross-examined the Socratic way: take
    the claim, derive a consequence, test the consequence for real. Each test
    here began as a refutation — the claim was FALSE when first examined —
    and stays as the proof that it is now true.

    Claim 1: "move secrets to environment variables; lmm reads them at call
    time" (printed by `lmm secrets`, documented in the README). Refuted:
    nothing expanded $VARS — the literal string "$OPENAI_API_KEY" went out
    as the bearer token.
    Claim 2: "every call lmm makes is metered". Refuted twice: `ask
    --verify` and `lmm bench` both called providers directly and spent off
    the books.
    """

    class _AuthStub(object):
        """Tiny server that records the Authorization header it receives."""

        def __init__(self):
            import http.server
            import socketserver
            import threading
            self.auth = []
            auth = self.auth

            class H(http.server.BaseHTTPRequestHandler):
                def do_POST(self):
                    auth.append(self.headers.get("Authorization"))
                    body = json.dumps({"choices": [{"message": {
                        "content": "The Schrodinger equation governs how the "
                                   "quantum state of a physical system evolves "
                                   "over time via the Hamiltonian, and it "
                                   "underlies superposition and interference "
                                   "throughout quantum mechanics."}}],
                        "usage": {"prompt_tokens": 10,
                                  "completion_tokens": 40}}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def log_message(self, *a):
                    pass

            class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
                daemon_threads = True

            self.httpd = S(("127.0.0.1", 0), H)
            self.port = self.httpd.server_address[1]
            threading.Thread(
                target=lambda: self.httpd.serve_forever(poll_interval=0.01),
                daemon=True).start()

        @property
        def base_url(self):
            return "http://127.0.0.1:%d/v1" % self.port

        def stop(self):
            self.httpd.shutdown()
            self.httpd.server_close()

    def test_a_dollar_api_key_sends_the_env_value_not_the_literal(self):
        stub = self._AuthStub()
        os.environ["LMM_TEST_KEY"] = "sk-real-value-from-env"
        try:
            cfg = {"providers": {"p": {
                "api_key": "$LMM_TEST_KEY", "base_url": stub.base_url,
                "model": "m", "kind": "cloud"}}}
            prov = lmm.merged_providers(cfg)["p"]
            lmm.call_provider(prov, "hi")
            self.assertEqual(stub.auth[-1], "Bearer sk-real-value-from-env",
                             "the documented $VAR pattern sent: %r"
                             % stub.auth[-1])
        finally:
            del os.environ["LMM_TEST_KEY"]
            stub.stop()

    def test_an_unset_env_var_is_named_by_doctor(self):
        """A missing variable must be reported by NAME — otherwise the only
        witness is a 401 from the provider."""
        import io
        import contextlib
        cfg = {"providers": {"p": {
            "api_key": "$LMM_UNSET_VAR_FOR_TEST",
            "base_url": "http://127.0.0.1:9/v1", "model": "m",
            "kind": "cloud"}}}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                frontend.cmd_doctor(cfg)
            except SystemExit:
                pass
        out = buf.getvalue()
        self.assertIn("LMM_UNSET_VAR_FOR_TEST", out)
        self.assertIn("api_key env var set", out)

    def test_verify_calls_are_metered_accepted_and_rejected(self):
        stub = self._AuthStub()
        try:
            with temp_state():
                cfg = {"providers": {"good": {
                    "api_key": "k", "base_url": stub.base_url,
                    "model": "m", "kind": "local"}},
                    "ask_order": ["good"]}
                name, reason, reply = lmm.route_and_verify(
                    "explain quantum mechanics", cfg)
                self.assertEqual(name, "good")
                metered = [e for e in lmm.read_usage()
                           if not e.get("event")
                           and e.get("source") == "verify"]
                self.assertEqual(len(metered), 1,
                                 "an accepted --verify call left no meter event")
                self.assertTrue(metered[0].get("accepted"))
                self.assertEqual(metered[0].get("out"), 40)
        finally:
            stub.stop()

    def test_a_rejected_verify_reply_is_still_billed(self):
        """The gate rejecting an answer does not un-spend the tokens."""
        saved = lmm.call_provider
        try:
            with temp_state():
                lmm.call_provider = lambda prov, task, **kw: {
                    "choices": [{"message": {"content": "Short."}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2}}
                cfg = {"providers": {"p": {
                    "api_key": "k", "base_url": "http://127.0.0.1:9/v1",
                    "model": "m", "kind": "local"}}, "ask_order": ["p"]}
                name, reason, reply = lmm.route_and_verify(
                    "explain quantum mechanics", cfg, max_tries=1)
                self.assertIsNone(name)
                metered = [e for e in lmm.read_usage()
                           if not e.get("event")
                           and e.get("source") == "verify"]
                self.assertEqual(len(metered), 1,
                                 "a rejected reply was spent off the books")
                self.assertFalse(metered[0].get("accepted"))
        finally:
            lmm.call_provider = saved

    def test_bench_meters_every_run_including_the_warmup(self):
        stub = StubBackend()
        try:
            with temp_state():
                import io
                import contextlib
                cfg = {"providers": {"stub": {
                    "api_key": "k", "base_url": stub.base_url,
                    "model": "m", "kind": "local"}},
                    "ask_order": ["stub"]}
                with contextlib.redirect_stdout(io.StringIO()):
                    frontend.cmd_bench(cfg, "stub", 2, "hello", 16)
                metered = [e for e in lmm.read_usage()
                           if not e.get("event")
                           and e.get("source") == "bench"]
                self.assertEqual(len(metered), 3,
                                 "bench runs (2) + warm-up (1) must all be "
                                 "billed, got %d" % len(metered))
        finally:
            stub.stop()


class TestTelemetrySurvivesElenchus(unittest.TestCase):
    """Two sworn statements from the code itself, cross-examined.

    "Telemetry must not be able to break the call it is measuring"
    (log_usage). First exhibit was thrown out: a chmod-based read-only
    directory proves nothing under root, which ignores permission bits —
    the "read-only" dir quietly accepted all three files. This version
    injects EROFS at the open() itself, so it binds for any uid.

    "Those tokens were generated and billed whether or not anyone read
    them" (hub_stream's finally). Proven by hanging up mid-stream and
    finding the spend on the books, flagged partial.
    """

    def setUp(self):
        self.stub = StubBackend()
        self.cfg = {"providers": {"stub": {
            "api_key": "k", "base_url": self.stub.base_url,
            "model": "m", "kind": "local"}}, "ask_order": ["stub"]}

    def tearDown(self):
        self.stub.stop()

    def test_a_read_only_disk_never_breaks_the_call(self):
        import builtins
        import errno
        with temp_state():
            targets = lmm.resolve_ask_targets(self.cfg, "x", None)
            real_open = builtins.open

            def refusing(path, *a, **kw):
                mode = a[0] if a else kw.get("mode", "r")
                if isinstance(path, str) and path.startswith(lmm.LMM_DIR) \
                        and any(m in mode for m in "wax+"):
                    raise OSError(errno.EROFS, "Read-only file system", path)
                return real_open(path, *a, **kw)

            builtins.open = refusing
            try:
                res, _ = lmm.hub_complete(self.cfg, "what is 2+2", targets,
                                          {"cache": True, "source": "ask"})
                self.assertIn("choices", res,
                              "a full disk/read-only disk broke the answer")
                b = lmm.CircuitBreaker(persist=True)
                for _ in range(3):
                    b.record_failure("dead")     # must not raise
                self.assertFalse(b.available("dead"),
                                 "the in-memory half stopped working too")
                frames = list(lmm.hub_stream(
                    self.cfg, [{"role": "user", "content": "hi"}], targets,
                    {"cache": True, "source": "chat"}))
                self.assertTrue(any(b"data:" in f for f in frames))
            finally:
                builtins.open = real_open

    def test_an_abandoned_stream_is_still_billed(self):
        with temp_state():
            targets = lmm.resolve_ask_targets(self.cfg, "hi", None)
            gen = lmm.hub_stream(self.cfg,
                                 [{"role": "user", "content": "hi"}],
                                 targets, {"cache": False, "source": "hub"})
            next(gen)                    # the client reads one frame...
            gen.close()                  # ...and hangs up
            evs = [e for e in lmm.read_usage() if not e.get("event")]
            self.assertEqual(len(evs), 1,
                             "tokens were generated but never billed")
            self.assertTrue(evs[0].get("partial"),
                            "an abandoned stream must be flagged partial")


class TestCompatSurvivesElenchus(unittest.TestCase):
    """Two acquittals with the evidence entered.

    "Config is backward compatible across every version" — examined against
    the REAL v1.0.0 example config, resurrected from the git tag and pinned
    here verbatim, not against a synthetic approximation of it.

    "A self-contained HTML dashboard" — the generated page must reference no
    external resource; a dashboard that phones home is neither self-contained
    nor private.
    """

    V1_CONFIG = {
        "pricing": {"my-model": {"in": 1.0, "out": 2.0, "cw": 1.0, "cr": 0.1}},
        "route": {"private": ["secret", "\u793e\u5185", "private", "local",
                              "offline"],
                  "heavy": ["code", "\u8a2d\u8a08", "refactor", "debug",
                            "architecture"]},
        "extra_runtimes": [
            {"name": "vLLM", "type": "local", "paid": False,
             "procs": ["vllm", "vllm.entrypoints"],
             "installed_paths": ["~/.vllm"],
             "endpoint": "http://localhost:8000/v1",
             "models_cmd": "curl -s http://localhost:8000/v1/models"},
            {"name": "My Remote Agent", "type": "remote", "paid": True,
             "procs": ["myagent"], "installed_paths": ["~/myagent"],
             "endpoint": "https://api.myagent.example"}]}

    def test_the_v1_example_config_still_drives_todays_code(self):
        cfg = dict(self.V1_CONFIG)
        self.assertIn("my-model", lmm.merged_pricing(cfg))
        self.assertIn("secret", lmm.merged_route(cfg)["private"])
        score, _ = lmm.prompt_strength(cfg, "hello")
        self.assertTrue(0.0 <= score <= 1.0)
        self.assertEqual(lmm.merged_providers(cfg), {},
                         "v1 had no providers key; today must not invent one")
        lmm.resolve_ask_targets(cfg, "hello", None)   # must not raise
        names = [r["name"] for r in cfg["extra_runtimes"]]
        self.assertEqual(names, ["vLLM", "My Remote Agent"])

    def test_the_dashboard_is_self_contained(self):
        import re
        with temp_state():
            html_text = frontend.build_dash({})
        ext = re.findall(r'(?:src|href)\s*=\s*["\'](https?://[^"\']+)',
                         html_text)
        self.assertEqual(ext, [],
                         "the dashboard references external resources")


class TestVarietySurvivesElenchus(unittest.TestCase):
    """Claim: cache.max_temp — "above this the caller wants variety, not a
    cache". Half-refuted: the store side honoured it (a hot answer was never
    frozen), but the LOOKUP side did not, so a hot repeat of a cached
    question was served the frozen answer — the exact thing the caller
    asked not to get. Both halves now hold, and both are pinned here.

    Also entered as acquittal evidence this sitting: chat keeps history
    across turns, and the semantic tier's embedder structurally cannot pay
    a cloud provider.
    """

    def setUp(self):
        self.stub = StubBackend()
        self.cfg = {"providers": {"stub": {
            "api_key": "k", "base_url": self.stub.base_url,
            "model": "m", "kind": "local"}}, "ask_order": ["stub"]}

    def tearDown(self):
        self.stub.stop()

    def _ask(self, prompt, temp=None):
        targets = lmm.resolve_ask_targets(self.cfg, prompt, None)
        opts = {"cache": True, "source": "ask"}
        if temp is not None:
            opts["extra"] = {"temperature": temp}
        return lmm.hub_complete(self.cfg, prompt, targets, opts)

    def test_a_hot_answer_is_never_frozen(self):
        with temp_state():
            self._ask("tell me a story", temp=1.5)
            self.assertEqual(len(lmm.cache_entries(lmm.merged_cache({}))), 0)

    def test_a_hot_repeat_is_not_served_the_frozen_answer(self):
        with temp_state():
            self._ask("what is 2+2")                       # cold: cached
            res, trace = self._ask("what is 2+2", temp=1.5)
            self.assertFalse(any("hit" in t for t in trace),
                             "variety was requested and the cache answered "
                             "anyway: %r" % trace)
            self.assertTrue(any("variety" in t for t in trace))

    def test_a_cold_repeat_still_hits(self):
        with temp_state():
            self._ask("what is 2+2")
            res, trace = self._ask("what is 2+2")
            self.assertTrue(any("[cache]" in t and "hit" in t for t in trace))

    def test_chat_carries_history_between_turns(self):
        """README: chat "keeps conversation history". Simulated 2-turn
        session; the second request must carry turn 1 and its reply."""
        import builtins
        import io
        import contextlib
        captured = []

        def fake_stream(cfg, messages, targets, opts=None):
            captured.append([m["content"] for m in messages])
            yield (b'data: {"choices":[{"delta":{"content":"reply%d"}}]}\n\n'
                   % len(captured))
            yield b"data: [DONE]\n\n"

        saved_hs = frontend.hub_stream
        saved_rat = frontend.resolve_ask_targets
        saved_in = builtins.input
        answers = iter(["first question", "second question", "exit"])
        try:
            frontend.hub_stream = fake_stream
            frontend.resolve_ask_targets = lambda cfg, line, prov: [
                ("p", {"model": "m", "kind": "local", "base_url": "http://x"})]
            builtins.input = lambda *a: next(answers)
            with contextlib.redirect_stdout(io.StringIO()):
                frontend.cmd_chat(None, {})
        finally:
            frontend.hub_stream = saved_hs
            frontend.resolve_ask_targets = saved_rat
            builtins.input = saved_in
        self.assertEqual(captured[1],
                         ["first question", "reply1", "second question"],
                         "turn 2 did not carry turn 1's exchange")

    def test_the_embedder_cannot_pay_anyone(self):
        """"the semantic tier uses local embeddings, never a paid one" —
        every URL the embedder touches must be localhost."""
        urls = []
        saved = lmm.http_post_json
        try:
            lmm.http_post_json = lambda url, *a, **kw: (
                urls.append(url) or {"error": "x"})
            lmm.embed_text("hello", "nomic-embed-text")
        finally:
            lmm.http_post_json = saved
        self.assertTrue(urls, "the embedder made no call at all")
        for u in urls:
            self.assertTrue(u.startswith("http://localhost:11434"),
                            "the embedder left the machine: %s" % u)


class TestRetryAfterSurvivesElenchus(unittest.TestCase):
    """Claim: "a 429's Retry-After is honoured but capped, so a hostile
    header cannot park the hub." Acquitted — these are the acquittal's
    evidence: the parser was tested, but nothing proved the retry LOOP
    actually waits the header's time, nor that the cap really binds.
    """

    def _run(self, retry_after, cap_ms=8000):
        sleeps = []
        calls = []
        saved = lmm.call_provider
        try:
            def fake(prov, prompt, temperature=0.7, extra=None, **kw):
                calls.append(1)
                if len(calls) == 1:
                    return {"error": "429", "retriable": True,
                            "retry_after": retry_after, "status": 429}
                return {"choices": [{"message": {"content": "ok"}}]}
            lmm.call_provider = fake
            res = lmm.call_with_retry(
                {"model": "m", "base_url": "http://x/v1"}, "hi",
                retry={"attempts": 3, "base_ms": 250, "cap_ms": cap_ms},
                sleep=sleeps.append)
            return res, sleeps
        finally:
            lmm.call_provider = saved

    def test_the_loop_waits_what_the_server_asked(self):
        res, sleeps = self._run(retry_after=2.0)
        self.assertEqual(sleeps, [2.0],
                         "Retry-After was parsed but not obeyed by the loop")
        self.assertIn("choices", res)

    def test_a_hostile_header_cannot_park_the_hub(self):
        res, sleeps = self._run(retry_after=86400.0, cap_ms=8000)
        self.assertEqual(sleeps, [8.0],
                         "a hostile Retry-After exceeded the backoff ceiling")


class TestPrivacyPinSurvivesElenchus(unittest.TestCase):
    """Claim on trial: "route.private pins a prompt to local providers."
    Cross-examined, the pin yielded three ways: with ask_order set the check
    was never consulted (cloud-first order sent the secret to the cloud with
    a local model sitting right there); with no ask_order the sort kept
    cloud as a FALLBACK, so a failing local leaked; and with no local at all
    it warned — in a trace printed after the request returned — and sent
    anyway. A pin that yields is not a pin.
    """

    PRIVATE = "summarise this confidential internal memo about the merger"

    def setUp(self):
        self.stub = StubBackend()          # plays the cloud
        self.sent = []
        self._real = lmm.call_with_retry

        def spy(prov, *a, **kw):
            self.sent.append(prov.get("kind"))
            return self._real(prov, *a, **kw)

        lmm.call_with_retry = spy

    def tearDown(self):
        lmm.call_with_retry = self._real
        self.stub.stop()

    def _cfg(self, order, providers):
        return {"providers": providers, "ask_order": order,
                "retry": {"attempts": 1},
                "route": {"private": ["confidential"]}}

    def test_cloud_first_ask_order_does_not_outrank_privacy(self):
        cfg = self._cfg(["cloud", "loc"], {
            "cloud": {"api_key": "k", "base_url": self.stub.base_url,
                      "model": "m", "kind": "cloud"},
            "loc": {"api_key": "k", "base_url": "http://127.0.0.1:9/v1",
                    "model": "m", "kind": "local"}})
        with temp_state():
            targets = lmm.resolve_ask_targets(cfg, self.PRIVATE, None)
            lmm.hub_complete(cfg, self.PRIVATE, targets,
                             {"cache": False, "source": "ask"})
        self.assertNotIn("cloud", self.sent,
                         "ask_order outranked the privacy pin")

    def test_a_failing_local_does_not_fall_back_to_the_cloud(self):
        """The subtlest leak: local listed first, local dead, cloud alive.
        The old sort kept the cloud in the list as a fallback."""
        cfg = self._cfg([], {
            "loc": {"api_key": "k", "base_url": "http://127.0.0.1:9/v1",
                    "model": "m", "kind": "local"},
            "cloud": {"api_key": "k", "base_url": self.stub.base_url,
                      "model": "m", "kind": "cloud"}})
        with temp_state():
            targets = lmm.resolve_ask_targets(cfg, self.PRIVATE, None)
            res, _ = lmm.hub_complete(cfg, self.PRIVATE, targets,
                                      {"cache": False, "source": "ask"})
        self.assertNotIn("cloud", self.sent,
                         "a failing local provider leaked the prompt to the cloud")
        self.assertIn("error", res)

    def test_no_local_at_all_is_refused_on_every_path(self):
        cfg = self._cfg(["cloud"], {
            "cloud": {"api_key": "k", "base_url": self.stub.base_url,
                      "model": "m", "kind": "cloud"}})
        with temp_state():
            targets = lmm.resolve_ask_targets(cfg, self.PRIVATE, None)
            res, _ = lmm.hub_complete(cfg, self.PRIVATE, targets,
                                      {"cache": False, "source": "ask"})
            self.assertIn("refusing", res.get("error", ""))
            frames = b"".join(lmm.hub_stream(
                cfg, [{"role": "user", "content": self.PRIVATE}], targets,
                {"cache": False, "source": "chat"}))
            self.assertIn(b"refusing", frames)
            name, reason, reply = lmm.route_and_verify(self.PRIVATE, cfg)
            self.assertIsNone(name)
            self.assertIn("refusing", reason)
        self.assertEqual(self.sent, [],
                         "a private prompt reached a provider on some path")
        self.assertEqual(self.stub.seen, [],
                         "the cloud stub actually received the secret")

    def test_a_public_prompt_is_untouched_by_the_pin(self):
        cfg = self._cfg(["cloud"], {
            "cloud": {"api_key": "k", "base_url": self.stub.base_url,
                      "model": "m", "kind": "cloud"}})
        with temp_state():
            targets = lmm.resolve_ask_targets(cfg, "what is 2+2", None)
            res, _ = lmm.hub_complete(cfg, "what is 2+2", targets,
                                      {"cache": False, "source": "ask"})
        self.assertNotIn("error", res)
        self.assertEqual(self.sent, ["cloud"])


class TestBreakerSurvivesElenchus(unittest.TestCase):
    """Claim: "a dead backend stops charging every request its full timeout."
    Cross-examined, it failed twice: `lmm ask` never engaged the breaker at
    all (the comment beside HUB_BREAKER even said ask "makes a fresh one per
    process" — it made none), and the state died with the process, so a CLI
    user re-paid the timeout on every single run.
    """

    def _cfg(self, stub):
        return {"providers": {
            "dead": {"api_key": "k", "base_url": "http://127.0.0.1:9/v1",
                     "model": "m", "kind": "local"},
            "good": {"api_key": "k", "base_url": stub.base_url,
                     "model": "m", "kind": "local"}},
            "ask_order": ["dead", "good"],
            "retry": {"attempts": 1},
            "breaker": {"enabled": True, "threshold": 3, "cooldown_s": 300}}

    def test_ask_stops_paying_for_a_dead_provider(self):
        """5 asks, dead provider first in ask_order: it may be attempted at
        most `threshold` times before the circuit opens and it is skipped."""
        stub = StubBackend()
        attempts = []
        real = lmm.call_with_retry

        def counting(prov, *a, **kw):
            attempts.append(prov["base_url"])
            return real(prov, *a, **kw)

        lmm.call_with_retry = counting
        try:
            with temp_state():
                import io
                import contextlib
                cfg = self._cfg(stub)
                for _ in range(5):
                    with contextlib.redirect_stdout(io.StringIO()):
                        frontend.cmd_ask("hi", None, cfg, no_cache=True)
                dead_tries = sum(1 for u in attempts if ":9/" in u)
                self.assertEqual(dead_tries, 3,
                                 "the dead provider was attempted %d times "
                                 "across 5 asks; the breaker never opened"
                                 % dead_tries)
        finally:
            lmm.call_with_retry = real
            stub.stop()

    def test_the_open_circuit_survives_the_process(self):
        """The half that made the claim false for CLI users: a second
        process must inherit the first one's open circuit."""
        import subprocess
        home = tempfile.mkdtemp(prefix="lmm-breaker-")
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ, HOME=home)
        script1 = (
            "import sys; sys.path.insert(0, %r); import backend\n"
            "for _ in range(3): backend.HUB_BREAKER.record_failure('dead')\n"
            "print(backend.HUB_BREAKER.available('dead'))" % root)
        script2 = (
            "import sys; sys.path.insert(0, %r); import backend\n"
            "print(backend.HUB_BREAKER.available('dead'))" % root)
        try:
            r1 = subprocess.run([sys.executable, "-c", script1], env=env,
                                capture_output=True, text=True, timeout=120)
            self.assertEqual(r1.stdout.strip(), "False", r1.stderr)
            r2 = subprocess.run([sys.executable, "-c", script2], env=env,
                                capture_output=True, text=True, timeout=120)
            self.assertEqual(r2.stdout.strip(), "False",
                             "a fresh process forgot the open circuit: "
                             + r2.stderr)
        finally:
            shutil.rmtree(home, ignore_errors=True)

    def test_healthy_traffic_writes_no_state_file(self):
        """record_success on a name with no history must not touch disk —
        otherwise every successful call becomes a disk write."""
        with temp_state():
            lmm.HUB_BREAKER.record_success("fine")
            self.assertFalse(
                os.path.exists(os.path.join(lmm.LMM_DIR, "breaker.json")),
                "a healthy call wrote breaker state")

    def test_recovery_is_forgettable(self):
        with temp_state():
            for _ in range(3):
                lmm.HUB_BREAKER.record_failure("dead")
            self.assertFalse(lmm.HUB_BREAKER.available("dead"))
            lmm.HUB_BREAKER.reset()
            self.assertTrue(lmm.HUB_BREAKER.available("dead"))
            self.assertFalse(
                os.path.exists(os.path.join(lmm.LMM_DIR, "breaker.json")))


class TestHubDoesNotLeak(unittest.TestCase):
    """The hub is a long-lived proxy, and nothing had ever measured what it
    accumulates. Under sustained load its RSS, file descriptors and thread
    count are flat, but the Python object graph grows ~2 objects per request
    and survives gc.collect() — which looks exactly like a slow leak.

    It is not lmm's. The identical growth appears with lmm removed from the
    loop entirely (a bare http.client talking to the same in-process stub),
    so the residual belongs to the test harness's own HTTP server. Measuring
    lmm against an absolute number would therefore pin stdlib behaviour and
    break on an interpreter upgrade; this compares lmm's path to that
    baseline instead, which is sharp against a real leak in the hub and
    indifferent to what the stdlib does around it.
    """

    def _objects(self):
        import gc
        gc.collect()
        return len(gc.get_objects())

    def test_the_hub_path_adds_nothing_over_a_bare_client(self):
        import http.client
        stub = StubBackend()
        try:
            with temp_state():
                cfg = {"providers": {"stub": {
                    "api_key": "k", "base_url": stub.base_url,
                    "model": "m", "kind": "local"}}, "ask_order": ["stub"]}
                targets = lmm.resolve_ask_targets(cfg, "x", None)

                def through_lmm(i):
                    lmm.hub_complete(cfg, "q%d" % i, targets,
                                     {"cache": False, "source": "hub"})

                def bare(i):
                    c = http.client.HTTPConnection(
                        "127.0.0.1", stub.port, timeout=30)
                    c.request("POST", "/v1/chat/completions",
                              json.dumps({"model": "m", "messages": [
                                  {"role": "user", "content": "q%d" % i}]}),
                              {"Content-Type": "application/json"})
                    c.getresponse().read()
                    c.close()

                # N is set from measurement, not taste: at N=60 the noise
                # swamped a real injected leak of one list per request, and
                # the first version of this test passed the very mutation it
                # was written to catch. At N=300 lmm's excess over the
                # baseline measures +0.000/request, so a margin of 0.5
                # convicts a 1-object leak while leaving room for jitter.
                N = 300
                for i in range(40):          # warm both paths first
                    through_lmm(i)
                    bare(i)

                a = self._objects()
                for i in range(N):
                    through_lmm(1000 + i)
                lmm_growth = (self._objects() - a) / float(N)

                b = self._objects()
                for i in range(N):
                    bare(2000 + i)
                base_growth = (self._objects() - b) / float(N)

                # Generous margin: this must catch a real per-request leak
                # (a retained payload, an unbounded list), not normal jitter.
                self.assertLess(
                    lmm_growth, base_growth + 0.5,
                    "the hub retains %.2f objects per request against a "
                    "baseline of %.2f — something on the request path is "
                    "holding on" % (lmm_growth, base_growth))
        finally:
            stub.stop()

    def test_sustained_load_does_not_leak_threads_or_descriptors(self):
        """RSS is noisy, but threads and file descriptors are not: a hub that
        loses one of either per request dies in production."""
        stub = StubBackend()
        try:
            with temp_state():
                cfg = {"providers": {"stub": {
                    "api_key": "k", "base_url": stub.base_url,
                    "model": "m", "kind": "local"}}, "ask_order": ["stub"]}
                targets = lmm.resolve_ask_targets(cfg, "x", None)
                import threading

                def fds():
                    try:
                        return len(os.listdir("/proc/self/fd"))
                    except OSError:
                        self.skipTest("no /proc on this platform")

                for i in range(20):
                    lmm.hub_complete(cfg, "warm%d" % i, targets,
                                     {"cache": False, "source": "hub"})
                fd0, th0 = fds(), threading.active_count()
                for i in range(120):
                    lmm.hub_complete(cfg, "q%d" % i, targets,
                                     {"cache": False, "source": "hub"})
                self.assertLessEqual(fds(), fd0 + 2,
                                     "descriptors grew under sustained load")
                self.assertLessEqual(threading.active_count(), th0 + 2,
                                     "threads grew under sustained load")
        finally:
            stub.stop()


class TestClosedLoop(unittest.TestCase):
    """The product's core claim: routing outcomes are measured, and the
    measurements are readable. The merge silently broke this — the readers
    (`lmm stats`, `priority --optimize`) survived while their writer did not,
    so the closed loop was open and nothing noticed. These are round-trips
    through the real path, not unit checks on either half.
    """

    def test_a_real_ask_is_visible_to_measure_performance(self):
        stub = StubBackend()
        try:
            with temp_state():
                cfg = {"providers": {"stub": {
                    "api_key": "k", "base_url": stub.base_url,
                    "model": "m", "kind": "local"}},
                    "ask_order": ["stub"]}
                targets = lmm.resolve_ask_targets(cfg, "hello", None)
                res, trace = lmm.hub_complete(cfg, "hello", targets,
                                              {"cache": False, "source": "ask"})
                self.assertNotIn("error", res)
                st = lmm.measure_performance()
                self.assertEqual(st["stub"]["ok"], 1,
                                 "a successful ask left no measurable trace")
        finally:
            stub.stop()

    def test_a_failed_attempt_is_visible_too(self):
        with temp_state():
            cfg = {"providers": {"dead": {
                "api_key": "k", "base_url": "http://127.0.0.1:9/v1",
                "model": "m", "kind": "local"}},
                "ask_order": ["dead"],
                "retry": {"attempts": 1}}
            targets = lmm.resolve_ask_targets(cfg, "hello", None)
            res, trace = lmm.hub_complete(cfg, "hello", targets,
                                          {"cache": False, "source": "ask"})
            self.assertIn("error", res)
            st = lmm.measure_performance()
            self.assertEqual(st["dead"]["fail"], 1,
                             "a failed attempt left no measurable trace")

    def test_chat_turns_are_metered(self):
        """chat used to bypass metering entirely — its spend was invisible
        to `lmm cost`, falsifying the "bill you can actually see" claim."""
        stub = StubBackend()
        try:
            with temp_state():
                cfg = {"providers": {"stub": {
                    "api_key": "k", "base_url": stub.base_url,
                    "model": "m", "kind": "local"}},
                    "ask_order": ["stub"]}
                targets = lmm.resolve_ask_targets(cfg, "hi", None)
                frames = list(lmm.hub_stream(
                    cfg, [{"role": "user", "content": "hi"}], targets,
                    {"source": "chat", "cache": False}))
                self.assertTrue(frames)
                metered = [e for e in lmm.read_usage()
                           if not e.get("event") and e.get("provider") == "stub"]
                self.assertEqual(len(metered), 1,
                                 "a chat turn produced no metering event")
                self.assertEqual(metered[0].get("source"), "chat")
        finally:
            stub.stop()


class TestOneGrader(unittest.TestCase):
    """verify_reply is a gate over verify_answer — the same grader the
    cascade reads as a score. Two independent graders disagreed exactly the
    way the two routers did: the cascade accepted script-fused hallucinations
    that only the gate could see.
    """

    def test_cascade_scores_a_fused_script_hallucination_low(self):
        score, why = lmm.verify_answer(
            "explain quantum mechanics",
            "The wavefunction propag\u30ec\u30fc\u30b7\u30e7\u30f3 describes "
            "everything about the system and its future evolution over time.")
        self.assertLess(score, 0.75,
                        "the cascade would accept a garbled hallucination")
        self.assertTrue(any("script-fused" in w for w in why), why)

    def test_normal_japanese_is_not_taxed(self):
        """'Python\u30b3\u30fc\u30c9' and 'API\u30ad\u30fc' are ordinary
        Japanese technical writing. The old gate rejected every reply that
        fused latin to kana, which taxed correct Japanese answers; the check
        only means "hallucination" when the conversation is not Japanese."""
        score, why = lmm.verify_answer(
            "\u3053\u306ePython\u30b3\u30fc\u30c9\u306e\u30d0\u30b0\u3092"
            "\u76f4\u3057\u3066",
            "\u3053\u306e\u30d0\u30b0\u306fAPI\u30ad\u30fc\u306e\u691c"
            "\u8a3c\u6f0f\u308c\u3067\u3059\u3002`validate()`\u3092\u547c"
            "\u3093\u3067\u304f\u3060\u3055\u3044\u3002")
        self.assertFalse(any("script-fused" in w for w in why),
                         "normal Japanese was flagged as hallucination: %s" % why)

    def test_verify_reply_is_the_gate_over_verify_answer(self):
        cases = [
            "Short.",
            "The Schrodinger equation governs how the quantum state of a "
            "physical system evolves over time via the Hamiltonian, and it "
            "underlies superposition and interference throughout physics.",
            "I cannot help with that request.",
        ]
        for reply in cases:
            score, _ = lmm.verify_answer("explain quantum mechanics", reply)
            ok, _ = lmm.verify_reply("explain quantum mechanics", reply)
            self.assertEqual(ok, score >= lmm.VERIFY_GATE,
                             "gate and grader disagree on: %r" % reply)

    def test_the_gate_still_catches_the_selftest_canary(self):
        ok, reason = lmm.verify_reply(
            "explain quantum mechanics",
            "The wavefunction propag\u30ec\u30fc\u30b7\u30e7\u30f3 "
            "describes everything.")
        self.assertFalse(ok)


class TestSelftestGate(unittest.TestCase):
    """`lmm selftest --guard` is what CI and the pre-push hook actually run."""

    def _run(self, *args):
        import subprocess
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # NO_SUITE, or each of these spawns a selftest that runs the whole
        # suite again: three tests x 8 s turned a 7.7 s run into 33.7 s,
        # undoing the acceleration round two commits earlier. These tests
        # are about the gate's OWN behaviour — its banner, its exit code,
        # its doctor check — not about the suite it carries. That the gate
        # really runs the suite is proven where it matters: by guard.sh in
        # pre-commit, pre-push and CI, on every single run.
        env = dict(os.environ, LMM_SELFTEST_SKIP_LIVE="1",
                   LMM_SELFTEST_NO_SUITE="1")
        return subprocess.run([sys.executable, os.path.join(root, "lmm.py"),
                               "selftest"] + list(args), cwd=root, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=300)

    def test_guard_mode_exits_zero_and_stays_quiet(self):
        r = self._run("--guard")
        out = r.stdout.decode("utf-8", "ignore")
        self.assertEqual(r.returncode, 0, out + r.stderr.decode("utf-8", "ignore"))
        self.assertEqual(out.strip(), "",
                         "a green guard run has nothing to say:\n" + out)

    def test_verbose_mode_reports_every_check(self):
        r = self._run()
        out = r.stdout.decode("utf-8", "ignore")
        self.assertEqual(r.returncode, 0, out)
        self.assertIn("SELFTEST PASS", out)
        self.assertIn("[PASS]", out)

    def test_the_gate_carries_the_unit_suite(self):
        """Until this existed, the 367 tests were gated NOWHERE: pre-commit,
        pre-push and CI all ran `selftest` and nothing else, so the suite
        that carries every claim in the README executed only when a human
        remembered to type the command. selftest is the one gate all three
        share, and the one this repo is allowed to edit — the workflow file
        needs a permission the App does not have."""
        import ast
        import inspect
        src = inspect.getsource(frontend.cmd_selftest)
        tree = ast.parse(src.lstrip())
        consts = {n.value for n in ast.walk(tree)
                  if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        self.assertIn("unittest", consts,
                      "the gate no longer runs the unit suite")
        self.assertIn("discover", consts)

    def test_the_gate_cannot_fork_itself_forever(self):
        """The suite spawns `selftest` (TestSelftestGate), and selftest now
        spawns the suite. The env flag is what stops that being a fork bomb;
        it is inherited by the subprocess, so the inner run skips."""
        import subprocess
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ, LMM_SELFTEST_SKIP_LIVE="1",
                   LMM_SELFTEST_NO_SUITE="1")
        r = subprocess.run([sys.executable, os.path.join(root, "lmm.py"),
                            "selftest"], cwd=root, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=300)
        out = r.stdout.decode("utf-8", "ignore")
        self.assertEqual(r.returncode, 0, out)
        self.assertIn("already inside one", out,
                      "the recursion guard did not engage")

    def test_a_machine_without_a_backend_does_not_fail_the_gate(self):
        """`doctor` grades the machine; `selftest` grades lmm.

        Requiring "doctor: HEALTHY" here made the gate unpassable on any host
        without a live backend — including this project's own CI runner, where
        the live checks are skipped for exactly that reason.
        """
        r = self._run()
        out = r.stdout.decode("utf-8", "ignore")
        self.assertIn("doctor command runs", out)
        self.assertNotIn("[FAIL] doctor", out)


def compact(keep=None):
    """Force a compaction from a test, taking the writer lock exactly as the
    hub does. This used to be a public `usage_compact` in the product, but
    nothing in the product ever called it — the tests were its only reason
    to exist, so it lives here now, where its one caller is."""
    with lmm._USAGE_LOCK:
        return lmm._usage_compact_locked(keep)


class temp_state(object):
    """Point lmm's usage log and cache at a throwaway directory."""

    def __enter__(self):
        self.dir = tempfile.mkdtemp(prefix="lmm-test-")
        self.saved = (lmm.LMM_DIR, lmm.USAGE_LOG, lmm.CACHE_LOG,
                      lmm.CLAUDE_PROJECTS)
        # The breaker is process-wide AND persisted; without a reset, one
        # test opening a circuit for "stub" poisons every later test that
        # reuses the name — for cooldown_s of wall time.
        lmm.HUB_BREAKER.reset()
        lmm.LMM_DIR = self.dir
        lmm.USAGE_LOG = os.path.join(self.dir, "usage.jsonl")
        lmm.CACHE_LOG = os.path.join(self.dir, "cache.jsonl")
        # Also point the Anthropic session-log reader at the sandbox: without
        # this, cost assertions pick up whatever ~/.claude happens to contain
        # on the machine running the tests.
        lmm.CLAUDE_PROJECTS = os.path.join(self.dir, "claude-projects")
        return self

    def __exit__(self, *exc):
        lmm.HUB_BREAKER.reset()
        (lmm.LMM_DIR, lmm.USAGE_LOG, lmm.CACHE_LOG,
         lmm.CLAUDE_PROJECTS) = self.saved
        for name in ("usage.jsonl", "cache.jsonl"):
            try:
                os.remove(os.path.join(self.dir, name))
            except OSError:
                pass
        try:
            os.rmdir(self.dir)
        except OSError:
            pass
        return False


class TestBillArithmeticSurvivesElenchus(unittest.TestCase):
    """The product's first promise is "a bill you can actually see". The
    fragments (usage_cost, price_for) were each tested, but no test walked the
    whole chain — published rate -> metered event -> aggregate -> report — and
    asserted an exact dollar figure. This is that known-answer walk: at
    $3/M in and $15/M out, 1M+1M tokens is $18.000000, end to end."""

    RATE = {"in": 3.0, "out": 15.0, "cw": 0.0, "cr": 0.0}
    PROV = {"kind": "remote", "price": RATE}

    def test_published_rate_to_report_is_exactly_18_dollars(self):
        rate = lmm.price_for(self.PROV, "m", lmm.merged_pricing({}))
        usd = lmm.usage_cost({"prompt_tokens": 1_000_000,
                              "completion_tokens": 1_000_000}, rate)
        self.assertAlmostEqual(usd, 18.0, places=6)
        with temp_state():
            lmm.log_usage({"provider": "p", "kind": "remote",
                           "in": 1_000_000, "out": 1_000_000,
                           "usd": round(usd, 6)})
            st = lmm.hub_cost_stats()
            report = lmm.cost_report({}, days=None)
        self.assertAlmostEqual(st["measured"], 18.0, places=6)
        self.assertIn("$18.0000", report)
        self.assertIn("in=1,000,000 out=1,000,000", report)


class TestCorruptStateSurvivesElenchus(unittest.TestCase):
    """~/.lmm/*.jsonl are plain text the README invites the user to look at —
    and therefore to touch. Claim on trial: "the readers survive corrupt state
    files". Verdict before the fix, measured: `lmm cost` and `lmm status` died
    on one type-mangled field (ValueError at float()), and one bad `at` value
    in cache.jsonl silently discarded every entry AFTER it. A bad field must
    cost itself, not the command and not the rest of the file."""

    HOSTILE_USAGE = [
        {"provider": "good1", "kind": "remote", "in": 100, "out": 100, "usd": 0.5},
        {"provider": "bad", "usd": "abc", "in": "5", "out": []},
        {"rollup": 1, "providers": []},
        {"rollup": 1, "providers": {"x": {"calls": "many", "usd": "??"}, "y": 3}},
        {"provider": "bad2", "stream": True, "ttft_ms": "x", "usd": 0.1},
        {"provider": "good2", "kind": "remote", "in": 200, "out": 200, "usd": 1.5},
    ]

    def test_cost_survives_and_keeps_lines_after_the_bad_one(self):
        with temp_state():
            for ev in self.HOSTILE_USAGE:
                lmm.log_usage(dict(ev))
            st = lmm.hub_cost_stats()          # must not raise
            report = lmm.cost_report({}, days=None)
        # good2 sits AFTER every hostile line; if it is missing, the reader
        # aborted the file instead of the field.
        self.assertIn("good2", report)
        self.assertAlmostEqual(st["measured"], 0.5 + 0.1 + 1.5, places=6)
        self.assertEqual(st["ttfts"], [])      # "x" is not a latency

    def test_cache_reader_keeps_entries_after_a_bad_timestamp(self):
        far = time.time() + 3600
        with temp_state():
            for e in ({"prompt": "good-before", "reply": "a", "at": far},
                      {"prompt": "bad", "reply": "b", "at": "soon"},
                      {"prompt": "good-after", "reply": "c", "at": far}):
                with open(lmm.CACHE_LOG, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(e) + "\n")
            names = [e.get("prompt") for e in lmm.cache_entries({})]
        self.assertIn("good-after", names)     # the fix: field-local failure

    def test_cache_survives_a_non_numeric_ttl_in_config(self):
        far = time.time() + 3600
        with temp_state():
            with open(lmm.CACHE_LOG, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"prompt": "p", "reply": "r", "at": far}) + "\n")
            entries = lmm.cache_entries({"ttl_hours": "week"})   # must not raise
        self.assertEqual([e.get("prompt") for e in entries], ["p"])


class TestConcurrentWritersSurviveElenchus(unittest.TestCase):
    """backend.py claims writers serialise, citing 21.9% of metering events
    lost under concurrency as the bug the lock fixed. Derive the consequence:
    then two concurrent lmm PROCESSES must not lose events either — which is
    exactly how the product ships, a resident `serve --hub` alongside `lmm
    ask`, `lmm bench` and the GUI. Measured with only the threading.Lock,
    800 submitted events came back as 639: 20.1% erased by compaction's
    os.replace. A threading.Lock cannot reach across a process boundary; the
    lock has to be a file lock too.
    """

    RACER = r'''
import sys, os, time
sys.path.insert(0, %r)
import backend as lmm
role, n, d = sys.argv[1], int(sys.argv[2]), sys.argv[3]
lmm.LMM_DIR = d
lmm.USAGE_LOG = os.path.join(d, "usage.jsonl")
lmm.USAGE_MAX_BYTES = 20000        # force compaction to run constantly
lmm.USAGE_KEEP_TAIL = 20
# Start barrier. Without it the two interpreters spend most of their lives
# importing, they barely overlap, and an unlocked run loses nothing -- the
# test passes the very mutation it exists to catch. Measured with the
# barrier and the cross-process lock removed, n=150 lost ~50 of 300 events
# on every trial; n=60 was flaky.
open(os.path.join(d, "ready." + role), "w").close()
while not all(os.path.exists(os.path.join(d, "ready." + r))
              for r in ("A", "B")):
    time.sleep(0.005)
for i in range(n):
    lmm.log_usage({"provider": role, "kind": "remote", "in": 1, "out": 1,
                   "usd": 0.001, "seq": i})
    if role == "B" and i %% 5 == 0:
        with lmm._USAGE_LOCK, lmm._xlock("usage"):
            lmm._usage_compact_locked()
''' % os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def test_two_processes_racing_compaction_lose_no_events(self):
        n = 150
        with temp_state():
            script = os.path.join(lmm.LMM_DIR, "racer.py")
            os.makedirs(lmm.LMM_DIR, exist_ok=True)
            with open(script, "w", encoding="utf-8") as fh:
                fh.write(self.RACER)
            procs = [subprocess.Popen(
                [sys.executable, script, role, str(n), lmm.LMM_DIR],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                for role in ("A", "B")]
            try:
                for p in procs:
                    self.assertEqual(p.wait(timeout=300), 0, p.stdout.read())
                events = lmm.read_usage()
            finally:
                for p in procs:
                    p.stdout.close()
                for extra in ("racer.py", ".usage.lock",
                              "ready.A", "ready.B"):
                    try:
                        os.remove(os.path.join(lmm.LMM_DIR, extra))
                    except OSError:
                        pass
        accounted = sum(e.get("events", 1) if e.get("rollup") else 1
                        for e in events)
        self.assertEqual(accounted, 2 * n)     # the whole point: nothing lost

    def test_state_writes_still_work_where_no_file_lock_exists(self):
        """A no-op lock is the documented fallback: telemetry must never be
        able to break the call it is measuring."""
        f, m = lmm._fcntl, lmm._msvcrt
        try:
            lmm._fcntl = lmm._msvcrt = None
            with temp_state():
                lmm.log_usage({"provider": "p", "kind": "remote", "in": 1,
                               "out": 1, "usd": 0.25})
                events = lmm.read_usage()
        finally:
            lmm._fcntl, lmm._msvcrt = f, m
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0]["usd"], 0.25, places=6)

    def test_the_lock_file_is_not_a_state_file(self):
        """The rewrites end in os.replace, so a lock held on usage.jsonl
        itself would be thrown away with the old inode mid-rewrite."""
        with temp_state():
            with lmm._xlock("usage"):
                pass
            names = set(os.listdir(lmm.LMM_DIR))
        self.assertIn(".usage.lock", names)
        self.assertNotEqual(os.path.basename(lmm.USAGE_LOG), ".usage.lock")

    def test_hub_threads_and_compaction_do_not_deadlock_in_one_process(self):
        """The file lock nests inside the threading.Lock; several hub threads
        appending while compaction runs must still finish."""
        with temp_state():
            lmm.USAGE_MAX_BYTES, keep = 20000, lmm.USAGE_KEEP_TAIL
            lmm.USAGE_KEEP_TAIL = 20
            try:
                def work(tag):
                    for i in range(40):
                        lmm.log_usage({"provider": tag, "kind": "remote",
                                       "in": 1, "out": 1, "usd": 0.001})
                ts = [threading.Thread(target=work, args=(str(i),))
                      for i in range(4)]
                for t in ts:
                    t.start()
                for t in ts:
                    t.join(timeout=120)
                    self.assertFalse(t.is_alive(), "writer thread hung")
                events = lmm.read_usage()
            finally:
                lmm.USAGE_MAX_BYTES = 4_000_000
                lmm.USAGE_KEEP_TAIL = keep
        accounted = sum(e.get("events", 1) if e.get("rollup") else 1
                        for e in events)
        self.assertEqual(accounted, 160)


class TestCacheAnswersTheAskedQuestionSurvivesElenchus(unittest.TestCase):
    """The cache's claim is that reuse is free. Reuse is only free if the
    reused answer answers the question that was actually asked -- and the hub
    is an OpenAI-compatible endpoint that forwards the caller's parameters.

    Measured before the fix, keyed on prompt+model alone: a request with
    `max_tokens: 4000` was served the answer produced for `max_tokens: 5`,
    and a `response_format: {"type": "json_object"}` request was served a
    cached prose answer, on which the client's json.loads raised. One
    provider call served three incompatible requests.
    """

    TARGETS = [("p", {"kind": "remote", "base": "http://x", "model": "m",
                      "key": "k"})]
    MSGS = [{"role": "user", "content": "count the primes under 10"}]
    CFG = {"cache": {"enabled": True, "semantic": False}}

    def setUp(self):
        self.calls = []
        self.saved = lmm.call_provider

        def fake(prov, prompt, temperature=0.7, extra=None, messages=None,
                 stream=False):
            self.calls.append(dict(extra or {}))
            e = extra or {}
            body = ('{"n": 4}' if e.get("response_format")
                    else "short" if e.get("max_tokens") == 5
                    else "a long prose answer " * 6)
            return {"choices": [{"message": {"content": body}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20}}
        lmm.call_provider = fake

    def tearDown(self):
        lmm.call_provider = self.saved

    def ask(self, extra):
        r = lmm.hub_complete(self.CFG, self.MSGS, list(self.TARGETS),
                             {"extra": dict(extra), "cascade": False,
                              "cache": True})
        r = r[0] if isinstance(r, tuple) else r
        return r["choices"][0]["message"]["content"]

    def test_a_token_ceiling_does_not_share_an_answer_with_a_wider_one(self):
        with temp_state():
            first = self.ask({"max_tokens": 5})
            second = self.ask({"max_tokens": 4000})
        self.assertEqual(first, "short")
        self.assertNotEqual(second, first)
        self.assertEqual(len(self.calls), 2)

    def test_json_mode_is_never_served_a_cached_prose_answer(self):
        with temp_state():
            self.ask({"max_tokens": 4000})              # prose, cached
            body = self.ask({"response_format": {"type": "json_object"}})
        json.loads(body)          # the client's parse must not raise

    def test_identical_requests_still_hit_the_cache(self):
        """The fix must not fragment the cache it exists to fill."""
        with temp_state():
            for _ in range(5):
                self.ask({"max_tokens": 100, "stream": False, "user": "a"})
        self.assertEqual(len(self.calls), 1)

    def test_parameters_that_cannot_change_an_answer_do_not_fragment_it(self):
        with temp_state():
            self.ask({"max_tokens": 100, "stream": False, "user": "alice"})
            self.ask({"max_tokens": 100, "stream": True, "user": "bob",
                      "metadata": {"x": 1}})
        self.assertEqual(len(self.calls), 1)

    def test_an_unknown_parameter_fails_toward_a_miss_not_a_wrong_answer(self):
        """A deny-list, not an allow-list: a parameter this code has never
        heard of must cost a cache miss, never a wrong answer."""
        a = lmm.cache_key(self.MSGS, "m", {"reasoning_effort": "low"})
        b = lmm.cache_key(self.MSGS, "m", {"reasoning_effort": "high"})
        self.assertNotEqual(a, b)

    def test_the_semantic_tier_also_refuses_a_differently_shaped_neighbour(self):
        """A near-identical prompt is still the wrong answer if it was
        produced under different parameters."""
        conf = {"enabled": True, "semantic": True, "similarity": 0.5,
                "ttl_hours": 168, "max_entries": 100}
        cfg = {"cache": conf}
        saved_embed = lmm.embed_text
        lmm.embed_text = lambda text, model: [1.0, 0.0, 0.0]
        try:
            with temp_state():
                lmm.cache_store(cfg, [{"role": "user", "content": "x"}], "m",
                                {"choices": [{"message": {"content": "prose"}}]},
                                0.0, None, {"max_tokens": 5})
                entry, how, _sim, _x = lmm.cache_lookup(
                    cfg, [{"role": "user", "content": "y"}], "m",
                    {"response_format": {"type": "json_object"}})
        finally:
            lmm.embed_text = saved_embed
        self.assertIsNone(entry)
        self.assertNotEqual(how, "semantic")


class TestAskIsScriptableSurvivesElenchus(unittest.TestCase):
    """The README places `lmm ask` on the same code path as the hub an app
    points at. Derive the consequence: a CLI on that path must be usable FROM
    a script -- `answer=$(lmm ask ...)` and `lmm ask ... || fallback` must
    mean what they say. Exit status and the stdout/stderr split are the
    contract a CLI has with the shell.

    Measured before the fix, with no provider configured: `lmm ask hi` exited
    0 and printed "[ask] no provider available" to STDOUT, so the shell
    captured the failure as the answer and never took the fallback branch.
    `lmm bench` also exited 0 with nothing measured. frontend.py contained
    zero writes to sys.stderr.
    """

    NOPROV = {"providers": {}, "ask_order": []}

    def _run(self, fn, *a, **k):
        import io
        import contextlib
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = fn(*a, **k)
        return rc, out.getvalue(), err.getvalue()

    def test_a_failed_ask_exits_non_zero_and_keeps_stdout_clean(self):
        saved = lmm.local_ollama_provider
        lmm.local_ollama_provider = lambda: None       # no safety net
        try:
            with temp_state():
                rc, out, err = self._run(frontend.cmd_ask, "hi", None, self.NOPROV)
        finally:
            lmm.local_ollama_provider = saved
        self.assertNotEqual(rc, 0)
        self.assertEqual(out, "")                      # $(lmm ask) gets nothing
        self.assertIn("no provider available", err)    # the reason, on stderr

    def test_an_unknown_provider_is_a_failure_too(self):
        with temp_state():
            rc, out, err = self._run(frontend.cmd_ask, "hi", "nope", self.NOPROV)
        self.assertNotEqual(rc, 0)
        self.assertEqual(out, "")
        self.assertIn("unknown provider", err)

    def test_a_successful_ask_exits_zero_with_only_the_answer_on_stdout(self):
        stub = StubBackend()
        try:
            cfg = {"providers": {"stub": {
                "api_key": "k", "base_url": stub.base_url, "model": "m",
                "kind": "local"}}, "ask_order": ["stub"]}
            with temp_state():
                rc, out, err = self._run(frontend.cmd_ask, "2+2?", "stub", cfg,
                                         explain=True)
        finally:
            stub.stop()
        self.assertEqual(rc, 0)
        self.assertIn("The answer is 4", out)
        self.assertNotIn("[route]", out)               # trace lines are not the answer
        self.assertNotIn("[warn]", out)

    def test_an_empty_prompt_is_a_usage_error(self):
        with temp_state():
            rc, out, err = self._run(frontend.cmd_ask, "   ", None, self.NOPROV)
        self.assertEqual(rc, 2)
        self.assertIn("usage:", err)

    def test_bench_with_nothing_to_measure_fails(self):
        saved = lmm.local_ollama_provider
        lmm.local_ollama_provider = lambda: None
        try:
            with temp_state():
                rc, _out, err = self._run(frontend.cmd_bench, self.NOPROV)
        finally:
            lmm.local_ollama_provider = saved
        self.assertNotEqual(rc, 0)
        self.assertIn("no provider available", err)

    def test_fit_that_cannot_size_the_model_fails(self):
        saved = lmm.ollama_model_info
        lmm.ollama_model_info = lambda name: None
        try:
            rc, out, _err = self._run(frontend.cmd_fit, "no-such-model:1b",
                                      None, 8.0, "f16", False)
        finally:
            lmm.ollama_model_info = saved
        self.assertNotEqual(rc, 0)
        self.assertIn("no metadata", out)              # the table still prints

    def test_main_carries_the_status_to_the_shell(self):
        """The wiring, not just the function: main() must sys.exit with it."""
        saved_argv, saved_lo = sys.argv, lmm.local_ollama_provider
        saved_cfg = frontend.load_config
        lmm.local_ollama_provider = lambda: None
        frontend.load_config = lambda: dict(self.NOPROV)
        try:
            with temp_state():
                sys.argv = ["lmm", "ask", "hi"]
                with self.assertRaises(SystemExit) as cm:
                    self._run(frontend.main)
        finally:
            sys.argv, lmm.local_ollama_provider = saved_argv, saved_lo
            frontend.load_config = saved_cfg
        self.assertNotEqual(cm.exception.code, 0)

    def test_every_hub_error_has_the_one_shape_openai_clients_parse(self):
        """Two error shapes for one server: 400 unknown-model used the
        OpenAI object form, 502/404 used a bare string. The SDK reads
        `error.message` and found nothing on the string."""
        cfg = {"providers": {"dead": {
            "api_key": "k", "base_url": "http://127.0.0.1:9/v1", "model": "m",
            "kind": "local"}}, "ask_order": ["dead"]}
        saved = lmm.local_ollama_provider
        lmm.local_ollama_provider = lambda: None
        try:
            with temp_state():
                hub = HubServer(cfg)
                seen = []
                for path, body in (("/v1/chat/completions",
                                    {"model": "dead", "messages": [
                                        {"role": "user", "content": "hi"}]}),
                                   ("/v1/nope", {"x": 1}),
                                   ("/v1/nothing", None)):
                    code, text = hub.request(path, body)
                    seen.append((path, code, json.loads(text)))
        finally:
            lmm.local_ollama_provider = saved
        for path, code, body in seen:
            self.assertGreaterEqual(code, 400, path)
            self.assertIsInstance(body.get("error"), dict, (path, body))
            self.assertIn("message", body["error"], path)
            self.assertIn("type", body["error"], path)
        self.assertEqual(seen[0][1], 502)


class TestStopKillsOnlyWhatItNamesSurvivesElenchus(unittest.TestCase):
    """`lmm stop <runtime>` names a runtime; the claim is that it stops that
    runtime. Cross-examined, the implementation said otherwise. It ran
    `pkill -f '<name>'` and `pgrep -fl 'a' 'b' ...` through a shell, and all
    three consequences were measured:

      - `pgrep -fl 'ollama' 'ollama app'` exits 2, "only one pattern can be
        provided" -- every registry entry lists 2+ names, so detection
        returned 0 on all Linux and macOS machines.
      - `-f` matches the command line: run from a process whose argv held
        the pattern, `pgrep -fl` listed our own interpreter, the invoking
        bash, and the `sh -c` wrapper. As `pkill`, `lmm stop ollama` would
        have killed lmm and the user's shell.
      - a `procs` entry from a WORKING-DIRECTORY config closed the quote and
        ran arbitrary shell; the payload's marker file appeared.
    """

    def setUp(self):
        self._saved = (lmm.CONFIG_PATH, lmm.CONFIG_TRUSTED)

    def tearDown(self):
        lmm.CONFIG_PATH, lmm.CONFIG_TRUSTED = self._saved

    def test_a_procs_entry_cannot_run_a_shell_command(self):
        """The one that was exploitable. Even from the user's OWN config --
        no shell means no injection, trusted or not."""
        d = tempfile.mkdtemp(prefix="lmm-inject-")
        marker = os.path.join(d, "pwned")
        payload = "[z]zz-nomatch'; touch %s; echo '" % marker
        try:
            lmm.CONFIG_PATH, lmm.CONFIG_TRUSTED = "/home/u/.lmm/config.json", True
            cfg = {"extra_runtimes": [{"name": "evil", "procs": [payload]}]}
            import io
            import contextlib
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                frontend.cmd_stop("evil", cfg)
            self.assertFalse(os.path.exists(marker), "procs entry ran a shell")
            self.assertEqual(lmm.proc_list([payload]), [])
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_an_untrusted_config_does_not_choose_what_stop_kills(self):
        """Same rule `models_cmd` already had: a config from the directory
        you happened to cd into must not name processes to terminate."""
        import io
        import contextlib
        lmm.CONFIG_PATH, lmm.CONFIG_TRUSTED = "lmm.config.json", False
        cfg = {"extra_runtimes": [{"name": "evil", "procs": ["python"]}]}
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(err):
            rc = frontend.cmd_stop("evil", cfg)
        self.assertNotEqual(rc, 0)                    # "evil" never registered
        self.assertIn("ignoring extra_runtimes", err.getvalue())

    def test_proc_list_never_returns_the_process_asking(self):
        """`lmm stop` runs inside a python process; a matcher that can see
        itself is a matcher that can kill itself."""
        me = os.getpid()
        for name in ("python", "python3", os.path.basename(sys.executable)):
            self.assertNotIn(me, [pid for pid, _n in lmm.proc_list([name])],
                             "proc_list returned its own pid for %r" % name)

    def test_it_still_finds_and_stops_a_real_process(self):
        """The other direction: a guard that stops nothing would pass every
        test above. Start a real child, find it by name, stop it."""
        child = subprocess.Popen(["sleep", "60"])
        try:
            found = [pid for pid, _n in lmm.proc_list(["sleep"])]
            self.assertIn(child.pid, found, "proc_list missed a live process")
            self.assertGreaterEqual(lmm.proc_count(["sleep"]), 1)
            lmm.stop_procs(["sleep"])
            self.assertEqual(child.wait(timeout=30), -15)   # SIGTERM
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)

    def test_a_partial_name_does_not_match(self):
        """`-f` substring matching is what made `stop` hit bystanders. The
        image name must match whole."""
        child = subprocess.Popen(["sleep", "60"])
        try:
            for partial in ("sle", "eep", "leep"):
                self.assertEqual(
                    [pid for pid, _n in lmm.proc_list([partial])
                     if pid == child.pid], [],
                    "%r matched a process named sleep" % partial)
        finally:
            child.kill()
            child.wait(timeout=10)

    def test_an_unusable_process_table_is_not_a_crash(self):
        """Telemetry-grade rule: probing must never break the command."""
        saved = subprocess.run
        try:
            subprocess.run = lambda *a, **k: (_ for _ in ()).throw(OSError("no ps"))
            self.assertEqual(lmm.proc_list(["sleep"]), [])
            self.assertEqual(lmm.proc_count(["sleep"]), 0)
        finally:
            subprocess.run = saved


class TestHubForwardsWhatItWasAskedSurvivesElenchus(unittest.TestCase):
    """The hub's whole pitch is "point your app at it". Derive the
    consequence: what the app asked for must reach the provider. It didn't.

    Forwarding was an allow-list of the OpenAI fields this code happened to
    know about. Measured against a provider that echoes its request, a client
    sending logprobs, top_logprobs, logit_bias, parallel_tool_calls,
    reasoning_effort and service_tier had all six dropped, with no error --
    a proxy answering a different question than the one it was asked.

    Same argument as the cache key: an unfamiliar field must fail toward
    being FORWARDED (the provider either accepts it or rejects it by name)
    rather than toward silence.
    """

    NEWER_THAN_THE_LIST = {"logprobs": True, "top_logprobs": 3,
                           "logit_bias": {"123": -100},
                           "parallel_tool_calls": False,
                           "reasoning_effort": "high",
                           "service_tier": "scale"}

    def test_fields_the_hardcoded_list_never_heard_of_reach_the_provider(self):
        stub = StubBackend()
        try:
            cfg = {"providers": {"stub": {
                "api_key": "k", "base_url": stub.base_url, "model": "m",
                "kind": "local"}}, "ask_order": ["stub"],
                "cache": {"enabled": False}}
            with temp_state():
                hub = HubServer(cfg)
                body = dict(self.NEWER_THAN_THE_LIST,
                            model="stub", max_tokens=77,
                            messages=[{"role": "user", "content": "hi"}])
                code, _text = hub.request("/v1/chat/completions", body)
            self.assertEqual(code, 200)
            got = stub.seen[-1]
        finally:
            stub.stop()
        for k, v in self.NEWER_THAN_THE_LIST.items():
            self.assertIn(k, got, "the hub dropped %r" % k)
            self.assertEqual(got[k], v, k)
        self.assertEqual(got["max_tokens"], 77)

    def test_the_hub_still_owns_the_fields_it_rewrites(self):
        """The other direction: forwarding everything would break routing.
        `model` is a provider alias here, not a model id the backend knows,
        and lmm's own controls are meaningless to a provider."""
        got = lmm.passthrough({"model": "stub", "messages": [],
                               "stream": True, "stream_options": {"x": 1},
                               "lmm_cascade": True, "lmm_no_cache": True,
                               "max_tokens": 5})
        self.assertEqual(got, {"max_tokens": 5})

    def test_a_forwarded_field_also_reaches_the_cache_key(self):
        """These two fixes have to agree: a field the hub forwards changes
        the answer, so it must also split the cache. An allow-listed
        forwarder made that impossible for anything it had not heard of."""
        msgs = [{"role": "user", "content": "hi"}]
        a = lmm.cache_key(msgs, "m", {"reasoning_effort": "low"})
        b = lmm.cache_key(msgs, "m", {"reasoning_effort": "high"})
        self.assertNotEqual(a, b)


class TestGraderDoesNotEscalateCorrectAnswersSurvivesElenchus(unittest.TestCase):
    """The cascade's claim is that it SPENDS LESS: try the cheap model, stop
    when the answer is good enough. Derive the consequence — a correct cheap
    answer must pass the gate, or the cascade pays the expensive model for
    nothing.

    Measured against VERIFY_GATE (0.75), the grader opened with
    `len(text.split()) < 3 -> 0.0, "empty or near-empty answer"`. So "Paris."
    for "What is the capital of France?" scored **0.00** and escalated, as
    did "4", "Yes.", "1989." and "404" — precisely the questions a small
    model gets right. Brevity is not emptiness; whether an answer is too
    short is a question about the PROMPT, and the reasoning / code /
    multi-part rules already ask it.
    """

    CORRECT_AND_SHORT = [
        ("What is 2+2?", "4"),
        ("What is the capital of France?", "Paris."),
        ("Is 17 prime?", "Yes."),
        ("What year did the Berlin Wall fall?", "1989."),
        ("Reply with just the HTTP status code for 'not found'.", "404"),
    ]

    # The other direction: brevity where the prompt demanded more, and
    # answers with no content at all. A grader that passed everything would
    # satisfy the cases above and be worthless.
    SHORT_AND_WRONG = [
        ("Explain how TCP congestion control works.", "It works."),
        ("Why is the sky blue?", "Blue."),
        ("Summarize this. It has many sentences. Here is another. And more.",
         "ok"),
        ("Write a python function to reverse a list.", "Sure."),
    ]

    NO_CONTENT = ["", "   ", "...", "???", "\n\n"]

    def test_a_correct_short_answer_is_not_escalated(self):
        for prompt, answer in self.CORRECT_AND_SHORT:
            with self.subTest(answer=answer):
                score, why = lmm.verify_answer(prompt, answer)
                self.assertGreaterEqual(
                    score, lmm.VERIFY_GATE,
                    "%r would escalate to the expensive model: %s"
                    % (answer, why))

    def test_brevity_the_prompt_did_not_invite_still_fails(self):
        for prompt, answer in self.SHORT_AND_WRONG:
            with self.subTest(answer=answer):
                score, why = lmm.verify_answer(prompt, answer)
                self.assertLess(score, lmm.VERIFY_GATE,
                                "%r passed the gate: %s" % (answer, why))
                self.assertTrue(why, "no reason given for the deduction")

    def test_an_answer_with_no_content_is_still_zero(self):
        for answer in self.NO_CONTENT:
            with self.subTest(answer=answer):
                score, why = lmm.verify_answer("What is 2+2?", answer)
                self.assertEqual(score, 0.0)
                self.assertEqual(why, ["empty or near-empty answer"])

    def test_the_cascade_stops_on_a_correct_short_answer(self):
        """End to end, through the real cascade: the cheap rung answers
        "Paris." and the expensive rung must never be called."""
        called = []

        def fake(prov, prompt, temperature=0.7, extra=None, messages=None,
                 stream=False):
            called.append(prov.get("model"))
            return {"choices": [{"message": {"content": "Paris."}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1}}

        saved = lmm.call_provider
        lmm.call_provider = fake
        try:
            targets = [("cheap", {"kind": "local", "base": "http://x",
                                  "model": "small", "key": "k"}),
                       ("dear", {"kind": "remote", "base": "http://y",
                                 "model": "big", "key": "k"})]
            with temp_state():
                lmm.hub_complete({"cache": {"enabled": False}},
                                 [{"role": "user",
                                   "content": "What is the capital of France?"}],
                                 targets, {"cascade": True, "cache": False})
        finally:
            lmm.call_provider = saved
        self.assertEqual(called, ["small"],
                         "the cascade paid for the expensive rung anyway")


class TestGraderRulesActuallyBiteSurvivesElenchus(unittest.TestCase):
    """Last round fixed the grader escalating correct answers. The paired
    consequence: on `--verify`, the same grader is the gatekeeper, and there
    letting a bad answer THROUGH is what costs. Two rules that exist were
    measured not doing their job.

    1. Markers were counted with `next(...)`, so only the first ever scored.
       "I think it might be 4, but I'm not sure, probably, maybe." -- five
       hedge markers -- scored 0.85, exactly what one hedge scores, and
       PASSED the 0.75 gate. The hedging rule exists to catch that reply.
    2. "truncated mid-sentence" fired on any answer without terminal
       punctuation, including complete single-token ones: "Bonjour", "404",
       "4" and "Paris" each lost 0.25 and landed on exactly 0.75 -- correct
       answers balanced on the gate, one deduction from being escalated.
       A single token is not a truncated sentence.
    """

    def test_stacked_hedges_cost_more_than_one(self):
        one = lmm.verify_answer("What is 2+2?",
                                "I think the answer is 4 and that is correct.")[0]
        many, why = lmm.verify_answer(
            "What is 2+2?",
            "I think it might be 4, but I'm not sure, probably, maybe.")
        self.assertLess(many, one, "five hedges scored like one: %s" % why)
        self.assertLess(many, lmm.VERIFY_GATE, "a hedged non-answer passed")

    def test_stacked_refusal_markers_cost_more_than_one(self):
        one = lmm.verify_answer("Name the CEO.",
                                "I cannot help with that request, sorry.")[0]
        many = lmm.verify_answer(
            "Name the CEO.",
            "I cannot help. I am unable to browse. As an AI, I don't have "
            "access, sorry, i must decline.")[0]
        self.assertLess(many, one)
        self.assertGreaterEqual(many, 0.0)      # still clamped

    def test_one_hedge_still_scores_what_it_always_did(self):
        """Counting must not silently re-tune the single-marker case."""
        score, why = lmm.verify_answer(
            "What is 2+2?", "I think the answer is 4 and that is correct.")
        self.assertAlmostEqual(score, 0.85, places=6)
        self.assertEqual(why, ["hedging (i think)"])

    def test_a_single_token_is_not_a_truncated_sentence(self):
        for prompt, answer in (("Translate 'hello' to French.", "Bonjour"),
                               ("HTTP status code for not found?", "404"),
                               ("What is 2+2?", "4"),
                               ("What is the capital of France?", "Paris")):
            with self.subTest(answer=answer):
                score, why = lmm.verify_answer(prompt, answer)
                self.assertNotIn("truncated mid-sentence", why)
                self.assertGreater(score, lmm.VERIFY_GATE,
                                   "%r sits on the gate: %s" % (answer, why))

    def test_a_real_truncation_is_still_caught(self):
        """The other direction: a rule that never fires is not a rule."""
        for answer in ("The answer is 4 because addition combines the two",
                       "It works because the"):
            with self.subTest(answer=answer):
                _score, why = lmm.verify_answer("Explain addition.", answer)
                self.assertIn("truncated mid-sentence", why)


if __name__ == "__main__":
    unittest.main()
