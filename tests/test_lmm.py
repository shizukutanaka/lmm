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
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lmm  # noqa: E402


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
            entry, how, sim = lmm.cache_lookup({}, msgs, "m")
            self.assertEqual(how, "exact")
            self.assertEqual(sim, 1.0)
            self.assertEqual(entry["result"], result)

    def test_high_temperature_is_not_cached(self):
        msgs = [{"role": "user", "content": "surprise me"}]
        with temp_state():
            lmm.cache_store({}, msgs, "m", {"ok": 1}, usd=0.0, temperature=0.9)
            entry, how, _ = lmm.cache_lookup({}, msgs, "m")
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
            entry, how, _ = lmm.cache_lookup({}, msgs, "m")
            self.assertIsNone(entry)

    def test_disabled_cache_never_hits(self):
        msgs = [{"role": "user", "content": "x"}]
        cfg = {"cache": {"enabled": False}}
        with temp_state():
            lmm.cache_store({}, msgs, "m", {"ok": 1}, temperature=0.0)
            entry, how, _ = lmm.cache_lookup(cfg, msgs, "m")
            self.assertIsNone(entry)


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

    def test_private_prompt_with_no_local_provider_warns(self):
        self.fake({})
        with temp_state():
            _res, trace = lmm.hub_complete({}, "summarize this secret memo",
                                           self.targets(),
                                           {"cascade": False, "cache": False})
        self.assertTrue(any(t.startswith("[warn]") for t in trace))

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
            lmm.cmd_examples()
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
            entry, how, _ = lmm.cache_lookup({}, lmm.as_messages("hi"), "p")
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
        def _lookup(cfg, messages, model):
            self.lookups.append(model)
            return self._lookup(cfg, messages, model)

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
            entry, how, _ = lmm.cache_lookup({}, lmm.as_messages("hi"), "p")
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


class temp_state(object):
    """Point lmm's usage log and cache at a throwaway directory."""

    def __enter__(self):
        self.dir = tempfile.mkdtemp(prefix="lmm-test-")
        self.saved = (lmm.LMM_DIR, lmm.USAGE_LOG, lmm.CACHE_LOG)
        lmm.LMM_DIR = self.dir
        lmm.USAGE_LOG = os.path.join(self.dir, "usage.jsonl")
        lmm.CACHE_LOG = os.path.join(self.dir, "cache.jsonl")
        return self

    def __exit__(self, *exc):
        lmm.LMM_DIR, lmm.USAGE_LOG, lmm.CACHE_LOG = self.saved
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


if __name__ == "__main__":
    unittest.main()
