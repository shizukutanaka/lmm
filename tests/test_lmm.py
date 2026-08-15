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
