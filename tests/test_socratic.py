"""Tests for the Socratic agent.

The contract tests matter most: the evaluator counts an exception, a malformed
response or a timeout as a miss, so "never raises and always returns something
scoreable" is a correctness property, not a nicety. They run against a small
synthetic catalogue so the suite stays fast.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from socratic.belief import Belief
from socratic.config import AgentConfig
from socratic.dialogue import parse
from socratic.index import CatalogIndex, product_clauses
from socratic.policy import ALLOWED_ATTRIBUTES, Policy, answer_signature, session_score
from starter.agent import Agent

CATALOG = [
    {
        "parent_asin": "B000000001",
        "title": "Columbia Men's Thistletown Park Crew",
        "features": ["67% Polyester, 33% Cotton", "Imported", "Machine Wash"],
        "description": ["A performance crew built for outdoor activity."],
        "price": 27.99,
        "categories": ["Clothing, Shoes & Jewelry", "Men", "Shirts"],
        "details": {"Department": "mens", "Manufacturer": "Columbia"},
        "average_rating": 4.7,
        "rating_number": 5531,
        "store": "Columbia",
    },
    {
        "parent_asin": "B000000002",
        "title": "Pentagram Alloy Pendant Necklace",
        "features": ["Material:alloy", "Triple Moon Pentagram Symbol", "Imported"],
        "description": ["A moon pendant in black alloy."],
        "price": 9.99,
        "categories": ["Clothing, Shoes & Jewelry", "Women", "Jewelry", "Necklaces"],
        "details": {"Department": "womens"},
        "average_rating": 4.2,
        "rating_number": 84,
        "store": "QIAN0813",
    },
    {
        "parent_asin": "B000000003",
        "title": "Plain Cotton Tee",
        "features": ["100% Cotton", "Imported"],
        "description": ["A white cotton t-shirt."],
        "price": 12.50,
        "categories": ["Clothing, Shoes & Jewelry", "Women", "Shirts"],
        "details": {"Department": "womens"},
        "average_rating": 4.0,
        "rating_number": 12,
        "store": "Generic",
    },
]


def write_catalog(directory: str) -> str:
    path = Path(directory) / "catalog.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for product in CATALOG:
            handle.write(json.dumps(product) + "\n")
    return str(path)


class AgentContractTest(unittest.TestCase):
    """The response must always be scoreable, whatever it is handed."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.catalog = write_catalog(cls._tmp.name)
        cls.valid = {p["parent_asin"] for p in CATALOG}
        # Offline: the tests must pass on a machine with no model daemon.
        cls.config = AgentConfig(llm_mode="off")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def agent(self) -> Agent:
        return Agent(self.catalog, config=self.config)

    def test_response_shape_is_valid(self) -> None:
        agent = self.agent()
        agent.reset("s1", {"preference_tags": ["fit"]})
        response = agent.respond("s1", "I'm looking for Jewelry Necklaces. A key requirement is: Material:alloy.", 1, 10)
        self.assertIsInstance(response["message"], str)
        self.assertTrue(response["message"])
        self.assertIn(response["ask_attribute"], set(ALLOWED_ATTRIBUTES) | {None})
        self.assertIsInstance(response["recommendations"], list)
        for item in response["recommendations"]:
            self.assertIn(item["parent_asin"], self.valid)
        self.assertGreaterEqual(response["usage"]["prompt_tokens"], 0)
        self.assertGreaterEqual(response["usage"]["completion_tokens"], 0)

    def test_recommendations_are_unique_and_bounded(self) -> None:
        agent = self.agent()
        agent.reset("s2", {})
        for turn in range(1, 11):
            response = agent.respond("s2", "For that, what matters is: Imported.", turn, 10)
            asins = [item["parent_asin"] for item in response["recommendations"]]
            self.assertEqual(len(asins), len(set(asins)))
            self.assertLessEqual(len(asins), 10)

    def test_respects_top_k(self) -> None:
        agent = self.agent()
        agent.reset("s3", {})
        response = agent.respond("s3", "I'm looking for shirts, but I'm still exploring.", 10, 2)
        self.assertLessEqual(len(response["recommendations"]), 2)

    def test_never_raises_on_hostile_input(self) -> None:
        agent = self.agent()
        agent.reset("s4", {})
        for message in ["", "   ", "?" * 5000, "\x00\x01", "DROP TABLE products;--", "🙂🙂🙂"]:
            response = agent.respond("s4", message, 1, 10)
            self.assertIsInstance(response["message"], str)

    def test_respond_without_reset_does_not_raise(self) -> None:
        agent = self.agent()
        response = agent.respond("never-reset", "hello", 1, 10)
        self.assertIsInstance(response["message"], str)

    def test_runs_with_model_disabled(self) -> None:
        """The scoring machine may have no daemon and no network."""
        agent = Agent(self.catalog, config=AgentConfig(llm_mode="off"))
        agent.reset("s5", {})
        response = agent.respond("s5", "I'm looking for necklaces. A key requirement is: Material:alloy.", 1, 10)
        self.assertEqual(response["usage"], {"prompt_tokens": 0, "completion_tokens": 0})
        self.assertTrue(response["recommendations"] or response["ask_attribute"])

    def test_usage_is_per_turn_not_cumulative(self) -> None:
        """The evaluator sums `usage` across turns.

        Reporting a running total makes reported token use grow quadratically
        in the number of turns. It must describe this turn only.
        """
        agent = self.agent()
        agent.reset("s7", {})
        for turn in range(1, 6):
            usage = agent.respond("s7", "For that, what matters is: Imported.", turn, 10)["usage"]
            self.assertEqual(usage, {"prompt_tokens": 0, "completion_tokens": 0})
        self.assertEqual(agent.total_usage, {"prompt_tokens": 0, "completion_tokens": 0})

    def test_is_deterministic(self) -> None:
        outputs = []
        for _ in range(2):
            agent = self.agent()
            agent.reset("s6", {})
            outputs.append(
                [
                    item["parent_asin"]
                    for item in agent.respond(
                        "s6", "I'm looking for necklaces. A key requirement is: Material:alloy.", 1, 10
                    )["recommendations"]
                ]
            )
        self.assertEqual(outputs[0], outputs[1])

    def test_sessions_are_isolated(self) -> None:
        agent = self.agent()
        agent.reset("a", {})
        agent.reset("b", {})
        agent.respond("a", "For that, what matters is: Material:alloy.", 1, 10)
        self.assertEqual(agent.sessions["b"].belief.disclosed, [])


class IndexTest(unittest.TestCase):
    def test_clauses_cover_features_details_and_materials(self) -> None:
        clauses = product_clauses(CATALOG[1])
        self.assertIn("material:alloy", clauses)
        self.assertIn("triple moon pentagram symbol", clauses)
        self.assertIn("department: womens", clauses)
        self.assertIn("alloy", clauses)          # material surfaced implicitly
        self.assertIn("color: black", clauses)   # colour surfaced from description

    def test_clause_lookup_finds_the_owning_product(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index = CatalogIndex.build(write_catalog(tmp))
            ids = index.lookup_clause("Triple Moon Pentagram Symbol")
            self.assertEqual([index.asins[i] for i in ids], ["B000000002"])

    def test_attribute_value_strings_are_indexed_under_the_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index = CatalogIndex.build(write_catalog(tmp))
            self.assertIsNotNone(index.lookup_clause("alloy"))


class BeliefTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.index = CatalogIndex.build(write_catalog(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_evidence_moves_the_posterior_to_the_matching_product(self) -> None:
        belief = Belief(self.index)
        belief.add("Material:alloy", 1)
        order, probabilities, _ = belief.posterior(top=3)
        self.assertEqual(self.index.asins[order[0]], "B000000002")
        self.assertGreater(probabilities[0], probabilities[1])

    def test_retraction_restores_the_earlier_posterior(self) -> None:
        belief = Belief(self.index)
        before = belief.posterior(top=3)[1]
        belief.add("Material:alloy", 1)
        belief.retract(lambda item: item.kind == "constraint")
        after = belief.posterior(top=3)[1]
        self.assertAlmostEqual(before[0], after[0], places=6)

    def test_ruling_out_demotes_a_product(self) -> None:
        belief = Belief(self.index)
        belief.add("Material:alloy", 1)
        leader = belief.posterior(top=3)[0][0]
        belief.rule_out([leader])
        self.assertNotEqual(belief.posterior(top=3)[0][0], leader)

    def test_readmit_undoes_elimination(self) -> None:
        belief = Belief(self.index)
        belief.add("Material:alloy", 1)
        leader = belief.posterior(top=3)[0][0]
        belief.rule_out([leader])
        belief.readmit()
        self.assertEqual(belief.posterior(top=3)[0][0], leader)


class PolicyTest(unittest.TestCase):
    def test_session_score_matches_the_published_formula(self) -> None:
        # 0.50*hit + 0.30*(1/rank) + 0.20*(11 - turn)/10
        self.assertAlmostEqual(session_score(1, 1), 0.5 + 0.3 + 0.2)
        self.assertAlmostEqual(session_score(2, 6), 0.5 + 0.15 + 0.1)
        self.assertAlmostEqual(session_score(10, 10), 0.5 + 0.03 + 0.02)

    def test_answer_signature_stops_at_two_and_skips_disclosed(self) -> None:
        clauses = ("cotton", "imported", "machine wash")
        buckets = ("material", "feature", "feature")
        self.assertEqual(
            answer_signature(clauses, buckets, "other", frozenset()), ("cotton", "imported")
        )
        self.assertEqual(
            answer_signature(clauses, buckets, "other", frozenset({"cotton"})),
            ("imported", "machine wash"),
        )
        self.assertEqual(
            answer_signature(clauses, buckets, "material", frozenset()), ("cotton",)
        )

    def test_final_turn_always_shows_the_full_list(self) -> None:
        """Nothing is held back on turn 10 -- there is no later turn to win."""
        policy = Policy(index=None)
        width, _ = policy.choose_width([0.4, 0.2, 0.1] + [0.01] * 7, 0.23, 10)
        self.assertEqual(width, 10)

    def test_a_confident_belief_is_worth_more_than_a_hopeless_one(self) -> None:
        policy = Policy(index=None)
        _, confident = policy.choose_width([0.9] + [0.005] * 9, 0.055, 3)
        _, hopeless = policy.choose_width([0.0005] * 10, 0.995, 3)
        self.assertGreater(confident, hopeless)


class DialogueTest(unittest.TestCase):
    def test_semicolon_list_becomes_separate_constraints(self) -> None:
        parsed = parse("For that, what matters is: Cotton blend; Machine Wash.", 2)
        self.assertEqual(parsed.constraints, ["cotton blend", "machine wash"])
        self.assertTrue(parsed.marked)

    def test_opening_message_yields_category_and_constraint(self) -> None:
        parsed = parse("I'm looking for Jewelry Necklaces. A key requirement is: Material:alloy.", 1)
        self.assertEqual(parsed.category, "jewelry necklaces")
        self.assertEqual(parsed.constraints, ["material:alloy"])

    def test_refusal_is_detected_and_carries_its_attribute(self) -> None:
        parsed = parse("I don't have a preference for color; please use your judgment.", 3)
        self.assertEqual(parsed.refused_attribute, "color")
        self.assertEqual(parsed.constraints, [])

    def test_override_announcement_is_not_itself_a_constraint(self) -> None:
        parsed = parse("Actually, ignore my earlier preference. What I need is: 100% Cotton.", 3)
        self.assertTrue(parsed.is_override)
        self.assertEqual(parsed.constraints, ["100% cotton"])

    def test_unmarked_paraphrase_still_yields_constraints(self) -> None:
        """No marker survives here; the general path has to carry it."""
        parsed = parse("the composition is 67% polyester and 33% cotton, plus a rubber sole", 2)
        self.assertFalse(parsed.marked)
        self.assertTrue(parsed.constraints)

    def test_stall_carries_no_information(self) -> None:
        parsed = parse("Those options are not quite right yet. Ask me about one specific attribute.", 4)
        self.assertTrue(parsed.is_stall)
        self.assertEqual(parsed.constraints, [])


if __name__ == "__main__":
    unittest.main()
