"""Tests for the generic (scenario-agnostic) sampling module."""

from __future__ import annotations

from decimal import Decimal

import pytest

from bilancio.experiments.sampling.generic import (
    generate_grid_generic,
    generate_lhs_generic,
)


class TestGenerateGridGeneric:
    def test_single_dimension(self):
        results = list(generate_grid_generic({"a": [Decimal("1"), Decimal("2"), Decimal("3")]}))
        assert len(results) == 3
        assert results[0] == {"a": Decimal("1")}
        assert results[2] == {"a": Decimal("3")}

    def test_two_dimensions(self):
        results = list(
            generate_grid_generic(
                {
                    "x": [Decimal("1"), Decimal("2")],
                    "y": [Decimal("10"), Decimal("20")],
                }
            )
        )
        assert len(results) == 4  # 2 × 2
        assert {"x": Decimal("1"), "y": Decimal("10")} in results
        assert {"x": Decimal("2"), "y": Decimal("20")} in results

    def test_three_dimensions(self):
        results = list(
            generate_grid_generic(
                {
                    "a": [Decimal("1"), Decimal("2")],
                    "b": [Decimal("3")],
                    "c": [Decimal("4"), Decimal("5"), Decimal("6")],
                }
            )
        )
        assert len(results) == 2 * 1 * 3  # 6

    def test_empty_dimensions(self):
        results = list(generate_grid_generic({}))
        assert results == []

    def test_preserves_dimension_names(self):
        results = list(
            generate_grid_generic(
                {"kappa": [Decimal("0.5")], "concentration": [Decimal("1")]}
            )
        )
        assert len(results) == 1
        assert "kappa" in results[0]
        assert "concentration" in results[0]


class TestGenerateLhsGeneric:
    def test_correct_count(self):
        results = list(
            generate_lhs_generic(
                10,
                {"x": (Decimal("0"), Decimal("1")), "y": (Decimal("0"), Decimal("10"))},
                seed=42,
            )
        )
        assert len(results) == 10

    def test_values_in_range(self):
        results = list(
            generate_lhs_generic(
                50,
                {
                    "x": (Decimal("0"), Decimal("1")),
                    "y": (Decimal("5"), Decimal("10")),
                },
                seed=123,
            )
        )
        for r in results:
            assert Decimal("0") <= r["x"] <= Decimal("1")
            assert Decimal("5") <= r["y"] <= Decimal("10")

    def test_dimension_names_preserved(self):
        results = list(
            generate_lhs_generic(
                3,
                {"kappa": (Decimal("0.1"), Decimal("5"))},
                seed=42,
            )
        )
        assert all("kappa" in r for r in results)

    def test_zero_count(self):
        results = list(
            generate_lhs_generic(0, {"x": (Decimal("0"), Decimal("1"))}, seed=42)
        )
        assert results == []

    def test_negative_count(self):
        results = list(
            generate_lhs_generic(-1, {"x": (Decimal("0"), Decimal("1"))}, seed=42)
        )
        assert results == []

    def test_reproducible(self):
        dims = {"a": (Decimal("0"), Decimal("10")), "b": (Decimal("0"), Decimal("1"))}
        r1 = list(generate_lhs_generic(5, dims, seed=42))
        r2 = list(generate_lhs_generic(5, dims, seed=42))
        assert r1 == r2

    def test_different_seeds_differ(self):
        dims = {"a": (Decimal("0"), Decimal("10"))}
        r1 = list(generate_lhs_generic(10, dims, seed=1))
        r2 = list(generate_lhs_generic(10, dims, seed=2))
        assert r1 != r2
