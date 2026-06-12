"""Ring compiler integer-face quantization.

The v2 kernel truncates financial amounts to whole units at action parse
time, so the ring-explorer compiler quantizes generated payable faces and
liquidity with largest-remainder rounding: ledger totals match Q_total and
the liquidity spec exactly, and no zero-face payables sever the ring cycle.
"""

from decimal import Decimal

from bilancio.config.models import RingExplorerGeneratorConfig
from bilancio.scenarios.ring.compiler import (
    _quantize_to_integers,
    compile_ring_explorer,
)


class TestQuantizeToIntegers:
    def test_preserves_integer_total(self):
        amounts = [Decimal("5.0689"), Decimal("12.6703"), Decimal("0.469"), Decimal("231.79")]
        out = _quantize_to_integers(amounts)
        assert sum(out) == Decimal(int(sum(amounts)))
        assert all(v == v.to_integral_value() for v in out)

    def test_largest_remainder_gets_the_leftover_unit(self):
        out = _quantize_to_integers([Decimal("1.2"), Decimal("1.9"), Decimal("1.9")])
        assert out == [Decimal(1), Decimal(2), Decimal(2)]

    def test_min_one_lifts_zero_entries(self):
        out = _quantize_to_integers(
            [Decimal("0.1"), Decimal("0.2"), Decimal("99.7")], min_one=True
        )
        assert sum(out) == Decimal(100)
        assert min(out) >= Decimal(1)

    def test_min_one_skipped_when_total_too_small(self):
        out = _quantize_to_integers([Decimal("0.5"), Decimal("0.5"), Decimal("1")], min_one=True)
        assert sum(out) == Decimal(2)

    def test_deterministic(self):
        amounts = [Decimal("3.5"), Decimal("2.5"), Decimal("4.0")]
        assert _quantize_to_integers(amounts) == _quantize_to_integers(amounts)

    def test_empty(self):
        assert _quantize_to_integers([]) == []


class TestCompiledScenarioIsIntegerQuantized:
    def _compile(self, **param_overrides):
        params = {
            "n_agents": 20,
            "seed": 42,
            "kappa": "0.5",
            "Q_total": "250",
            "liquidity": {"allocation": {"mode": "uniform"}},
            "inequality": {"scheme": "dirichlet", "concentration": "0.5"},
            "maturity": {"days": 5, "mode": "lead_lag", "mu": "0"},
        }
        params.update(param_overrides)
        config = RingExplorerGeneratorConfig.model_validate(
            {
                "version": 1,
                "generator": "ring_explorer_v1",
                "name_prefix": "Quantization Test",
                "params": params,
                "compile": {"emit_yaml": False},
            }
        )
        return compile_ring_explorer(config)

    def test_payable_faces_are_positive_integers_summing_to_q_total(self):
        scenario = self._compile()
        faces = [
            Decimal(str(a["create_payable"]["amount"]))
            for a in scenario["initial_actions"]
            if "create_payable" in a
        ]
        assert len(faces) == 20
        assert all(f == f.to_integral_value() for f in faces)
        assert all(f >= 1 for f in faces)
        assert sum(faces) == Decimal(250)

    def test_liquidity_is_integer_and_sums_to_spec(self):
        scenario = self._compile()
        cash = [
            Decimal(str(a["mint_cash"]["amount"]))
            for a in scenario["initial_actions"]
            if "mint_cash" in a
        ]
        assert all(c == c.to_integral_value() for c in cash)
        # kappa=0.5 on Q=250 -> L=125
        assert sum(cash) == Decimal(125)
