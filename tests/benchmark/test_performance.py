"""Performance benchmarks for bilancio simulation operations.

Each test is marked with @pytest.mark.slow so that they can be excluded
from the default test suite:

    uv run pytest tests/ -v -m "not slow"        # skip benchmarks
    uv run pytest tests/benchmark/ -v -m slow     # run benchmarks only

Tests use stdlib time.perf_counter() for timing and assert generous upper
bounds to avoid flaky failures while still catching major regressions.
"""

import time
from decimal import Decimal

import pytest

from bilancio.config.apply import apply_to_system
from bilancio.config.loaders import preprocess_config
from bilancio.config.models import RingExplorerGeneratorConfig, ScenarioConfig
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.household import Household
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.engines.simulation import run_day
from bilancio.engines.system import System
from bilancio.scenarios.ring_explorer import compile_ring_explorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_ring_system(n_agents: int, *, seed: int = 42) -> System:
    """Build a ring system with *n_agents* households, ready for simulation."""
    gen_config = RingExplorerGeneratorConfig.model_validate(
        {
            "version": 1,
            "generator": "ring_explorer_v1",
            "name_prefix": "benchmark",
            "params": {
                "n_agents": n_agents,
                "kappa": "1",
                "seed": seed,
                "Q_total": str(100 * n_agents),
                "inequality": {"scheme": "dirichlet", "concentration": "1"},
                "maturity": {"days": 5, "mode": "lead_lag", "mu": "0.5"},
                "liquidity": {"allocation": {"mode": "uniform"}},
            },
            "compile": {"emit_yaml": False},
        }
    )
    scenario_dict = compile_ring_explorer(gen_config, source_path=None)
    scenario_dict = preprocess_config(scenario_dict)
    config = ScenarioConfig(**scenario_dict)
    system = System()
    apply_to_system(config, system)
    return system


def _create_payable(system: System, debtor: str, creditor: str, amount: int, due_day: int) -> str:
    """Create a single payable and add it to the system. Returns the contract id."""
    pid = system.new_contract_id("P")
    payable = Payable(
        id=pid,
        kind=InstrumentKind.PAYABLE,
        amount=amount,
        denom="X",
        asset_holder_id=creditor,
        liability_issuer_id=debtor,
        due_day=due_day,
    )
    system.add_contract(payable)
    return pid


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSettlementThroughput:
    """Measure the time to settle a batch of payables via run_day."""

    def test_settlement_throughput(self, ring_system_10):
        """Create 50 payables due today and time a single run_day settlement."""
        system, _ = ring_system_10

        # Ensure every household has enough cash to cover obligations
        for i in range(1, 11):
            system.mint_cash(f"H{i}", 50_000)

        # Create 50 payables all due on the current day
        current_day = system.state.day
        for idx in range(50):
            debtor = f"H{(idx % 10) + 1}"
            creditor = f"H{((idx + 1) % 10) + 1}"
            _create_payable(system, debtor, creditor, 100, current_day)

        payable_count_before = sum(
            1
            for c in system.state.contracts.values()
            if c.kind == InstrumentKind.PAYABLE and c.due_day == current_day
        )
        assert payable_count_before >= 50, "Expected at least 50 payables before settlement"

        start = time.perf_counter()
        run_day(system)
        elapsed = time.perf_counter() - start

        print(f"\n  settlement_throughput: {payable_count_before} payables settled in {elapsed:.4f}s")

        # Generous bound: should complete well under 5 seconds even on slow CI
        assert elapsed < 5.0, f"Settlement took {elapsed:.2f}s, expected < 5s"


@pytest.mark.slow
class TestRingCreationScaling:
    """Measure how long it takes to compile and apply ring scenarios of varying size."""

    @pytest.mark.parametrize("n_agents", [10, 50, 100])
    def test_ring_creation_scaling(self, n_agents):
        """Time the full ring creation pipeline: compile + apply_to_system."""
        start = time.perf_counter()
        system = _build_ring_system(n_agents)
        elapsed = time.perf_counter() - start

        agent_count = len(system.state.agents)
        contract_count = len(system.state.contracts)

        print(
            f"\n  ring_creation n={n_agents}: "
            f"{agent_count} agents, {contract_count} contracts in {elapsed:.4f}s"
        )

        # Generous upper bounds that scale with agent count
        max_seconds = 2.0 + (n_agents / 50)  # 2s base + linear allowance
        assert elapsed < max_seconds, (
            f"Ring creation for {n_agents} agents took {elapsed:.2f}s, expected < {max_seconds:.1f}s"
        )


@pytest.mark.slow
class TestRunDayThroughput:
    """Measure the throughput of running multiple simulation days."""

    def test_run_day_throughput(self):
        """Run 10 days of simulation and measure total elapsed time.

        Uses expel-agent default mode so that defaults (which are expected
        in a ring with kappa=1) do not raise and halt the simulation.
        """
        # Build a dedicated system with expel-agent mode
        gen_config = RingExplorerGeneratorConfig.model_validate(
            {
                "version": 1,
                "generator": "ring_explorer_v1",
                "name_prefix": "benchmark_days",
                "params": {
                    "n_agents": 10,
                    "kappa": "2",
                    "seed": 42,
                    "Q_total": "1000",
                    "inequality": {"scheme": "dirichlet", "concentration": "1"},
                    "maturity": {"days": 5, "mode": "lead_lag", "mu": "0.5"},
                    "liquidity": {"allocation": {"mode": "uniform"}},
                },
                "compile": {"emit_yaml": False},
            }
        )
        scenario_dict = compile_ring_explorer(gen_config, source_path=None)
        scenario_dict = preprocess_config(scenario_dict)
        config = ScenarioConfig(**scenario_dict)
        system = System(default_mode="expel-agent")
        apply_to_system(config, system)

        n_days = 10
        start = time.perf_counter()
        for _ in range(n_days):
            run_day(system)
        elapsed = time.perf_counter() - start

        per_day = elapsed / n_days
        final_day = system.state.day

        print(
            f"\n  run_day_throughput: {n_days} days in {elapsed:.4f}s "
            f"({per_day:.4f}s/day), final day={final_day}"
        )

        # 10 days should be very fast (sub-second), allow 5s for safety
        assert elapsed < 5.0, f"10 days took {elapsed:.2f}s, expected < 5s"
        assert final_day == n_days, f"Expected final day {n_days}, got {final_day}"


@pytest.mark.slow
class TestContractLookupPerformance:
    """Measure the performance of looking up contracts by due day."""

    def test_contract_lookup_performance(self):
        """Create 100+ contracts across multiple due days and time lookups."""
        system = System()
        cb = CentralBank(id="CB", name="Central Bank", kind="central_bank")
        system.bootstrap_cb(cb)

        # Create 20 households
        households = []
        for i in range(1, 21):
            hh = Household(id=f"H{i}", name=f"Household {i}", kind="household")
            system.add_agent(hh)
            system.mint_cash(f"H{i}", 100_000)
            households.append(hh)

        # Create 120 payables spread across 10 due days (12 per day)
        n_contracts = 120
        n_due_days = 10
        for idx in range(n_contracts):
            debtor = f"H{(idx % 20) + 1}"
            creditor = f"H{((idx + 3) % 20) + 1}"
            due_day = (idx % n_due_days) + 1
            _create_payable(system, debtor, creditor, 50, due_day)

        total_contracts = len(system.state.contracts)
        assert total_contracts >= n_contracts, (
            f"Expected at least {n_contracts} contracts, got {total_contracts}"
        )

        # Time 1000 lookups via contracts_by_due_day index
        n_lookups = 1000
        start = time.perf_counter()
        for i in range(n_lookups):
            day = (i % n_due_days) + 1
            bucket = system.state.contracts_by_due_day.get(day, [])
            # Force iteration to simulate real usage
            _ = [system.state.contracts[cid] for cid in bucket]
        elapsed = time.perf_counter() - start

        per_lookup_us = (elapsed / n_lookups) * 1_000_000

        print(
            f"\n  contract_lookup: {n_lookups} lookups over {total_contracts} contracts "
            f"in {elapsed:.4f}s ({per_lookup_us:.1f} us/lookup)"
        )

        # 1000 dict lookups should be very fast — generous 2s bound
        assert elapsed < 2.0, f"Lookups took {elapsed:.2f}s, expected < 2s"

        # Also time the naive full-scan approach for comparison
        start_naive = time.perf_counter()
        for i in range(n_lookups):
            day = (i % n_due_days) + 1
            _ = [
                c
                for c in system.state.contracts.values()
                if getattr(c, "due_day", None) == day
            ]
        elapsed_naive = time.perf_counter() - start_naive

        per_lookup_naive_us = (elapsed_naive / n_lookups) * 1_000_000

        print(
            f"  contract_lookup (naive scan): {n_lookups} lookups in {elapsed_naive:.4f}s "
            f"({per_lookup_naive_us:.1f} us/lookup)"
        )

        # The indexed approach should be faster than naive scan
        # (Not asserting strict ratio — just documenting the difference)
        if elapsed < elapsed_naive:
            speedup = elapsed_naive / max(elapsed, 1e-9)
            print(f"  Index speedup: {speedup:.1f}x faster than naive scan")
        else:
            print("  Note: Index lookup was not faster (small dataset, overhead dominates)")
