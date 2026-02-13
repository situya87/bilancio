"""Tests for the rating agency: agent, profile, engine, and integration."""

from decimal import Decimal

import pytest

from bilancio.domain.agent import AgentKind
from bilancio.domain.agents import RatingAgency
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.firm import Firm
from bilancio.domain.agents.household import Household
from bilancio.domain.policy import PolicyEngine
from bilancio.decision.profiles import RatingProfile
from bilancio.engines.rating import RatingConfig, run_rating_phase, _compute_rating
from bilancio.engines.system import System
from bilancio.information.presets import RATING_AGENCY_REALISTIC, LENDER_WITH_RATINGS


# ── Helpers ──────────────────────────────────────────────────────────────


def _build_system_with_rating_agency():
    """Build a minimal system with a CB, rating agency, and a few firms."""
    system = System()
    with system.setup():
        cb = CentralBank(id="cb", name="Central Bank", kind="central_bank")
        system.add_agent(cb)
        ra = RatingAgency(id="ra", name="Rating Agency")
        system.add_agent(ra)
        for i in range(3):
            firm = Firm(id=f"f{i}", name=f"Firm {i}", kind="firm")
            system.add_agent(firm)
        # Give each firm some cash
        for i in range(3):
            system.mint_cash(f"f{i}", 100)
    return system


# ── 1. Agent tests ───────────────────────────────────────────────────────


class TestRatingAgencyAgent:
    def test_creation(self):
        ra = RatingAgency(id="ra1", name="My RA")
        assert ra.id == "ra1"
        assert ra.name == "My RA"
        assert ra.kind == AgentKind.RATING_AGENCY
        assert ra.kind == "rating_agency"
        assert ra.asset_ids == []
        assert ra.liability_ids == []
        assert ra.defaulted is False

    def test_kind_not_overridable(self):
        """kind is set by field default, init=False."""
        ra = RatingAgency(id="ra1", name="RA")
        assert ra.kind == AgentKind.RATING_AGENCY

    def test_policy_engine_compatibility(self):
        """PolicyEngine.default() includes RATING_AGENCY in mop_rank."""
        policy = PolicyEngine.default()
        assert AgentKind.RATING_AGENCY in policy.mop_rank
        assert policy.mop_rank[AgentKind.RATING_AGENCY] == []

    def test_settlement_order_empty(self):
        """Rating agency has no settlement activity."""
        policy = PolicyEngine.default()
        ra = RatingAgency(id="ra", name="RA")
        assert list(policy.settlement_order(ra)) == []


# ── 2. RatingProfile tests ──────────────────────────────────────────────


class TestRatingProfile:
    def test_defaults(self):
        p = RatingProfile()
        assert p.lookback_window == 5
        assert p.balance_sheet_weight == Decimal("0.4")
        assert p.history_weight == Decimal("0.6")
        assert p.conservatism_bias == Decimal("0.02")
        assert p.coverage_fraction == Decimal("0.8")
        assert p.no_data_prior == Decimal("0.15")

    def test_frozen_immutability(self):
        p = RatingProfile()
        with pytest.raises(AttributeError):
            p.lookback_window = 10  # type: ignore[misc]

    def test_custom_values(self):
        p = RatingProfile(
            lookback_window=10,
            balance_sheet_weight=Decimal("0.5"),
            history_weight=Decimal("0.5"),
            conservatism_bias=Decimal("0.05"),
            coverage_fraction=Decimal("1.0"),
            no_data_prior=Decimal("0.20"),
        )
        assert p.lookback_window == 10
        assert p.balance_sheet_weight == Decimal("0.5")

    def test_validation_lookback_window(self):
        with pytest.raises(ValueError, match="lookback_window"):
            RatingProfile(lookback_window=0)
        with pytest.raises(ValueError, match="lookback_window"):
            RatingProfile(lookback_window=31)

    def test_validation_balance_sheet_weight(self):
        with pytest.raises(ValueError, match="balance_sheet_weight"):
            RatingProfile(balance_sheet_weight=Decimal("-0.1"))
        with pytest.raises(ValueError, match="balance_sheet_weight"):
            RatingProfile(balance_sheet_weight=Decimal("1.1"))

    def test_validation_conservatism_bias(self):
        with pytest.raises(ValueError, match="conservatism_bias"):
            RatingProfile(conservatism_bias=Decimal("-0.01"))
        with pytest.raises(ValueError, match="conservatism_bias"):
            RatingProfile(conservatism_bias=Decimal("0.3"))

    def test_validation_coverage_fraction(self):
        with pytest.raises(ValueError, match="coverage_fraction"):
            RatingProfile(coverage_fraction=Decimal("0"))
        with pytest.raises(ValueError, match="coverage_fraction"):
            RatingProfile(coverage_fraction=Decimal("1.1"))

    def test_validation_no_data_prior(self):
        with pytest.raises(ValueError, match="no_data_prior"):
            RatingProfile(no_data_prior=Decimal("0"))
        with pytest.raises(ValueError, match="no_data_prior"):
            RatingProfile(no_data_prior=Decimal("1"))


# ── 3. Rating methodology tests ─────────────────────────────────────────


class TestRatingMethodology:
    def test_high_net_worth_low_default(self):
        """Agent with high net worth relative to liabilities gets low p_default."""
        system = _build_system_with_rating_agency()
        profile = RatingProfile()
        # f0 has 100 cash and no liabilities → high coverage → low p_default
        p = _compute_rating(None, "f0", 0, profile, system)
        # Should be low (balance sheet is very healthy)
        assert p < Decimal("0.15")

    def test_negative_net_worth_high_default(self):
        """Agent with negative net worth gets high p_default."""
        system = _build_system_with_rating_agency()
        # Create a large liability for f0
        from bilancio.domain.instruments.credit import Payable
        from bilancio.domain.instruments.base import InstrumentKind
        payable = Payable(
            id="PAY_test",
            kind=InstrumentKind.PAYABLE,
            amount=500,
            denom="X",
            asset_holder_id="f1",
            liability_issuer_id="f0",
            due_day=5,
        )
        system.state.contracts["PAY_test"] = payable
        system.state.agents["f0"].liability_ids.append("PAY_test")
        system.state.agents["f1"].asset_ids.append("PAY_test")

        profile = RatingProfile()
        p = _compute_rating(None, "f0", 0, profile, system)
        # Net worth = 100 - 500 = -400, should be elevated default prob
        assert p > Decimal("0.20")

    def test_conservatism_bias_additive(self):
        """Conservatism bias adds to the rating."""
        system = _build_system_with_rating_agency()
        p_no_bias = _compute_rating(
            None, "f0", 0,
            RatingProfile(conservatism_bias=Decimal("0")),
            system,
        )
        p_with_bias = _compute_rating(
            None, "f0", 0,
            RatingProfile(conservatism_bias=Decimal("0.10")),
            system,
        )
        assert p_with_bias > p_no_bias
        # The difference should be approximately the bias
        diff = p_with_bias - p_no_bias
        assert Decimal("0.05") <= diff <= Decimal("0.15")

    def test_clamping(self):
        """Ratings are clamped to [0.01, 0.99]."""
        system = _build_system_with_rating_agency()
        profile = RatingProfile(conservatism_bias=Decimal("0"))
        p = _compute_rating(None, "f0", 0, profile, system)
        assert Decimal("0.01") <= p <= Decimal("0.99")


# ── 4. run_rating_phase tests ───────────────────────────────────────────


class TestRunRatingPhase:
    def test_publishes_to_registry(self):
        """run_rating_phase populates the rating_registry."""
        system = _build_system_with_rating_agency()
        config = RatingConfig(
            rating_profile=RatingProfile(coverage_fraction=Decimal("1.0"))
        )
        events = run_rating_phase(system, 0, config)
        assert len(events) == 1
        assert events[0]["kind"] == "RatingsPublished"
        assert events[0]["n_rated"] == 3
        assert events[0]["n_eligible"] == 3
        # Registry should have entries for all firms
        assert "f0" in system.state.rating_registry
        assert "f1" in system.state.rating_registry
        assert "f2" in system.state.rating_registry

    def test_coverage_fraction_limits(self):
        """With coverage < 1, not all agents are rated."""
        system = _build_system_with_rating_agency()
        # Add more agents so coverage fraction makes a visible difference
        with system.setup():
            for i in range(3, 10):
                system.add_agent(Firm(id=f"f{i}", name=f"Firm {i}", kind="firm"))
                system.mint_cash(f"f{i}", 100)

        config = RatingConfig(
            rating_profile=RatingProfile(coverage_fraction=Decimal("0.3"))
        )
        events = run_rating_phase(system, 0, config)
        assert events[0]["n_rated"] < events[0]["n_eligible"]

    def test_defaulted_agents_rated_one(self):
        """Defaulted agents get p_default=1.0 in registry."""
        system = _build_system_with_rating_agency()
        system.state.agents["f0"].defaulted = True
        system.state.defaulted_agent_ids.add("f0")

        config = RatingConfig(
            rating_profile=RatingProfile(coverage_fraction=Decimal("1.0"))
        )
        events = run_rating_phase(system, 0, config)
        assert system.state.rating_registry.get("f0") == Decimal("1.0")

    def test_no_agency_returns_empty(self):
        """Without a rating agency, no events are produced."""
        system = System()
        with system.setup():
            cb = CentralBank(id="cb", name="CB", kind="central_bank")
            system.add_agent(cb)
            system.add_agent(Firm(id="f0", name="Firm 0", kind="firm"))
        events = run_rating_phase(system, 0, RatingConfig())
        assert events == []

    def test_no_eligible_agents_returns_empty(self):
        """With only CB and rating agency, no events produced."""
        system = System()
        with system.setup():
            system.add_agent(CentralBank(id="cb", name="CB", kind="central_bank"))
            system.add_agent(RatingAgency(id="ra", name="RA"))
        events = run_rating_phase(system, 0, RatingConfig())
        assert events == []

    def test_carry_forward(self):
        """Unrated agents keep their previous rating in the registry."""
        system = _build_system_with_rating_agency()
        # Day 0: rate all with coverage=1.0
        config_full = RatingConfig(
            rating_profile=RatingProfile(coverage_fraction=Decimal("1.0"))
        )
        run_rating_phase(system, 0, config_full)
        old_rating = system.state.rating_registry.get("f0")
        assert old_rating is not None

        # Day 1: rate only a subset (coverage=0.5 with 3 agents → at least 1)
        # Manually set a low coverage to potentially skip f0
        config_partial = RatingConfig(
            rating_profile=RatingProfile(coverage_fraction=Decimal("0.4"))
        )
        run_rating_phase(system, 1, config_partial)
        # The old rating should still exist
        assert "f0" in system.state.rating_registry


# ── 5. InformationService integration ────────────────────────────────────


class TestInformationServiceIntegration:
    def test_raw_default_probs_uses_rating_registry(self):
        """_raw_default_probs returns registry values when populated."""
        system = _build_system_with_rating_agency()
        system.state.rating_registry = {
            "f0": Decimal("0.10"),
            "f1": Decimal("0.20"),
            "f2": Decimal("0.30"),
        }

        from bilancio.information.service import InformationService
        from bilancio.information.profile import InformationProfile
        info = InformationService(system, InformationProfile(), observer_id="ra")
        probs = info._raw_default_probs(0)
        assert probs["f0"] == Decimal("0.10")
        assert probs["f1"] == Decimal("0.20")
        assert probs["f2"] == Decimal("0.30")

    def test_raw_default_probs_fallback_without_registry(self):
        """Without rating_registry, falls back to heuristic."""
        system = _build_system_with_rating_agency()
        # Empty registry → fallback
        system.state.rating_registry = {}

        from bilancio.information.service import InformationService
        from bilancio.information.profile import InformationProfile
        info = InformationService(system, InformationProfile(), observer_id="ra")
        probs = info._raw_default_probs(0)
        # Should still have values (from fallback heuristic)
        assert "f0" in probs
        assert "cb" in probs

    def test_defaulted_agent_in_registry(self):
        """Defaulted agents get 1.0 even with rating registry."""
        system = _build_system_with_rating_agency()
        system.state.agents["f0"].defaulted = True
        system.state.rating_registry = {
            "f0": Decimal("0.05"),  # stale rating
            "f1": Decimal("0.10"),
        }

        from bilancio.information.service import InformationService
        from bilancio.information.profile import InformationProfile
        info = InformationService(system, InformationProfile(), observer_id="ra")
        probs = info._raw_default_probs(0)
        assert probs["f0"] == Decimal("1.0")  # defaulted overrides


# ── 6. Lending integration ──────────────────────────────────────────────


class TestLendingIntegration:
    def test_estimate_default_probs_uses_registry(self):
        """Lending engine's _estimate_default_probs uses rating_registry."""
        system = _build_system_with_rating_agency()
        system.state.rating_registry = {
            "f0": Decimal("0.12"),
            "f1": Decimal("0.22"),
        }

        from bilancio.engines.lending import _estimate_default_probs
        probs = _estimate_default_probs(system, 0)
        assert probs["f0"] == Decimal("0.12")
        assert probs["f1"] == Decimal("0.22")
        # f2 not in registry → default 0.15
        assert probs["f2"] == Decimal("0.15")

    def test_estimate_default_probs_fallback_empty_registry(self):
        """Empty registry falls through to heuristic."""
        system = _build_system_with_rating_agency()
        system.state.rating_registry = {}

        from bilancio.engines.lending import _estimate_default_probs
        probs = _estimate_default_probs(system, 0)
        assert "f0" in probs


# ── 7. Simulation integration ───────────────────────────────────────────


class TestSimulationIntegration:
    def test_run_day_with_enable_rating(self):
        """run_day with enable_rating logs SubphaseB_Rating."""
        system = _build_system_with_rating_agency()
        system.state.rating_config = RatingConfig(
            rating_profile=RatingProfile(coverage_fraction=Decimal("1.0"))
        )

        from bilancio.engines.simulation import run_day
        run_day(system, enable_rating=True)

        event_kinds = [e["kind"] for e in system.state.events]
        assert "SubphaseB_Rating" in event_kinds
        assert "RatingsPublished" in event_kinds

    def test_run_day_rating_before_dealer(self):
        """SubphaseB_Rating runs before SubphaseB_Dealer."""
        system = _build_system_with_rating_agency()
        system.state.rating_config = RatingConfig()

        from bilancio.engines.simulation import run_day
        run_day(system, enable_rating=True)

        event_kinds = [e["kind"] for e in system.state.events]
        rating_idx = event_kinds.index("SubphaseB_Rating")
        # B2 (settlement) comes after
        b2_idx = event_kinds.index("SubphaseB2")
        assert rating_idx < b2_idx

    def test_run_day_no_rating_without_config(self):
        """Without rating_config, no rating phase runs even if enable_rating=True."""
        system = _build_system_with_rating_agency()
        # rating_config is None by default

        from bilancio.engines.simulation import run_day
        run_day(system, enable_rating=True)

        event_kinds = [e["kind"] for e in system.state.events]
        assert "SubphaseB_Rating" not in event_kinds


# ── 8. Config tests ─────────────────────────────────────────────────────


class TestConfig:
    def test_rating_agency_scenario_config_defaults(self):
        from bilancio.config.models import RatingAgencyScenarioConfig
        cfg = RatingAgencyScenarioConfig()
        assert cfg.enabled is False
        assert cfg.lookback_window == 5
        assert cfg.balance_sheet_weight == Decimal("0.4")
        assert cfg.info_profile == "realistic"

    def test_scenario_config_accepts_rating_agency(self):
        from bilancio.config.models import ScenarioConfig, RatingAgencyScenarioConfig
        cfg = ScenarioConfig(
            name="test",
            agents=[{"id": "cb", "kind": "central_bank", "name": "CB"}],
            rating_agency=RatingAgencyScenarioConfig(enabled=True),
        )
        assert cfg.rating_agency is not None
        assert cfg.rating_agency.enabled is True

    def test_agent_spec_accepts_rating_agency_kind(self):
        from bilancio.config.models import AgentSpec
        spec = AgentSpec(id="ra", kind="rating_agency", name="Rating Agency")
        assert spec.kind == "rating_agency"

    def test_create_agent_rating_agency(self):
        from bilancio.config.models import AgentSpec
        from bilancio.config.apply import create_agent
        spec = AgentSpec(id="ra", kind="rating_agency", name="Rating Agency")
        agent = create_agent(spec)
        assert isinstance(agent, RatingAgency)
        assert agent.kind == AgentKind.RATING_AGENCY


# ── 9. Preset tests ────────────────────────────────────────────────────


class TestPresets:
    def test_rating_agency_realistic_no_market(self):
        """RATING_AGENCY_REALISTIC has no market price access."""
        from bilancio.information.levels import AccessLevel
        assert RATING_AGENCY_REALISTIC.dealer_quotes.level == AccessLevel.NONE
        assert RATING_AGENCY_REALISTIC.vbt_anchors.level == AccessLevel.NONE

    def test_rating_agency_realistic_has_balance_sheet(self):
        """RATING_AGENCY_REALISTIC has noisy balance sheet access."""
        from bilancio.information.levels import AccessLevel
        assert RATING_AGENCY_REALISTIC.counterparty_cash.level == AccessLevel.NOISY
        assert RATING_AGENCY_REALISTIC.counterparty_net_worth.level == AccessLevel.NOISY

    def test_lender_with_ratings_institutional_channel(self):
        """LENDER_WITH_RATINGS uses institutional channel for default history."""
        from bilancio.information.levels import AccessLevel
        from bilancio.information.noise import SampleNoise
        access = LENDER_WITH_RATINGS.counterparty_default_history
        assert access.level == AccessLevel.NOISY
        # InstitutionalChannel(staleness_days=1, coverage=0.8) → SampleNoise(0.8)
        # because coverage_gap=0.2 > lag_error=0.05
        assert isinstance(access.noise, SampleNoise)
        assert access.noise.sample_rate == Decimal("0.8")
