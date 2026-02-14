"""Tests for channel-binding-driven source resolution.

Covers:
1. ChannelBinding dataclass construction
2. InformationProfile with channel_bindings field
3. InformationService._raw_default_probs() follows bindings
4. InformationService._raw_default_probs() without bindings uses legacy waterfall
5. InformationService.get_default_probability_detail() provenance from bindings
6. lending._estimate_default_probs() with bindings
7. lending._estimate_default_probs() without bindings (backward compat)
8. LENDER_RATINGS_BOUND preset works correctly
"""

from decimal import Decimal

import pytest

from bilancio.information import (
    LENDER_RATINGS_BOUND,
    OMNISCIENT,
    ChannelBinding,
    InformationProfile,
    InformationService,
    InstitutionalChannel,
    SelfDerivedChannel,
)


# ── Helpers ────────────────────────────────────────────────────────


def _build_system(with_rating_registry=False):
    """Build a minimal system for testing."""
    from bilancio.domain.agents.central_bank import CentralBank
    from bilancio.domain.agents.firm import Firm
    from bilancio.domain.agents.non_bank_lender import NonBankLender
    from bilancio.domain.agents.bank import Bank
    from bilancio.engines.system import System

    system = System()
    cb = CentralBank(id="CB01", name="CB", kind="central_bank")
    bank = Bank(id="B01", name="Bank", kind="bank")
    lender = NonBankLender(id="NBL01", name="Lender")
    firm1 = Firm(id="F01", name="Firm 1", kind="firm")
    firm2 = Firm(id="F02", name="Firm 2", kind="firm")

    system.bootstrap_cb(cb)
    system.add_agent(bank)
    system.add_agent(lender)
    system.add_agent(firm1)
    system.add_agent(firm2)

    system.mint_cash("NBL01", 10000)
    system.mint_cash("F01", 500)

    if with_rating_registry:
        system.state.rating_registry = {
            "F01": Decimal("0.10"),
            "F02": Decimal("0.20"),
        }

    return system


# ── 1. ChannelBinding construction ─────────────────────────────────


class TestChannelBinding:
    def test_basic_construction(self):
        b = ChannelBinding(
            "default_prob", "rating_registry",
            InstitutionalChannel(staleness_days=1, coverage=Decimal("0.8")),
            priority=0,
        )
        assert b.category == "default_prob"
        assert b.source == "rating_registry"
        assert b.priority == 0

    def test_default_priority(self):
        b = ChannelBinding(
            "default_prob", "system_heuristic",
            SelfDerivedChannel(sample_size=10),
        )
        assert b.priority == 0

    def test_frozen(self):
        b = ChannelBinding(
            "default_prob", "rating_registry",
            InstitutionalChannel(),
        )
        with pytest.raises(AttributeError):
            b.source = "other"  # type: ignore[misc]


# ── 2. InformationProfile with channel_bindings ────────────────────


class TestProfileChannelBindings:
    def test_default_empty(self):
        p = InformationProfile()
        assert p.channel_bindings == ()

    def test_with_bindings(self):
        bindings = (
            ChannelBinding(
                "default_prob", "rating_registry",
                InstitutionalChannel(), priority=0,
            ),
        )
        p = InformationProfile(channel_bindings=bindings)
        assert len(p.channel_bindings) == 1
        assert p.channel_bindings[0].source == "rating_registry"

    def test_from_hierarchy_passes_bindings(self):
        bindings = (
            ChannelBinding(
                "default_prob", "system_heuristic",
                SelfDerivedChannel(), priority=0,
            ),
        )
        p = InformationProfile.from_hierarchy(channel_bindings=bindings)
        assert p.channel_bindings == bindings

    def test_from_hierarchy_default_empty(self):
        p = InformationProfile.from_hierarchy()
        assert p.channel_bindings == ()


# ── 3. Service follows bindings ────────────────────────────────────


class TestServiceBindings:
    def test_registry_binding_uses_registry(self):
        """With binding for rating_registry, service uses registry even though
        no dealer risk assessor exists."""
        system = _build_system(with_rating_registry=True)
        profile = InformationProfile(
            channel_bindings=(
                ChannelBinding(
                    "default_prob", "rating_registry",
                    InstitutionalChannel(), priority=0,
                ),
            ),
        )
        info = InformationService(system, profile, observer_id="NBL01")
        p = info.get_default_probability("F01", 0)
        # Registry has F01 = 0.10
        assert p == Decimal("0.10")

    def test_binding_skips_unavailable_source(self):
        """Binding for dealer_risk_assessor that isn't available falls through
        to next binding."""
        system = _build_system(with_rating_registry=True)
        profile = InformationProfile(
            channel_bindings=(
                ChannelBinding(
                    "default_prob", "dealer_risk_assessor",
                    SelfDerivedChannel(), priority=0,
                ),
                ChannelBinding(
                    "default_prob", "rating_registry",
                    InstitutionalChannel(), priority=1,
                ),
            ),
        )
        info = InformationService(system, profile, observer_id="NBL01")
        p = info.get_default_probability("F01", 0)
        # No dealer → falls through to registry → 0.10
        assert p == Decimal("0.10")

    def test_binding_all_unavailable_falls_to_heuristic(self):
        """When all bound sources are unavailable, falls back to heuristic."""
        system = _build_system(with_rating_registry=False)
        # Bind only dealer (unavailable) and registry (empty)
        profile = InformationProfile(
            channel_bindings=(
                ChannelBinding(
                    "default_prob", "dealer_risk_assessor",
                    SelfDerivedChannel(), priority=0,
                ),
                ChannelBinding(
                    "default_prob", "rating_registry",
                    InstitutionalChannel(), priority=1,
                ),
            ),
        )
        info = InformationService(system, profile, observer_id="NBL01")
        p = info.get_default_probability("F01", 0)
        # Both unavailable → heuristic: base_rate(0) + 0.05 = 0.05
        assert p == Decimal("0.05")

    def test_heuristic_binding_always_works(self):
        """System heuristic binding always succeeds."""
        system = _build_system(with_rating_registry=True)
        profile = InformationProfile(
            channel_bindings=(
                ChannelBinding(
                    "default_prob", "system_heuristic",
                    SelfDerivedChannel(), priority=0,
                ),
            ),
        )
        info = InformationService(system, profile, observer_id="NBL01")
        p = info.get_default_probability("F01", 0)
        # Heuristic: base_rate(0) + 0.05 = 0.05
        assert p == Decimal("0.05")

    def test_binding_priority_order(self):
        """Lower priority number is tried first."""
        system = _build_system(with_rating_registry=True)
        # Heuristic first (priority=0), registry second (priority=1)
        profile = InformationProfile(
            channel_bindings=(
                ChannelBinding(
                    "default_prob", "rating_registry",
                    InstitutionalChannel(), priority=1,
                ),
                ChannelBinding(
                    "default_prob", "system_heuristic",
                    SelfDerivedChannel(), priority=0,
                ),
            ),
        )
        info = InformationService(system, profile, observer_id="NBL01")
        p = info.get_default_probability("F01", 0)
        # Heuristic is tried first (priority 0) → 0.05
        assert p == Decimal("0.05")

    def test_unknown_source_skipped(self):
        """Unknown source names are skipped, falling through to next binding."""
        system = _build_system(with_rating_registry=True)
        profile = InformationProfile(
            channel_bindings=(
                ChannelBinding(
                    "default_prob", "nonexistent_source",
                    SelfDerivedChannel(), priority=0,
                ),
                ChannelBinding(
                    "default_prob", "rating_registry",
                    InstitutionalChannel(), priority=1,
                ),
            ),
        )
        info = InformationService(system, profile, observer_id="NBL01")
        p = info.get_default_probability("F01", 0)
        # Unknown source skipped → falls to registry → 0.10
        assert p == Decimal("0.10")


# ── 4. Service without bindings uses legacy waterfall ──────────────


class TestServiceLegacyWaterfall:
    def test_no_bindings_uses_registry_when_available(self):
        """Without bindings, legacy waterfall uses registry (no dealer present)."""
        system = _build_system(with_rating_registry=True)
        info = InformationService(system, OMNISCIENT, observer_id="NBL01")
        p = info.get_default_probability("F01", 0)
        assert p == Decimal("0.10")

    def test_no_bindings_heuristic_when_no_sources(self):
        """Without bindings and no sources, falls to heuristic."""
        system = _build_system(with_rating_registry=False)
        info = InformationService(system, OMNISCIENT, observer_id="NBL01")
        p = info.get_default_probability("F01", 0)
        # Heuristic: 0 defaults / 5 agents → base=0 → 0.05
        assert p == Decimal("0.05")


# ── 5. Provenance from bindings ────────────────────────────────────


class TestProvenanceFromBindings:
    def test_detail_channel_source_from_binding(self):
        """get_default_probability_detail() reports correct channel_source."""
        system = _build_system(with_rating_registry=True)
        profile = InformationProfile(
            channel_bindings=(
                ChannelBinding(
                    "default_prob", "rating_registry",
                    InstitutionalChannel(), priority=0,
                ),
            ),
        )
        info = InformationService(system, profile, observer_id="NBL01")
        est = info.get_default_probability_detail("F01", 0)
        assert est is not None
        assert est.inputs["channel_source"] == "rating_registry"

    def test_detail_channel_source_legacy(self):
        """Without bindings, channel_source comes from legacy waterfall."""
        system = _build_system(with_rating_registry=True)
        info = InformationService(system, OMNISCIENT, observer_id="NBL01")
        est = info.get_default_probability_detail("F01", 0)
        assert est is not None
        assert est.inputs["channel_source"] == "rating_registry"

    def test_detail_channel_source_heuristic_fallback(self):
        """When no sources available, reports system_heuristic."""
        system = _build_system(with_rating_registry=False)
        info = InformationService(system, OMNISCIENT, observer_id="NBL01")
        est = info.get_default_probability_detail("F01", 0)
        assert est is not None
        assert est.inputs["channel_source"] == "system_heuristic"


# ── 6. Lending with bindings ──────────────────────────────────────


class TestLendingBindings:
    def test_with_bindings_uses_registry(self):
        from bilancio.engines.lending import _estimate_default_probs

        system = _build_system(with_rating_registry=True)
        bindings = (
            ChannelBinding(
                "default_prob", "rating_registry",
                InstitutionalChannel(), priority=0,
            ),
        )
        probs = _estimate_default_probs(system, 0, channel_bindings=bindings)
        assert probs["F01"] == Decimal("0.10")
        assert probs["F02"] == Decimal("0.20")

    def test_with_bindings_skips_unavailable(self):
        from bilancio.engines.lending import _estimate_default_probs

        system = _build_system(with_rating_registry=True)
        bindings = (
            ChannelBinding(
                "default_prob", "dealer_risk_assessor",
                SelfDerivedChannel(), priority=0,
            ),
            ChannelBinding(
                "default_prob", "rating_registry",
                InstitutionalChannel(), priority=1,
            ),
        )
        probs = _estimate_default_probs(system, 0, channel_bindings=bindings)
        assert probs["F01"] == Decimal("0.10")


# ── 7. Lending backward compatibility ─────────────────────────────


class TestLendingBackwardCompat:
    def test_no_bindings_uses_legacy_waterfall(self):
        from bilancio.engines.lending import _estimate_default_probs

        system = _build_system(with_rating_registry=True)
        probs = _estimate_default_probs(system, 0)
        # Legacy waterfall: no dealer → registry → F01=0.10
        assert probs["F01"] == Decimal("0.10")

    def test_no_bindings_no_sources_uses_heuristic(self):
        from bilancio.engines.lending import _estimate_default_probs

        system = _build_system(with_rating_registry=False)
        probs = _estimate_default_probs(system, 0)
        # Heuristic: 0.05
        assert probs["F01"] == Decimal("0.05")


# ── 8. LENDER_RATINGS_BOUND preset ────────────────────────────────


class TestLenderRatingsBoundPreset:
    def test_has_bindings(self):
        assert len(LENDER_RATINGS_BOUND.channel_bindings) == 2

    def test_first_binding_is_registry(self):
        first = LENDER_RATINGS_BOUND.channel_bindings[0]
        assert first.source == "rating_registry"
        assert first.priority == 0

    def test_second_binding_is_heuristic(self):
        second = LENDER_RATINGS_BOUND.channel_bindings[1]
        assert second.source == "system_heuristic"
        assert second.priority == 1

    def test_uses_registry_over_dealer(self):
        """Even if a dealer were present, this preset uses registry."""
        system = _build_system(with_rating_registry=True)
        info = InformationService(
            system, LENDER_RATINGS_BOUND, observer_id="NBL01",
        )
        p = info.get_default_probability("F01", 0)
        # Binding says: registry first → raw 0.10, then SampleNoise(0.8)
        # scales it: 0.10 * 0.8 = 0.08
        assert p == Decimal("0.08")

    def test_falls_to_heuristic_without_registry(self):
        """Without registry, falls to system_heuristic (second binding)."""
        system = _build_system(with_rating_registry=False)
        info = InformationService(
            system, LENDER_RATINGS_BOUND, observer_id="NBL01",
        )
        p = info.get_default_probability("F01", 0)
        # Heuristic → raw 0.05, then SampleNoise(0.8) → 0.05 * 0.8 = 0.04
        assert p == Decimal("0.04")
