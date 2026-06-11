# Plan 060: Clearinghouse Stage 2 — Clearinghouse Loan Certificates

**Status**: Plan only — not yet implemented · **Date**: 2026-06-11 · **Branch**: `plan/059-clearinghouse-suite`
**Depends on**: Plan 059 (clearinghouse netting / `ClearingPhase` scaffolding) — see "Dependencies".

## Goal

In the no-bank / no-dealer Kalecki ring, the only means of payment is outside cash; at low
kappa with skewed maturities, agents default on dues today while holding good receivables due
tomorrow. Stage 2 of the clearinghouse suite adds **clearinghouse loan certificates**, modeled
on the 19th-century US clearinghouse association instrument (1857–1907 panics): a member short
of cash pledges receivables to the clearinghouse, which issues haircut, interest-bearing bearer
certificates that other members must accept in settlement — **endogenous payment elasticity
against the ring's own assets**, bridging temporal mismatch without banks, dealers, or outside
liquidity. Hypothesis: certificates substantially cut `delta_total` precisely in the (low κ,
high μ-skew) region where Plan 059's multilateral netting is weakest.

## Dependencies

1. **Plan 059 merged.** This plan builds on 059's `clearinghouse:` scenario block and the
   `ClearingPhase` plugin (multilateral netting of due-today payables). 059 exists only as a
   sibling plan doc (`docs/plans/059_clearinghouse_netting.md`) — no code yet; nothing named
   clearinghouse exists in `src/` (the legacy v1 interbank DNS clearing in
   `src/bilancio/engines/clearing.py` is unrelated). The interface below mirrors the 059 draft;
   re-verify after 059 lands. Netting runs first; certificates cover the **post-netting** shortfall.
2. **Roadmap WP-1 sliver.** Pledging a receivable at a haircut is a thin slice of the collateral
   framework in `docs/plans/056_master_development_roadmap.md` WP-1 (collateral link on
   instruments, haircut valuation). This plan deliberately consumes only that sliver — a
   `pledged_to` marker on `Payable` plus a fixed flat haircut — **without** the general
   `CollateralValuer`, margin calls, or rehypothecation. `pledged_to` is the same field WP-1
   specifies, so the facility migrates cleanly when WP-1 lands.

### 059 interface this plan extends (per the 059 draft; verify after merge)

- `ScenarioConfig.clearinghouse: ClearinghouseConfig | None` with `enabled: bool = False`,
  `mode: Literal["netting"]` — this plan widens the Literal with `"certificates"`.
- `bilancio_v2/plugins/clearing.py::ClearingPhase` (name `SubphaseB_Clearing`), inserted in
  `engine.prepare_scenario` immediately before `SettlementPhase`; nets due-today payables by
  reducing `Payable` amounts (`Payable.netted_amount`, `ledger.netted_rollover_queue`).
- `scenario_gates.build_clearing_config(config) -> CleanClearingConfig | None`; this plan adds
  certificate fields to `CleanClearingConfig`.
- ring sweep `--clearing/--no-clearing` flag; `netting_efficiency` results column.
- 059 introduces **no** agent (netting needs no balance sheet); the `clearinghouse` kind and
  CH1 are new here. Post-netting shortfall is read directly from the surviving due-today
  payable amounts — no separate net-position API is required.

## Scope

**IN**
- New agent kind `clearinghouse` (single instance, id `CH1`), auto-registered when
  `clearinghouse.mode == "certificates"`.
- New instrument (bearer certificate balance) + new contract record `CertificatePledge`.
- New phase `CertificateFacilityPhase` between `ClearingPhase` and `SettlementPhase`.
- Means-of-payment extension: certificates rank **below cash** for `household`/`firm`;
  acceptance mandatory.
- Redemption, interest accrual, recourse-on-collateral-default, terminal waterfall.
- Ring sweep flag `--clearing-certificates`; metrics `certificates_issued_total`,
  `certificates_outstanding_peak`, `cert_default_losses`.
- Unit + integration + golden tests; all defaults OFF (backward compatible).

**OUT**
- Voluntary acceptance variant (parameter reserved; OPEN QUESTION 2).
- CH initial capital / member subscription fund (CH issues against collateral only; its equity
  buffer is accrued interest).
- General WP-1 collateral framework (valuation, margin calls, rehypothecation).
- Certificates for bank/dealer/lender arms (firms/households in the pure ring only).
- Secondary-market pricing of certificates (they pass at par by rule).
- v1 legacy engine support — v2 kernel only, consistent with the post-#168 cutover.

## Location in codebase

| Change type | Path | Description |
|-------------|------|-------------|
| New file | `src/bilancio_v2/plugins/certificates.py` | `CertificateFacilityPhase`: issuance, redemption, interest, recourse |
| Modified | `src/bilancio_v2/plugins/clearing.py` (from 059) | expose post-netting per-member net dues for the facility |
| Modified | `src/bilancio_v2/ledger.py` | `certificates` balance map, `CertificatePledge` record, certificate ops + conservation invariant |
| Modified | `src/bilancio_v2/policy.py` | `MOP_CLEARINGHOUSE_CERT`; `clearinghouse` row; `with_certificate_mop()` extension |
| Modified | `src/bilancio_v2/plugins/settlement.py` | certificate leg in `settle_payable` mop loop; include certificates in default recovery/liquid assets |
| Modified | `src/bilancio_v2/engine.py` | register CH1; insert `CertificateFacilityPhase` after `ClearingPhase` |
| Modified | `src/bilancio_v2/scenario_gates.py` | `build_clearing_config` (059) extended for `mode: certificates` + profile params |
| Modified | `src/bilancio_v2/subsystem_config.py` | certificate fields on `CleanClearingConfig` (059) |
| Modified | `src/bilancio/config/models.py` | `ClearinghouseConfig` (059) certificate fields; allow `clearinghouse` agent kind |
| Modified | `src/bilancio/decision/profiles.py` | new `ClearinghouseProfile` dataclass (tunables) |
| Modified | `src/bilancio/domain/instruments/base.py` | `InstrumentKind.CLEARINGHOUSE_CERTIFICATE` (legacy enum, display/export only — see deviation note) |
| Modified | `src/bilancio/scenarios/ring_explorer.py`, `config/models.py::RingExplorerParamsModel` | emit `clearinghouse` block when requested |
| Modified | `src/bilancio/ui/cli/_sweep_ring.py` | `--clearing-certificates` flag (implies `--clearing`) |
| Modified | `src/bilancio/experiments/ring.py` | pass flag to compiler; registry columns for new metrics |
| Modified | `src/bilancio/analysis/report.py` | `compute_run_level_metrics`: new certificate metrics from events |
| Modified | `src/bilancio_v2/views.py` | CH1 balance-sheet view (pledged claims / certificates outstanding) |
| New test | `tests/v2/test_certificates.py` | unit tests for issuance/redemption/recourse/conservation |
| New test | `tests/integration/test_clearinghouse_certificates.py` | end-to-end ring scenario with certificates |
| New golden | `tests/v2/golden_cases/` + `golden/` | one certificates-mode scenario pinned as golden |
| Config | `examples/scenarios/clearinghouse_certificates.yaml` | minimal 4-firm demonstration scenario |

**Deviation from the design brief**: the brief says "Update `InstrumentKind`, `policy.py`, and
create the instrument dataclass." The v2 kernel has **no** `InstrumentKind` enum — it exists
only in the legacy v1 domain (`src/bilancio/domain/instruments/base.py`); the v2 ledger uses
balance maps (`cash`, `reserves`, `deposits`) plus contract records (`Payable`, `BankLoan`, …).
Certificates therefore become (a) a new balance map `certificates: defaultdict[str, Decimal]`
(bearer, like cash) and (b) a `CertificatePledge` record (the CH-side contract). The legacy
enum gains a member only so v1-typed export/visualization code stays coherent.

## Design

### Scenario configuration

```yaml
clearinghouse:
  enabled: true
  mode: certificates        # extends 059's "netting"; netting still runs first
  cert_haircut: 0.25        # certificates issued = (1 - haircut) × pledged face
  cert_rate: 0.06           # interest accruing to CH against the pledging member (see OPEN QUESTION 1)
  max_issuance_per_member: 1.0   # cap as fraction of member gross dues today
  cert_max_tenor: null      # eligible receivables due within N days; null = scenario maturity_days
  mandatory_acceptance: true     # stage 2 fixed true (see OPEN QUESTION 2)
```

`scenario_gates.build_clearing_config` (from 059) maps the new fields onto `CleanClearingConfig`
(frozen dataclass in `subsystem_config.py`), the lender/rating builder pattern. When
`mode: certificates`, `engine.prepare_scenario` registers agent `CH1` (kind `clearinghouse`,
name "Clearinghouse") if not declared in the scenario, and appends `CertificateFacilityPhase`
immediately after `ClearingPhase` and before `SettlementPhase`.

### Agent: Clearinghouse (CH1)

Not a bank (no deposits, no reserves, no CB access) and not a dealer (no quotes, no inventory).
Balance sheet — **assets**: claims on pledged receivables (`CertificatePledge` records over
`Payable`s marked `pledged_to="CH1"`) plus transient cash from matured-collateral proceeds;
**liabilities**: certificates outstanding (sum of the `certificates` balance map);
**equity**: accrued certificate interest (its only loss buffer).

### Instrument: clearinghouse certificate

- Bearer: a per-agent `Decimal` balance (`ledger.certificates[agent_id]`), denominated 1:1
  with cash. No lot tracking (unlike cash) — certificates are fungible; emission/transfer/burn
  events carry amounts only.
- Tracked totals `certificates_issued_total` / `certificates_retired_total`; invariant
  `sum(certificates.values()) == issued − retired` added to `Ledger.check_invariants()`.
- Interest (`cert_rate`) accrues to the CH **against the pledging member** — a financing charge
  on the pledge, not a coupon to the holder.

**OPEN QUESTION 1 — day-rate conversion.** `cert_rate: 0.06` is annual-equivalent, but the
simulation has no calendar; every existing v2 rate (`NonBankLoan`, `BankLoan`, `CBLoan`) is a
**flat rate over the instrument's life** (`repayment = amount × (1 + rate)`, int-rounded).
Options: (a) flat over pledge life like existing instruments, (b) per-diem `cert_rate / 360`,
charged as `Decimal(int(issued × cert_rate / 360 × days_outstanding))` at redemption.
**Recommendation**: (b) — historical certificates carried per-diem interest, and flat-over-life
makes short pledges as costly as long ones, distorting the pledge decision.

### Mechanism: `CertificateFacilityPhase` (name `SubphaseB_Certificates`)

Runs each day between `ClearingPhase` and `SettlementPhase`. Deterministic member order
(sorted agent id), matching the determinism discipline of the other plugins.

1. **Shortfall detection** (need-based, no member discretion in stage 2): for each non-defaulted
   `household`/`firm` member, `shortfall = max(0, net_dues_today − cash − certificates)` where
   `net_dues_today` is the post-netting obligation from `ClearingPhase`. Members with
   `shortfall == 0` do nothing.
2. **Eligible collateral**: the member's receivables (`Payable`s where it is creditor) that
   are (a) unsettled, (b) `due_day > ledger.day`, (c) `due_day − ledger.day <= cert_max_tenor`,
   (d) not own-issued (`debtor != member`), (e) unpledged (`pledged_to is None`), (f) debtor
   not defaulted. Sorted earliest due day first (minimizes interest carry).
3. **Issuance**: pledge whole receivables in order until issued certificates cover the
   shortfall or collateral is exhausted, capped at cumulative issuance today
   ≤ `max_issuance_per_member × member gross dues today` (no `Payable` splitting in stage 2 —
   OPEN QUESTION 4). Per pledge: mark `payable.pledged_to = "CH1"`, append
   `CertificatePledge(id, member, payable_id, pledged_face, certificates_issued, issuance_day)`,
   credit `ledger.certificates[member] += Decimal(int(pledged_face × (1 − cert_haircut)))`,
   emit `CertificatePledgeCreated` + `CertificatesIssued`.
4. The phase returns `False` for the stability impact signal (like `RatingPhase` /
   `LendingPhase` — credit decisions are not impactful settlement activity).

A pledged receivable still settles normally in `SettlementPhase` on its due day, **but the
proceeds are redirected to CH1** (see Redemption). Pledged receivables are excluded from
dealer/lender/rollover interactions (`pledged_to is not None` guard) — irrelevant in the target
no-bank/no-dealer ring, but enforced so combined arms cannot double-use collateral.

### Settlement integration (means of payment)

`policy.py` gains `MOP_CLEARINGHOUSE_CERT = "clearinghouse_certificate"` and a `clearinghouse`
capability row (`mop_order=(MOP_CASH,)`, `can_default=False`). When certificates mode is on,
`prepare_scenario` applies `policy.with_certificate_mop()`, which appends the new mop **after**
`cash` in the `household`/`firm` rows (cash first — certificates are the inferior money,
accepted at par only inside the association). The default `CapabilityMatrix` is untouched, so
every non-certificates scenario sees identical policy (golden safety).

`settlement.settle_payable`'s mop loop gains an
`elif means == "clearinghouse_certificate":` leg that pays
`min(ledger.certificates[debtor], remaining)` via `ledger.transfer_certificates(debtor,
creditor, paid)` (emits `CertificatesTransferred`), mirroring the cash leg. Acceptance is
**mandatory**: any ring-member creditor is paid in certificates without consent (historical
rule). Payments **to CH1 itself** in certificates are burned on receipt (CH receiving its own
liability retires it) — relevant to the recourse path.

**State-synchronization audit (checklist item 9).** Every site that computes "can the debtor
pay" or distributes debtor liquidity must count certificates per the mop order:

| Site | File:symbol | Change |
|------|-------------|--------|
| Payable settlement | `plugins/settlement.py::settle_payable` | certificate leg after cash (above) |
| Default recovery pool | `settlement.py::distribute_pro_rata_recovery` + `ledger.agent_liquid_assets` | include certificate balance in the defaulted member's liquid assets; recovery transfers certificates after deposits/cash |
| Rollover return-flow | `settlement.py::rollover_single_payable` | unchanged in stage 2: rollover returns cash/deposits only; certificates are NOT lent back (OPEN QUESTION 5) |
| Certificate facility itself | `plugins/certificates.py` | shortfall = dues − cash − certificates (no double-issuance) |
| Lender shortfall screen | `plugins/lending.py::collect_lending_opportunities` | out of scope (no lender arm with certificates in stage 2); guard documented |
| Views / balance display | `views.py` | certificates shown as asset of holder, liability of CH1 |
| Invariants | `ledger.check_invariants` | certificate conservation + non-negativity |

### Redemption

When a pledged receivable settles on its due day, `settle_payable` routes the payment to CH1
instead of the original creditor (consulting `pledged_to`). Redemption is processed at the
start of the next day's facility phase, before new issuance (deterministic):

1. Compute `interest` per OPEN QUESTION 1; `owed = certificates_issued + interest`.
2. **Burn-from-member**: retire up to `min(ledger.certificates[member], certificates_issued)`
   of the member's own holdings (`CertificatesRetired`).
3. **Cash settlement of the remainder**: the member's certificates may have circulated away.
   CH keeps proceeds covering `owed` minus what was burned at par; the residual
   `proceeds − owed` returns to the member in cash (`CertificateExcessReturned`).
4. Cash retained against still-circulating certificates funds a **redemption window**: holders
   redeem certificates for cash while `ledger.cash["CH1"] > 0`, in deterministic holder order
   at the end of each facility phase, and unconditionally at simulation end (`finalize` hook,
   mirroring `finalize_banking`). Primary retirement remains collateral maturity.

**OPEN QUESTION 3 — burn-vs-redeem ordering.** The design brief assumes the member still holds
its certificates at maturity ("retires that member's certificates… returns any excess"), but
certificates are bearer and circulate. **Recommendation** (encoded above): burn the member's own
holdings first, hold cash against circulating paper in the redemption pool, return only the true
excess. Conservation reads `outstanding == issued − retired`, `retired` counting burns + redemptions.

### Recourse and the loss waterfall

If a **pledged receivable defaults** (its debtor is expelled before paying):

1. The pledge closes at recovered value (pro-rata recovery + reassigned receivables flow to CH1
   as claim holder — `reassign_receivables` reassigns by creditor, so either the pledged
   payable's creditor-of-record becomes CH1 at pledge time, or reassignment consults
   `pledged_to`; the implementation must pick one and test it).
2. The member's **deficiency** `certificates_issued + interest − recovered` becomes a new
   `Payable` member → CH1 due **next day** (`maturity_distance=1`, reason
   `"certificate_recourse"`, event `CertificateRecourseCreated`). Settling it with the member's
   own certificates burns them (par offset).
3. If the member then defaults on the recourse payable, standard `expel-agent` machinery runs
   and CH1 joins the pro-rata pool like any creditor. CH1 **writes down** the remainder: first
   against its **accumulated interest margin** (equity), then — terminal backstop — a
   **pro-rata haircut on all certificate holders** (`CertificateHaircutApplied`, reducing
   `certificates[holder]` balances and the `retired` total).

**OPEN QUESTION 6 — exact waterfall.** Alternatives: (a) CH equity first, then certificate
haircut (encoded above — holders lose last, which made the historical instrument acceptable);
(b) socialize deficiencies via member assessment payables (historically accurate, but a new
obligation channel); (c) let CH default like any agent. **Recommendation**: (a) for stage 2,
(b) as the stage-3 extension. Under (a) **CH cannot fail** (`can_default=False`).

### Events (new kinds; payloads carry member/amount/pledge ids)

`CertificatePledgeCreated`, `CertificatesIssued`, `CertificatesTransferred`,
`CertificatesRetired`, `CertificateInterestCharged`, `CertificateExcessReturned`,
`CertificateRecourseCreated`, `CertificateHaircutApplied`, `CertificateRedemptionWindow` —
drive metrics and the HTML event log; none appears unless mode is `certificates`.

## New Agent Type Design Checklist (CLAUDE.md, all 9 items)

1. **Instruments** — Holds (assets): pledged receivables via `CertificatePledge` (claim on a
   `Payable` marked `pledged_to`); transient cash from collateral proceeds. Issues (liabilities):
   certificates (bearer balance map; par value, `cert_rate` interest charged to the pledging
   member, no fixed maturity — retired by collateral maturity / redemption). Legacy
   `InstrumentKind.CLEARINGHOUSE_CERTIFICATE` added for export/display parity only (deviation above).
2. **Means of payment** — CH1 settles its own (rare) obligations in cash:
   row `"clearinghouse": AgentCapabilities(mop_order=(MOP_CASH,), can_default=False)`. Member
   mop change (only via `with_certificate_mop()`, defaults untouched): `household`
   `(bank_deposit, cash, clearinghouse_certificate)`; `firm` `(cash, bank_deposit, clearinghouse_certificate)`.
3. **Decision model** — CH is rule-based, no optimization: parameters
   `{cert_haircut=0.25, cert_rate=0.06, max_issuance_per_member=1.0 (× member gross dues),
   eligibility: due within cert_max_tenor days (default = scenario maturity_days)}`. Members'
   pledge decision is need-based (post-netting shortfall), zero discretion in stage 2. A
   `ClearinghouseProfile` dataclass (in `bilancio/decision/profiles.py`, mirroring
   `LenderProfile`) carries the tunables so later stages add behavior without schema churn.
4. **Information model** — Full observability inside the ring: CH reads members' post-netting
   due-today obligations and receivable books directly from the ledger. No noise, sampling, or
   Bayesian updating in stage 2 (contrast `RiskAssessor`); the haircut is the only risk control.
   Future: condition the haircut on the rating registry.
5. **Capitalization** — None. CH issues only against collateral; its equity buffer is accrued
   interest. An optional initial guarantee fund is explicitly OUT of scope (would interact with
   κ accounting and the waterfall).
6. **Timing / phase** — `SubphaseB_Certificates`, after `ClearingPhase`, before
   `SettlementPhase`: netting first shrinks gross dues; issuance before settlement is the
   point. Redemptions process at phase start, then new issuance. Returns `False` for the
   stability impact signal.
7. **Failure mode** — CH cannot fail in stage 2: recourse + haircut + interest-margin equity
   absorb losses; the pro-rata holder haircut is the terminal backstop (`can_default=False`,
   like the central bank). Members fail normally; their certificates join the recovery pool.
8. **Interactions** — Firms and households in the ring only; no bank, dealer, lender, or CB
   required or consulted. Pledged receivables are fenced off from dealer trading and lender
   collateral by the `pledged_to` guard. CH never pledges, never trades.
9. **State synchronization** — Certificates change the cash-sufficiency computation in
   settlement. Audited sites (table above): `settle_payable` mop loop, `agent_liquid_assets` /
   `distribute_pro_rata_recovery`, rollover return-flow (unchanged, documented), the facility's
   own shortfall computation, lender screen (guarded out), views, invariants. Cross-phase test:
   a member pays partly in certificates then defaults later; recovery counts the remaining
   certificates exactly once.

## Sweep surface

### Backend layers

| Layer | Touched? | How? |
|-------|----------|------|
| **Domain** (agent types, policy, instruments) | yes | `clearinghouse` kind + capability row; certificate mop id; legacy `InstrumentKind` member |
| **Decision** (profiles) | yes | new `ClearinghouseProfile` dataclass (tunables only; rule-based) |
| **Engines** (phases) | yes | `CertificateFacilityPhase` between clearing and settlement; settlement mop leg; recovery inclusion |
| **Ops** (settlement mechanics) | yes | certificate transfer/burn ops on the ledger; proceeds redirection for pledged receivables |
| **Scenarios** (ring builder, config) | yes | `clearinghouse.mode=certificates` block; `RingExplorerParamsModel.clearinghouse` passthrough |
| **State** | yes | `ledger.certificates`, `certificate_pledges`, issued/retired totals, `Payable.pledged_to` |

### Sweep pipeline layers

| Layer | Touched? | How? |
|-------|----------|------|
| **CLI params** (`ui/cli/_sweep_ring.py`) | yes | `--clearing-certificates` flag (implies 059's `--clearing`); haircut/rate/cap options with spec defaults |
| **Sweep config** (`experiments/ring.py`) | yes | runner fields + registry columns `certificates_issued_total`, `certificates_outstanding_peak`, `cert_default_losses` |
| **Runner logic** | yes | flag → generator config → compiled scenario `clearinghouse:` block; no new arm type (ring sweep is single-arm) |
| **Metrics collection** (`analysis/report.py`) | yes | `compute_run_level_metrics` derives the three metrics from certificate events |
| **Post-sweep reports** | yes | new columns in `results.csv`; delta-vs-kappa chart split by certificates on/off |
| **Pre-flight checks** | yes | new viability note **V9**: certificates only bind when κ < 1 and μ-skew leaves receivables maturing after dues; warn if `cert_haircut ≥ 1` or `maturity_days == 1` (no temporal mismatch to bridge) |

### Interaction expectations (hypothesis)

- Region of effect: **κ ∈ [0.25, 0.75], μ ∈ [0.5, 1]** (front-loaded dues, back-loaded
  receivables — exactly where 059 netting is weak because offsetting dues do not coincide in
  time). Expectation: `delta_total` falls substantially (target ≥ 30% relative reduction at
  κ=0.5, μ=0.75 vs netting-only), `phi_total` rises correspondingly, and
  `certificates_issued_total > 0` there with small nonzero `cert_default_losses` at the lowest
  κ (recourse path exercised).
- κ ≥ 2 or μ = 0: certificates ≈ inert (`certificates_issued_total ≈ 0`); deltas match
  netting-only within seed noise.
- Flag off: byte-identical events to pre-change runs (golden criterion).

### Default value discipline

| Parameter | Default | Why this default | Backward-compatible? |
|-----------|---------|------------------|---------------------|
| `clearinghouse.mode` | `netting` (from 059) | certificates are opt-in | Yes — absent/netting block never builds the phase |
| `cert_haircut` | `0.25` | historical 75% advance rate (1873/1893/1907 associations) | Yes — only read in certificates mode |
| `cert_rate` | `0.06` annual-equiv. | historical 6%; conversion per OPEN QUESTION 1 | Yes — only read in certificates mode |
| `max_issuance_per_member` | `1.0` × gross dues | cap exists but does not bind in normal runs | Yes |
| `cert_max_tenor` | `null` → scenario `maturity_days` | all ring receivables eligible by default | Yes |
| `mandatory_acceptance` | `true` | historical rule; voluntary variant deferred | Yes — only read in certificates mode |
| `--clearing-certificates` (CLI) | off | sweep behavior unchanged unless requested | Yes |
| `Payable.pledged_to` | `None` | matches WP-1 field design; `None` ≡ today's behavior | Yes — no code path reads it when no pledges exist |

## Open questions — RESOLVED (2026-06-11; details inline above)

1. **DECIDED** — cert_rate day conversion: per diem `cert_rate/360`.
2. **DECIDED (user-confirmed)** — acceptance is **mandatory** in stage 2 (historical rule); expose
   `mandatory_acceptance` now (validated `true`) so a voluntary stage 3 is a behavior, not schema, change.
3. **DECIDED** — burn member's own holdings first, redemption pool for circulating paper, excess back to member.
4. **DECIDED** — whole receivables only (no splitting); over-issuance bounded by one face value.
5. **DECIDED** — rollover return-flow stays cash/deposit only in stage 2.
6. **DECIDED** — waterfall (a): CH interest-margin equity, then pro-rata holder haircut; CH cannot fail in stage 2.

## Acceptance criteria

1. **Inert when off**: with no `clearinghouse:` block or `mode: netting`, the phase is not
   constructed, the policy matrix is unmodified, and `tests/v2/test_golden.py` plus the full
   suite (`uv run pytest tests/ -v`) pass with **zero golden regeneration**.
2. **Issuance respects haircut and eligibility**: a member with a shortfall and one eligible
   receivable of face F receives exactly `int(F × (1 − cert_haircut))` certificates; own-issued,
   due-today, already-pledged, defaulted-debtor, and beyond-tenor receivables are never pledged;
   the per-member cap binds when configured below need.
3. **Certificates settle and circulate**: a debtor with 0 cash and certificates ≥ due settles in
   full (`PayableSettled` after `CertificatesTransferred`); the receiving creditor spends the
   same certificates on its own dues a later day (3-firm chain integration test).
4. **Cash-first mop order**: a debtor holding both pays cash to exhaustion before any
   certificate transfer (asserted on event order and amounts).
5. **Redemption and conservation**: at collateral maturity, proceeds route to CH1, interest is
   charged, excess returned; at every day boundary
   `sum(certificates) == issued_total − retired_total` and the invariant suite passes; at
   simulation end outstanding certificates redeem against CH cash via the finalize hook.
6. **Recourse path**: pledged-receivable default creates the next-day member→CH1 recourse
   payable; member default on it routes losses through CH interest margin first, then a
   pro-rata holder haircut, with `cert_default_losses` equal to the written-down amount.
7. **Metrics wired end-to-end**: a local smoke sweep
   (`uv run bilancio sweep ring --clearing-certificates --n-agents 10 --maturity-days 5 --kappas "0.5,1" --mus "0.75"`)
   completes; `results.csv` has the three new columns, nonzero `certificates_issued_total` at
   κ=0.5/μ=0.75, and `delta_total` ≤ the netting-only baseline there.
8. **Tests enumerated**: unit — issuance math, eligibility filter, cap, interest accrual,
   burn/redeem ordering, conservation invariant, haircut waterfall; integration — circulation
   chain, recourse cascade, certificates + expel-agent recovery (cross-phase sync test,
   checklist item 9), inert-when-off event-stream equality; golden — one certificates scenario
   captured and pinned.

## Stage-3 candidates (out of scope)

Member assessments (waterfall b), voluntary acceptance, rating-conditioned haircuts,
certificates in bank/dealer/lender arms, migration onto full WP-1 collateral (roadmap 056).
