# Architecture

## Package Dependency Graph

Arrows point from dependent to dependency. Acyclic by design.

```mermaid
graph BT
    core["core\nids, errors, atomic_tx,\ninvariants, events"]
    domain["domain\nagent, instruments,\npolicy, goods"]
    ops["ops\nprimitives, banking,\ncashflows, aliases"]
    engines["engines\nsystem, simulation,\nsettlement, clearing"]
    dealer_eng["engines/dealer_*\nintegration, wiring,\ntrades, sync"]
    dealer_mod["dealer\nmodels, kernel, trading,\nrisk_assessment, metrics"]
    config["config\nmodels, loaders, apply"]
    analysis["analysis\nbalances, metrics,\nvisualization, network"]
    runners["runners\nlocal_executor,\ncloud_executor"]
    ui["ui\ncli, html_export,\nrun, display"]
    cloud["cloud\nmodal_app"]
    storage["storage\nsupabase_client"]
    experiments["experiments\nring, balanced_comparison"]
    jobs["jobs\nmanager, models"]

    domain --> core
    ops --> domain
    ops --> core
    engines --> ops
    engines --> domain
    engines --> core
    dealer_eng --> engines
    dealer_eng --> dealer_mod
    dealer_eng --> domain
    config --> engines
    config --> domain
    analysis --> engines
    analysis --> domain
    runners --> engines
    runners --> config
    ui --> runners
    ui --> analysis
    ui --> engines
    ui --> config
    cloud --> runners
    cloud --> storage
    experiments --> runners
    experiments --> config
    jobs --> storage

    style core fill:#e1f5ff
    style domain fill:#f3e5f5
    style ops fill:#e8f5e9
    style engines fill:#fff3e0
    style dealer_eng fill:#fce4ec
    style dealer_mod fill:#fce4ec
```

## Day Simulation Pipeline

Each simulation day follows this fixed phase sequence (see `engines/simulation.py:run_day`).

```mermaid
graph TD
    A["Phase A\nStart of Day\nLog event, advance clock"]
    B1["Subphase B1\nScheduled Actions\napply_action() for today"]
    BD{"Dealer\nenabled?"}
    BDealer["Subphase B-Dealer\nrun_dealer_trading_phase()\nsync_dealer_to_system()"]
    B2["Subphase B2\nSettlement\nsettle_due() for payables\nand delivery obligations"]
    BR{"Rollover\nenabled?"}
    BRoll["Subphase B-Rollover\nrollover_settled_payables()\nCreate replacement payables"]
    C["Phase C\nClearing\nsettle_intraday_nets()\nNet interbank positions"]
    D{"CB\nexists?"}
    DPhase["Phase D\nCB Corridor\ncredit_reserve_interest()\ncb_repay_loan()"]
    Inc["Increment Day\nsystem.state.day += 1"]

    A --> B1
    B1 --> BD
    BD -->|yes| BDealer --> B2
    BD -->|no| B2
    B2 --> BR
    BR -->|yes| BRoll --> C
    BR -->|no| C
    C --> D
    D -->|yes| DPhase --> Inc
    D -->|no| Inc

    style A fill:#e3f2fd
    style B1 fill:#ffe0b2
    style BDealer fill:#fce4ec
    style B2 fill:#ffe0b2
    style BRoll fill:#f0f4c3
    style C fill:#e0f2f1
    style DPhase fill:#f3e5f5
```

## Dealer Subsystem

The dealer subsystem bridges the main simulation (Payables) with a secondary market (Tickets).
Split across four modules in `engines/`:

```mermaid
graph LR
    subgraph Main["Main System"]
        contracts["Payables\n(contracts dict)"]
        agents["Agents\n(cash, holdings)"]
    end

    subgraph Orchestration["dealer_integration.py"]
        subsystem["DealerSubsystem\ndataclass"]
    end

    subgraph Init["dealer_wiring.py"]
        wiring["convert payables to tickets\ninit market makers\ninit traders"]
    end

    subgraph Trading["dealer_trades.py"]
        trades["build eligible sellers/buyers\nexecute interleaved order flow\nrecord & execute trades"]
    end

    subgraph Sync["dealer_sync.py"]
        sync["sync cash & ownership\nupdate maturities\ncapture snapshots"]
    end

    subgraph DealerMod["dealer/ module"]
        kernel["kernel: quotes & spreads"]
        executor["trading: TradeExecutor"]
        risk["risk_assessment"]
    end

    contracts -->|"initialize"| wiring
    wiring --> subsystem
    subsystem -->|"each day"| trades
    trades --> executor
    trades --> kernel
    trades --> risk
    subsystem -->|"after trading"| sync
    sync -->|"update"| contracts
    sync -->|"update"| agents

    style Main fill:#fff9c4
    style Orchestration fill:#fff3e0
    style Init fill:#e8f5e9
    style Trading fill:#fce4ec
    style Sync fill:#e0f2f1
    style DealerMod fill:#f3e5f5
```

## Target Architecture

This is the target architecture the codebase is being incrementally reorganized toward.
Each phase produces a fully working codebase — no big-bang rewrites.

```mermaid
graph BT
    core["<b>Core</b>\nSystem primitives + State + Invariants\n<i>core/, engines/state.py</i>"]
    domain["<b>Domain</b>\nAgents + Instruments + Rules\n(limitations, capacities, means of payment, defaults)\n<i>domain/, domain/policy.py</i>"]
    interaction["<b>Agent Interaction Specification</b>\nAction specs + Decision profiles\nInformation pipeline:\naccess → estimation → valuation → risk assessment → decision\n<i>decision/, information/, specification/</i>"]
    simulator["<b>Simulator</b>\nSimulation engine + Settlement + Clearing\n<i>engines/, ops/</i>"]
    loader["<b>Scenario Loader</b>\nPlugin + Configuration + Compiler + Viability checks\n<i>config/, scenarios/</i>"]
    analysis["<b>Analysis</b>\nDescriptive + Causal + Network + Temporal/Equilibrium\n<i>analysis/</i>"]

    domain --> core
    interaction --> domain
    interaction --> core
    simulator --> interaction
    simulator --> domain
    simulator --> core
    loader --> simulator
    loader --> domain
    analysis --> simulator
    analysis --> domain

    style core fill:#e1f5ff,stroke:#0277bd,stroke-width:2px
    style domain fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style interaction fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style simulator fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style loader fill:#fce4ec,stroke:#c62828,stroke-width:2px
    style analysis fill:#f0f4c3,stroke:#827717,stroke-width:2px
```

### Package-to-Layer Mapping

| Current Package | Target Layer | Notes |
|---|---|---|
| `core/` | Core | Already correct |
| `domain/` | Domain | Already correct |
| `engines/state.py` | Core (State) | Extract from `engines/system.py` |
| `engines/system.py` | Simulator | After State extraction |
| `engines/simulation.py` | Simulator | Already correct |
| `engines/settlement.py` | Simulator | Default rules move to Domain |
| `decision/` | Agent Interaction | Already correct, gains `RiskAssessor` |
| `dealer/risk_assessment.py` | Agent Interaction | Move to `decision/risk_assessment.py` |
| `information/` | Agent Interaction | Wire into all agent decisions |
| `specification/` | Agent Interaction | Design-time validation tool |
| `config/` | Scenario Loader | Already correct |
| `scenarios/` | Scenario Loader | Gains viability checks |
| `analysis/` | Analysis | Expand with causal/network/temporal |
| `ops/` | Simulator | Already correct |
| `dealer/` | Simulator (secondary market) | Models + kernel stay, risk moves out |

### Migration Status

- [x] Phase 1: RiskAssessor extraction to `decision/` + architecture doc
- [ ] Phase 2: State extraction from `engines/system.py`
- [ ] Phase 3: Domain rules consolidation
- [ ] Phase 4: InformationService wiring into trader decisions
- [ ] Phase 5: Agent intention architecture
- [ ] Phase 6: Analysis expansion
