# Literature Review: Debt Topology, Payment Networks, and Settlement ABMs

> **Date**: 2026-03-17
> **Purpose**: Survey the academic landscape around bilancio's core themes — circular debt topology, payment network clearing, agent-based settlement, and secondary market intermediation — to identify gaps and position the project for publication.

---

## Table of Contents

1. [Eisenberg-Noe Clearing Models](#1-eisenberg-noe-clearing-models)
2. [DebtRank and Network Contagion](#2-debtrank-and-network-contagion)
3. [Agent-Based Models of Payment Systems](#3-agent-based-models-of-payment-systems)
4. [Graph-Theoretic Mutual Debt Netting](#4-graph-theoretic-mutual-debt-netting)
5. [Gridlock Resolution and Liquidity-Saving Mechanisms](#5-gridlock-resolution-and-liquidity-saving-mechanisms)
6. [Stock-Flow Consistent (SFC) and AB-SFC Models](#6-stock-flow-consistent-sfc-and-ab-sfc-models)
7. [Dealer / Market-Maker ABMs](#7-dealer--market-maker-abms)
8. [Kalecki Profit Equation](#8-kalecki-profit-equation)
9. [Circular Debt in Developing Economies](#9-circular-debt-in-developing-economies)
10. [Gaps in the Literature](#10-gaps-in-the-literature)
11. [Bilancio's Unique Positioning](#11-bilancios-unique-positioning)
12. [Potential Publication Venues](#12-potential-publication-venues)
13. [References](#13-references)

---

## 1. Eisenberg-Noe Clearing Models

The foundational framework for interbank clearing is **Eisenberg & Noe (2001)**, "Systemic Risk in Financial Systems" in *Management Science*. They model a financial system as a directed graph where nodes are institutions and edges are nominal obligations, then prove existence of a unique clearing payment vector via Tarski's fixed-point theorem. The clearing vector satisfies limited liability (no agent pays more than it has) and proportional repayment (when insolvent, pay creditors pro rata).

Key extensions:

- **Rogers & Veraart (2013)** add dead-weight losses from defaults — when an institution fails, its estate shrinks, amplifying contagion.
- **Veraart (2020)** extends to multiple maturities in *SIAM Journal on Financial Mathematics*, allowing obligations with different due dates — closer to bilancio's maturity structure but still solved via fixed-point, not sequentially.
- **Banerjee et al. (2024)** develop dynamic clearing with contagion, adding time but not behavioral heterogeneity.
- **Sonin & Sonin (2022)** study clearing payments in dynamic financial networks.

**Key limitation**: All these models are **static or quasi-static**. They find a fixed-point clearing vector given a liability matrix. They do not model sequential day-by-day settlement with agent decision-making, adaptive behavior, or intermediary intervention.

## 2. DebtRank and Network Contagion

**Battiston et al. (2012)** introduced **DebtRank**, a network centrality measure for systemic risk, applied to FED emergency loan data (2008–2010). It quantifies how much of the system's economic value is affected by the distress of a single node.

Key subsequent work:

- **Poledna & Thurner** show that multi-layer network analysis (credit, derivatives, FX, securities) reveals that neglecting inter-layer interactions underestimates systemic losses.
- **Bardoscia et al.** find that higher connectivity can increase instability due to cyclic structures — directly relevant to ring topology.
- **Acemoglu, Ozdaglar & Tahbaz-Salehi**, "Systemic Risk and Stability in Financial Networks" — demonstrate the **"robust-yet-fragile"** property: dense networks share small shocks efficiently but propagate large shocks catastrophically.

**Key limitation**: DebtRank measures contagion exposure but does not model the settlement process, agent decisions, or potential interventions (like secondary markets).

## 3. Agent-Based Models of Payment Systems

- **Galbiati & Soramäki (2011)**, "An agent-based model of payment systems" in *Journal of Economic Dynamics and Control* — agents choose how much liquidity to post in an RTGS system. One of the closest models to bilancio, but focused on intraday liquidity management in RTGS, not multi-day settlement with defaults.
- **BoF-PSS2** (Bank of Finland Payment and Settlement System Simulator) — 117 licenses in 40 countries, used by ECB, Fed NY, and Bank of England. Replicates production settlement logics but lacks theoretical framework, topology control, behavioral parameters, or secondary markets.
- **Bookstaber et al. (OFR)**, "Agent-Based Model for Financial Vulnerability" — network-based vulnerability analysis from the Office of Financial Research.
- **Soramäki et al.**, "The Topology of Interbank Payment Flows" (NY Fed Staff Report) — empirical analysis of Fedwire topology, finding small-world and scale-free properties.

**Key limitation**: These models either replicate specific real-world systems (BoF-PSS2) or study abstract liquidity management. None embed a secondary market for receivables within the settlement network.

## 4. Graph-Theoretic Mutual Debt Netting

A practical literature focuses on algorithms for finding and cancelling cycles in obligation networks:

- **Gazda, Horvath & Resovsky (2015)**, "An Application of Graph Theory in the Process of Mutual Debt Compensation" — applies Klein's cycle-cancelling algorithm to debt digraphs.
- **Gavrila & Popa (2020)**, "A novel algorithm for clearing financial obligations between companies" — deployed at the **Romanian Ministry of Economy**, clearing over 100M EUR in three weeks. Demonstrates the real-world significance of circular debt clearing.
- **Guichon et al. (2025)**, "Integral B2B Debt Netting" — proves that **integral netting is NP-complete** and proposes heuristic approaches.

**Key insight for bilancio**: The Romanian deployment and the NP-completeness result confirm that mutual debt clearing is a real, hard problem. But this literature treats it purely algorithmically — it asks "how can an omniscient planner cancel cycles?" rather than "what if agents could _sell_ their claims to a market-maker?" Bilancio addresses the latter.

## 5. Gridlock Resolution and Liquidity-Saving Mechanisms

- **Bech & Soramäki (2001)**, "Gridlock Resolution in Interbank Payment Systems" — seminal work defining gridlock and resolution algorithms.
- **BIS Working Paper (2024)** — Auction-based liquidity-saving mechanisms.
- **Management Science (2024)** — Hybrid quantum annealing for payment reordering, confirming the problem is NP-hard.

**Relevance**: Bilancio's dealer mechanism is an alternative gridlock resolution approach — instead of reordering payments (a central planner's tool), it provides a market where agents can convert illiquid receivables to cash.

## 6. Stock-Flow Consistent (SFC) and AB-SFC Models

The SFC tradition (Godley & Lavoie, 2007) and its agent-based extensions (AB-SFC) model economies as systems of interlocking balance sheets where every flow has a counterpart and every stock is someone's asset and someone's liability.

- **Caiani et al. (2016)**, "Agent-based stock-flow consistent macroeconomics" — the canonical AB-SFC framework.
- **Dawid et al. (2019)**, the EURACE model — large-scale AB-SFC with spatial structure.

**Key limitation**: AB-SFC models operate at the **sectoral** or **representative-agent** level. They enforce accounting consistency but abstract away network topology — there is no directed graph of bilateral obligations, no ring structure, no settlement sequence. Bilancio bridges the gap between AB-SFC's accounting rigor and network models' topological detail.

## 7. Dealer / Market-Maker ABMs

- **JP Morgan Research** — Multi-agent simulation for pricing and hedging in a dealer market.
- **ArXiv (2023)**, "Dealer Strategies in Agent-Based Models" — risk aversion, quote sizing in OTC markets.
- **ArXiv (2024)**, "Modelling Opaque Bilateral Market Dynamics" — Australian government bond OTC market.
- **Springer (2020)**, "Market makers activity: behavioural and agent-based approach."
- **MDPI (2025)**, "Liquidity Drivers in Illiquid Markets" — heterogeneous agent simulation.

**Key limitation**: These focus on **equity and bond markets**, studying price discovery and spreads. None embed dealer market-making inside a payment obligation network to study whether secondary market liquidity reduces default cascades.

## 8. Kalecki Profit Equation

Kalecki's profit equation (1930s–1940s) states that aggregate profits equal investment plus capitalist consumption minus worker saving — fundamentally, that spending creates income in a circular flow.

- **Setterfield & Budd (2011)**, "A Keynes-Kalecki Model of Cyclical Growth with Agent-Based Features" — the closest integration of Kalecki with ABM, though still at the macro level.
- **Laski & Walther (2013)**, "Kalecki's Profit Equation After 80 Years" (wiiw Working Paper).
- **Bond Economics blog** (Romanchuk) — accessible treatment of the profit equation's implications.

**Key gap**: The Kalecki equation is widely known as a macroeconomic accounting identity but has **never been operationalized as a network-topological construct** with agent-based simulation. Bilancio's ring — where each agent's ability to pay depends on receiving payment from the agent behind it — is the micro-foundation of Kalecki's circular flow.

## 9. Circular Debt in Developing Economies

- **Syed Sajid Ali (2010)**, "Dynamics of Circular Debt in Pakistan and Its Resolution" — describes how government arrears cascade through the energy supply chain.
- **Italian government trade debt** — cascading payment failures through supply chains, addressed by 9B EUR emergency payments.
- **Bussoli & Conte (2020)** — trade credit and firm profitability in Italy.

**Key limitation**: This literature is **descriptive and policy-oriented**, not computational. It describes the problem and proposes interventions but lacks formal simulation models to test them.

---

## 10. Gaps in the Literature

| # | Gap | Relevant Strands |
|---|-----|-----------------|
| 1 | **No integration of Kalecki profit equation with network topology ABM.** The equation is treated as a macro identity; nobody has embedded it in a directed-graph payment network. | Kalecki, AB-SFC, Network |
| 2 | **Eisenberg-Noe models are static.** They find clearing vectors but do not model sequential day-by-day settlement with agent behavior or intermediary intervention. | Eisenberg-Noe |
| 3 | **No model combines settlement with secondary market intermediation.** Dealer ABMs study securities markets; payment ABMs study settlement mechanics. Nobody has put them together. | Dealer ABMs, Payment ABMs |
| 4 | **Ring topology is underexplored as a theoretical device.** The literature uses empirically calibrated or random graphs. The ring — the minimal structure capturing circular payment dependencies — has not been systematically studied. | Network topology |
| 5 | **No configurable behavioral profiles for settlement agents.** Existing ABMs use fixed rules (BoF-PSS2) or RL. Parameterized profiles (risk aversion, planning horizon, trading motive) with economic interpretations are absent. | Payment ABMs |
| 6 | **Mutual debt compensation literature is algorithmic, not economic.** Graph netting papers find optimal cycle cancellations but do not model agent preferences about timing, risk, or liquidity. | Graph netting |
| 7 | **Circular debt literature is descriptive, not computational.** Pakistan/Italy cases describe the problem without formal simulation. | Circular debt |
| 8 | **No paired experimental framework for causal identification.** Payment simulators run one-off scenarios. Systematic paired comparisons (same seed, intervention on/off) with parameter sweeps are absent. | Methodology |
| 9 | **Credit intermediation is absent from settlement networks.** Banks and non-bank lenders are studied separately from payment clearing. No model integrates lending into a payment obligation network. | Banking, NBFI |
| 10 | **Bayesian belief updating by settlement agents is absent.** No payment system model has agents that update beliefs about counterparty risk via Bayesian inference and adjust settlement/trading strategies. | Risk assessment |

---

## 11. Bilancio's Unique Positioning

Bilancio sits at the intersection of several literatures, occupying a position that none of them individually covers:

| Literature Strand | What It Does | What It Doesn't Do |
|-------------------|-------------|-------------------|
| Eisenberg-Noe | Static clearing vectors | No behavior, no time, no market |
| DebtRank | Contagion centrality metrics | No settlement mechanics, no liquidity choice |
| SFC / AB-SFC | Balance-sheet consistency | Aggregate sectors, no topology |
| Payment ABMs (BoF-PSS2) | Real settlement rules | No theoretical framework, no dealer |
| Graph netting (Romanian, Klein) | Find and cancel cycles | No behavior, no secondary market |
| Market microstructure | Dealer pricing, spreads | Only securities, never trade credit |
| Kalecki | Profit = Investment identity | Macro only, never micro-network |

### Unique Feature 1: Kalecki Ring as Theoretical Primitive

Bilancio operationalizes the Kalecki circular flow as a **directed ring topology** where N agents each owe their successor. This is not just a network structure — it embodies Kalecki's insight that payment obligations form a loop where one agent's ability to pay depends on receiving payment from another. The kappa parameter (L₀/S₁) directly controls the "leakage" from the circular flow.

### Unique Feature 2: Settlement + Secondary Market Integration

Bilancio is the first model to embed a **dealer market-making system** (bid/ask quotes, inventory management, maturity buckets, VBT anchoring) inside a **payment settlement network**. This answers a question no existing model can: "Does secondary market liquidity for receivables reduce default cascades?" Results from 125-pair sweeps show a 19.3% mean default reduction (p < 10⁻¹⁸), with zero cases of worsening.

### Unique Feature 3: Configurable Behavioral Profiles

The `TraderProfile` dataclass (risk_aversion, planning_horizon, aggressiveness, buy_reserve_fraction, default_observability, trading_motive) creates a high-dimensional behavioral space for controlled experiments. This goes beyond both fixed-rule agents (BoF-PSS2) and RL-learned agents (recent dealer ABMs).

### Unique Feature 4: Four-Stage Risk Assessment Pipeline

The `RiskAssessor` decomposes into BeliefTracker (Bayesian updating), EVValuer (expected value), PositionAssessor (urgency-adjusted thresholds), and TradeGate (accept/reject). This modular pipeline enables independent validation of each stage.

### Unique Feature 5: Hierarchical Monetary Architecture

Bilancio models a realistic monetary hierarchy: CentralBank issues reserves and cash, Banks hold reserves and issue deposits, Firms/Households hold deposits and cash. Settlement follows means-of-payment priority rules (`mop_rank`).

### Unique Feature 6: Paired Experimental Design

The balanced comparison framework (same seed, same topology, intervention on/off) with systematic parameter sweeps enables **causal identification** of intermediary effects — a methodological contribution absent from the literature.

### Unique Feature 7: Multiple Intermediary Types

Bilancio can compare passive (no intermediaries), dealer-only, NBFI-lender-only, bank-only, and combined configurations. The finding that dealer trading dominates NBFI lending (+16.9pp vs +5.8pp at κ=0.3) and that combined effects are not additive is novel.

### Unique Feature 8: ODD Protocol Documentation

The model follows the ODD (Overview, Design concepts, Details) protocol standard for ABM documentation and maintains a validation matrix linking behavioral claims to executable tests.

### Positioning Statement

> Bilancio is the first agent-based model that integrates Kalecki's circular flow theory with Eisenberg-Noe clearing mechanics, secondary market intermediation, and configurable behavioral decision profiles in a ring payment network, enabling causal identification of how different liquidity interventions (market-making, credit provision, central banking) affect default cascades.

---

## 12. Potential Publication Venues

| Venue | Why | Precedent |
|-------|-----|-----------|
| *Journal of Economic Dynamics and Control* | Published Galbiati-Soramäki; strong ABM tradition | Payment system ABMs |
| *Management Science* | Published Eisenberg-Noe; high impact | Clearing models |
| *Journal of Financial Stability* | Systemic risk focus | Network contagion |
| *Computational Economics* | ABM methodology | Agent-based finance |
| *JASSS* (Journal of Artificial Societies and Social Simulation) | ODD protocol, open-source tradition | ABM documentation standards |
| *Journal of Financial Market Infrastructures* (Risk.net) | Payment system practitioners | FMI design |
| *Quantitative Finance* | Market microstructure + risk | Dealer models |
| *Cambridge Journal of Economics* | Heterodox, Kalecki tradition | Post-Keynesian ABM |
| *Metroeconomica* | Kalecki/Kaleckian growth models | Heterodox macro theory |

### Suggested Publication Strategy

**Option A — Extend Eisenberg-Noe** (most established audience): Frame bilancio as "dynamic Eisenberg-Noe with behavioral agents and endogenous secondary markets." Target *JEDC* or *Management Science*. Lead with the formal structure, then show simulation results.

**Option B — Kalecki micro-foundation** (most distinctive): Frame the ring as the micro-foundation of Kalecki's profit equation, showing how macro accounting identities emerge from bilateral settlement. Target *Cambridge Journal of Economics* or *Metroeconomica*. Lead with the theoretical insight, then validate computationally.

**Option C — Practical FMI design** (most applied): Frame as a tool for payment system designers, showing that embedding secondary markets in settlement systems reduces gridlock. Target *JFMI* or submit to a central bank working paper series.

---

## 13. References

### Clearing and Systemic Risk

- Eisenberg, L. & Noe, T. (2001). Systemic Risk in Financial Systems. *Management Science*, 47(2), 236–249. [Link](https://pubsonline.informs.org/doi/10.1287/mnsc.47.2.236.9835)
- Rogers, L.C.G. & Veraart, L.A.M. (2013). Failure and Rescue in an Interbank Network. *Management Science*, 59(4), 882–898.
- Veraart, L.A.M. (2020). Distress and Default Contagion in Financial Networks. *SIAM Journal on Financial Mathematics*, 11(1). [Link](https://epubs.siam.org/doi/10.1137/18M1180542)
- Banerjee, T. et al. (2024). Dynamic Clearing and Contagion in Financial Networks. [Link](https://arxiv.org/pdf/1801.02091)
- Sonin, K. & Sonin, I. (2022). Clearing Payments in Dynamic Financial Networks. [Link](https://arxiv.org/abs/2201.12898)

### Network Contagion

- Battiston, S. et al. (2012). DebtRank: Too Central to Fail? Financial Networks, the FED and Systemic Risk. *Scientific Reports*, 2, 541.
- Acemoglu, D., Ozdaglar, A. & Tahbaz-Salehi, A. (2015). Systemic Risk and Stability in Financial Networks. *American Economic Review*, 105(2), 564–608. [Link](https://economics.mit.edu/sites/default/files/publications/Systemic%20Risk%20and%20Stability%20in%20Financial%20Networks..pdf)
- Bardoscia, M. et al. (2017). Pathways Towards Instability in Financial Networks. *Nature Communications*, 8, 14416.

### Payment System ABMs and Tools

- Galbiati, M. & Soramäki, K. (2011). An agent-based model of payment systems. *Journal of Economic Dynamics and Control*, 35(6), 859–875. [Link](https://ideas.repec.org/a/eee/dyncon/v35y2011i6p859-875.html)
- Bank of Finland. BoF-PSS2 Simulator. [Link](https://www.suomenpankki.fi/en/financial-stability/bof-pss-simulator/product/)
- Soramäki, K. et al. (2007). The Topology of Interbank Payment Flows. *Physica A*, 379(1), 317–333. [NY Fed Staff Report](https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr243.pdf)
- Bookstaber, R. et al. (2014). An Agent-Based Model for Financial Vulnerability. OFR Working Paper. [Link](https://www.financialresearch.gov/working-papers/files/OFRwp2014-05_BookstaberPaddrikTivnan_Agent-basedModelforFinancialVulnerability_revised.pdf)

### Graph-Theoretic Debt Netting

- Gazda, V., Horvath, D. & Resovsky, M. (2015). An Application of Graph Theory in the Process of Mutual Debt Compensation. *Acta Polytechnica Hungarica*, 12(3). [Link](https://acta.uni-obuda.hu/Gazda_Horvath_Resovsky_59.pdf)
- Gavrila, L. & Popa, A. (2020). A Novel Algorithm for Clearing Financial Obligations Between Companies — An Application Within the Romanian Ministry of Economy. [Link](https://arxiv.org/abs/2012.05564)
- Guichon, C. et al. (2025). Integral B2B Debt Netting. [Link](https://hal.science/hal-04947742v2/file/B2BDebtNetting.pdf)

### Gridlock Resolution

- Bech, M. & Soramäki, K. (2001). Gridlock Resolution in Interbank Payment Systems. *Bank of Finland Discussion Papers*. [Link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3018053)
- BIS (2024). Auction-Based Liquidity Saving Mechanisms. *BIS Working Paper 1318*. [Link](https://www.bis.org/publ/work1318.pdf)
- Braine, L. et al. (2024). Improving the Efficiency of Payments Systems Using Quantum Computing. *Management Science*. [Link](https://pubsonline.informs.org/doi/10.1287/mnsc.2023.00314)

### SFC and AB-SFC

- Godley, W. & Lavoie, M. (2007). *Monetary Economics: An Integrated Approach to Credit, Money, Income, Production and Wealth*. Palgrave Macmillan.
- Caiani, A. et al. (2016). Agent-Based Stock-Flow Consistent Macroeconomics: Towards a Benchmark Model. *Journal of Economic Dynamics and Control*, 69, 375–408.

### Dealer / Market-Maker ABMs

- JP Morgan Research. Multi-Agent Simulation for Pricing and Hedging in a Dealer Market. [Link](https://www.jpmorgan.com/content/dam/jpm/cib/complex/content/technology/ai-research-publications/pdf-10.pdf)
- Cont, R. & Bouchaud, J.-P. (2000). Herd Behavior and Aggregate Fluctuations in Financial Markets. *Macroeconomic Dynamics*, 4(2), 170–196.

### Kalecki

- Kalecki, M. (1942). A Theory of Profits. *Economic Journal*, 52, 258–267.
- Setterfield, M. & Budd, A. (2011). A Keynes-Kalecki Model of Cyclical Growth with Agent-Based Features. In *Bentham Science*. [Link](https://link.springer.com/chapter/10.1057/9780230313750_13)
- Laski, K. & Walther, H. (2013). Kalecki's Profit Equation After 80 Years. wiiw Working Paper. [Link](https://wiiw.ac.at/kalecki-s-profit-equation-after-80-years-dlp-3020.pdf)
- Romanchuk, B. (2018). Primer: Kalecki Profit Equation. [Link](http://www.bondeconomics.com/2018/06/primer-kalecki-profit-equation-part-i.html)

### Circular Debt

- Syed Sajid Ali (2010). Dynamics of Circular Debt in Pakistan and Its Resolution. *Lahore Journal of Economics*, 15. [Link](https://lahoreschoolofeconomics.edu.pk/assets/uploads/lje/Volume15/04_Syed_Sajid_EDITED_TTC_11-10-10.pdf)

### Multilateral Netting

- Duffie, D. & Zhu, H. (2011). Does a Central Clearing Counterparty Reduce Counterparty Risk? *Review of Asset Pricing Studies*, 1(1), 74–95.
- Cont, R. & Kokholm, T. (2014). Central Clearing of OTC Derivatives: Bilateral vs Multilateral Netting. *Statistics & Risk Modeling*, 31(1), 3–22.
