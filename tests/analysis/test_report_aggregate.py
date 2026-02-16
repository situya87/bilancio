from decimal import Decimal

from bilancio.analysis.report import aggregate_runs, render_dashboard


def test_aggregate_runs_and_dashboard(tmp_path):
    base = tmp_path
    registry_dir = base / "registry"
    runs_dir = base / "runs"
    aggregate_dir = base / "aggregate"
    registry_dir.mkdir()
    run_dir = runs_dir / "grid_abcd"
    out_dir = run_dir / "out"
    out_dir.mkdir(parents=True)
    aggregate_dir.mkdir()

    metrics_csv = out_dir / "metrics.csv"
    metrics_csv.write_text(
        "day,S_t,phi_t,delta_t,G_t,alpha_t,Mpeak_t,v_t,HHIplus_t\n"
        "1,100,1,0,0,0.4,50,2,0.5\n"
        "2,50,0.5,0.5,10,0.2,40,1.5,0.3\n"
    )

    registry_csv = registry_dir / "experiments.csv"
    registry_csv.write_text(
        "run_id,phase,seed,n_agents,kappa,concentration,mu,monotonicity,Q_total,S1,L0,scenario_yaml,events_jsonl,balances_csv,metrics_csv,metrics_html,run_html,status,time_to_stability,phi_total,delta_total,error\n"
        "grid_abcd,grid,42,5,1,0.5,0.25,0,150,150,120,../runs/grid_abcd/scenario.yaml,../runs/grid_abcd/out/events.jsonl,../runs/grid_abcd/out/balances.csv,../runs/grid_abcd/out/metrics.csv,../runs/grid_abcd/out/metrics.html,../runs/grid_abcd/run.html,completed,2,,,\n"
    )

    results_csv = aggregate_dir / "results.csv"
    rows = aggregate_runs(registry_csv, results_csv)

    assert rows
    row = rows[0]
    assert row["phi_total"] == Decimal("0.8333333333333333333333333333")
    assert row["delta_total"] == Decimal("0.1666666666666666666666666667")
    assert row["monotonicity"] == "0"
    assert results_csv.exists()

    dashboard_html = aggregate_dir / "dashboard.html"
    render_dashboard(results_csv, dashboard_html)
    assert dashboard_html.exists()
    html_content = dashboard_html.read_text()
    assert "grid_abcd" in html_content
