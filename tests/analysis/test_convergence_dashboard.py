"""Tests for convergence_dashboard module."""

from pathlib import Path

from bilancio.analysis.convergence import (
    ChannelResult,
    ChannelSnapshot,
    ConvergenceConfig,
    ConvergenceResult,
)
from bilancio.analysis.convergence_dashboard import generate_convergence_dashboard


def _make_trajectory(n_days: int = 5, base: float = 0.5) -> list[ChannelSnapshot]:
    """Create a simple trajectory."""
    snaps = [ChannelSnapshot(day=0, value=base, delta=None)]
    for d in range(1, n_days):
        val = base + 0.01 * d
        snaps.append(ChannelSnapshot(day=d, value=val, delta=0.01))
    return snaps


def _make_result(converged: bool = True, n_channels: int = 2) -> ConvergenceResult:
    """Create a minimal ConvergenceResult for testing."""
    channels: dict[str, ChannelResult] = {}
    names = ["clearing", "default", "price", "belief", "credit", "contagion"]
    for i in range(min(n_channels, len(names))):
        name = names[i]
        channels[name] = ChannelResult(
            name=name,
            converged=converged,
            convergence_day=3 if converged else None,
            final_value=0.95,
            trajectory=_make_trajectory(),
        )
    conv_count = len(channels) if converged else 0
    return ConvergenceResult(
        converged=converged,
        convergence_day=3 if converged else None,
        quality=0.92 if converged else 0.0,
        channels=channels,
        active_channels=len(channels),
        converged_channels=conv_count,
    )


class TestGenerateConvergenceDashboard:
    """Tests for the generate_convergence_dashboard function."""

    def test_generates_html_file(self, tmp_path: Path) -> None:
        """Dashboard generates an HTML file at the specified path."""
        result = _make_result()
        out = tmp_path / "conv.html"
        returned = generate_convergence_dashboard(result, output_path=out)
        assert returned == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_html_contains_plotly(self, tmp_path: Path) -> None:
        """Generated HTML includes Plotly JS and chart divs."""
        result = _make_result()
        out = tmp_path / "conv.html"
        generate_convergence_dashboard(result, output_path=out)
        html = out.read_text()
        assert "plotly" in html.lower()
        assert "<html" in html

    def test_html_contains_status(self, tmp_path: Path) -> None:
        """Generated HTML contains convergence status."""
        result = _make_result(converged=True)
        out = tmp_path / "conv.html"
        generate_convergence_dashboard(result, output_path=out)
        html = out.read_text()
        assert "Converged" in html
        assert "92.0%" in html  # quality

    def test_not_converged_status(self, tmp_path: Path) -> None:
        """Dashboard shows not-converged status."""
        result = _make_result(converged=False)
        out = tmp_path / "conv.html"
        generate_convergence_dashboard(result, output_path=out)
        html = out.read_text()
        assert "Not Converged" in html

    def test_custom_title(self, tmp_path: Path) -> None:
        """Custom title is included in HTML."""
        result = _make_result()
        out = tmp_path / "conv.html"
        generate_convergence_dashboard(result, output_path=out, title="My Dashboard")
        html = out.read_text()
        assert "My Dashboard" in html

    def test_with_config(self, tmp_path: Path) -> None:
        """Custom config is accepted."""
        result = _make_result()
        config = ConvergenceConfig(window=5, epsilon_clearing=0.02)
        out = tmp_path / "conv.html"
        generate_convergence_dashboard(result, output_path=out, config=config)
        assert out.exists()

    def test_multiple_channels(self, tmp_path: Path) -> None:
        """Dashboard handles multiple channels."""
        result = _make_result(n_channels=5)
        out = tmp_path / "conv.html"
        generate_convergence_dashboard(result, output_path=out)
        html = out.read_text()
        # Should have chart sections for each channel
        assert "Clearing" in html
        assert "Default" in html

    def test_empty_channels(self, tmp_path: Path) -> None:
        """Dashboard handles zero channels gracefully."""
        result = ConvergenceResult(
            converged=True,
            convergence_day=None,
            quality=1.0,
            channels={},
            active_channels=0,
            converged_channels=0,
        )
        out = tmp_path / "conv.html"
        generate_convergence_dashboard(result, output_path=out)
        assert out.exists()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Dashboard creates parent directories if needed."""
        result = _make_result()
        out = tmp_path / "sub" / "dir" / "conv.html"
        generate_convergence_dashboard(result, output_path=out)
        assert out.exists()

    def test_channel_timeline_included(self, tmp_path: Path) -> None:
        """Composite timeline chart is included."""
        result = _make_result(n_channels=3)
        out = tmp_path / "conv.html"
        generate_convergence_dashboard(result, output_path=out)
        html = out.read_text()
        assert "Channel Convergence Timeline" in html
