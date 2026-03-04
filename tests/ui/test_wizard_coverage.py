"""Coverage tests for wizard.py.

Targets: create_scenario_wizard with different templates and complexity levels,
file template loading, unknown template handling.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml

from bilancio.ui.wizard import create_scenario_wizard


# ============================================================================
# Template-based wizard paths
# ============================================================================


class TestWizardTemplateFile:
    """Cover the file-template branch of create_scenario_wizard."""

    def test_yaml_file_template(self, tmp_path):
        """Loading a YAML file as template."""
        # Create a template file
        template_content = {
            "version": 1,
            "name": "Template Scenario",
            "description": "From template",
            "agents": [{"id": "CB", "kind": "central_bank", "name": "CB"}],
        }
        template_path = tmp_path / "template.yaml"
        with open(template_path, "w") as f:
            yaml.dump(template_content, f)

        output_path = tmp_path / "output.yaml"

        # Mock Prompt.ask to return defaults
        with patch("bilancio.ui.wizard.Prompt.ask", side_effect=["My New Scenario", "New desc"]):
            create_scenario_wizard(output_path, template=str(template_path))

        assert output_path.exists()
        with open(output_path) as f:
            config = yaml.safe_load(f)
        assert config["name"] == "My New Scenario"
        assert config["description"] == "New desc"
        assert len(config["agents"]) == 1


class TestWizardComplexityTemplates:
    """Cover different complexity templates."""

    def test_simple_template(self, tmp_path):
        output_path = tmp_path / "simple.yaml"
        with patch("bilancio.ui.wizard.Prompt.ask", side_effect=["Test", ""]):
            create_scenario_wizard(output_path, template="simple")

        assert output_path.exists()
        with open(output_path) as f:
            config = yaml.safe_load(f)
        assert config["name"] == "Test"
        assert len(config["agents"]) == 4  # CB, B1, H1, H2
        assert len(config["initial_actions"]) == 4

    def test_standard_template(self, tmp_path):
        output_path = tmp_path / "standard.yaml"
        with patch("bilancio.ui.wizard.Prompt.ask", side_effect=["Standard Test", ""]):
            create_scenario_wizard(output_path, template="standard")

        assert output_path.exists()
        with open(output_path) as f:
            config = yaml.safe_load(f)
        assert len(config["agents"]) == 6  # CB, B1, B2, H1, H2, F1

    def test_complex_template(self, tmp_path):
        output_path = tmp_path / "complex.yaml"
        with patch("bilancio.ui.wizard.Prompt.ask", side_effect=["Complex Test", ""]):
            create_scenario_wizard(output_path, template="complex")

        assert output_path.exists()
        with open(output_path) as f:
            config = yaml.safe_load(f)
        assert len(config["agents"]) == 8  # CB, T1, B1, B2, H1, H2, F1, F2
        assert "policy_overrides" in config


class TestWizardUnknownTemplate:
    """Cover the unknown template branch."""

    def test_unknown_template_falls_through(self, tmp_path):
        output_path = tmp_path / "unknown.yaml"
        # unknown template -> asks for complexity -> user says "simple"
        with patch(
            "bilancio.ui.wizard.Prompt.ask",
            side_effect=["simple", "Test", ""],
        ):
            create_scenario_wizard(output_path, template="nonexistent_template")

        assert output_path.exists()
        with open(output_path) as f:
            config = yaml.safe_load(f)
        assert config["name"] == "Test"


class TestWizardNoTemplate:
    """Cover the no-template interactive path."""

    def test_no_template_simple(self, tmp_path):
        output_path = tmp_path / "interactive.yaml"
        with patch(
            "bilancio.ui.wizard.Prompt.ask",
            side_effect=["simple", "Interactive", ""],
        ):
            create_scenario_wizard(output_path, template=None)

        assert output_path.exists()
        with open(output_path) as f:
            config = yaml.safe_load(f)
        assert config["name"] == "Interactive"

    def test_non_interactive_eof(self, tmp_path):
        """When stdin is closed (non-interactive), should use defaults."""
        output_path = tmp_path / "noninteractive.yaml"
        # First call for complexity works, but name/desc raise EOFError
        call_count = 0

        def mock_ask(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "simple"
            raise EOFError

        with patch("bilancio.ui.wizard.Prompt.ask", side_effect=mock_ask):
            create_scenario_wizard(output_path, template=None)

        assert output_path.exists()
        with open(output_path) as f:
            config = yaml.safe_load(f)
        assert config["name"] == "My Scenario"


class TestWizardOutputDirectory:
    """Cover the parent directory creation."""

    def test_creates_parent_dirs(self, tmp_path):
        output_path = tmp_path / "sub" / "dir" / "scenario.yaml"
        with patch("bilancio.ui.wizard.Prompt.ask", side_effect=["simple", "Test", ""]):
            create_scenario_wizard(output_path, template="simple")
        assert output_path.exists()
