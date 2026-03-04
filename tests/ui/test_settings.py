"""Tests for bilancio.ui.settings module."""

from __future__ import annotations

from bilancio.ui.settings import DEFAULT_SETTINGS, Settings


class TestSettingsDefaults:
    """Test that Settings dataclass has correct default values."""

    def test_default_console_width(self):
        s = Settings()
        assert s.console_width == 100

    def test_default_currency_denom(self):
        s = Settings()
        assert s.currency_denom == "X"

    def test_default_currency_precision(self):
        s = Settings()
        assert s.currency_precision == 2

    def test_default_html_inline_styles(self):
        s = Settings()
        assert s.html_inline_styles is True

    def test_default_balance_table_width(self):
        s = Settings()
        assert s.balance_table_width == 80

    def test_default_event_panel_width(self):
        s = Settings()
        assert s.event_panel_width == 90

    def test_default_zero_threshold(self):
        s = Settings()
        assert s.zero_threshold == 0.01

    def test_default_max_event_lines(self):
        s = Settings()
        assert s.max_event_lines == 10

    def test_default_show_event_icons(self):
        s = Settings()
        assert s.show_event_icons is True

    def test_default_phase_colors(self):
        s = Settings()
        assert s.phase_colors == {"A": "cyan", "B": "yellow", "C": "green"}


class TestSettingsCustom:
    """Test custom Settings construction."""

    def test_custom_console_width(self):
        s = Settings(console_width=120)
        assert s.console_width == 120

    def test_custom_currency_denom(self):
        s = Settings(currency_denom="USD")
        assert s.currency_denom == "USD"

    def test_custom_phase_colors(self):
        colors = {"A": "red", "B": "blue", "C": "magenta"}
        s = Settings(phase_colors=colors)
        assert s.phase_colors == colors


class TestSettingsFrozen:
    """Test that Settings is immutable."""

    def test_frozen_raises_on_attribute_set(self):
        s = Settings()
        import dataclasses

        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            s.console_width = 200  # type: ignore[misc]


class TestDefaultSettingsInstance:
    """Test the module-level DEFAULT_SETTINGS instance."""

    def test_default_settings_is_settings(self):
        assert isinstance(DEFAULT_SETTINGS, Settings)

    def test_default_settings_has_default_values(self):
        assert DEFAULT_SETTINGS.console_width == 100
        assert DEFAULT_SETTINGS.currency_denom == "X"
        assert DEFAULT_SETTINGS.currency_precision == 2
        assert DEFAULT_SETTINGS.html_inline_styles is True
        assert DEFAULT_SETTINGS.balance_table_width == 80
        assert DEFAULT_SETTINGS.event_panel_width == 90
        assert DEFAULT_SETTINGS.zero_threshold == 0.01
        assert DEFAULT_SETTINGS.max_event_lines == 10
        assert DEFAULT_SETTINGS.show_event_icons is True
        assert DEFAULT_SETTINGS.phase_colors == {"A": "cyan", "B": "yellow", "C": "green"}
