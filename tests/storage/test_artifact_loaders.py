"""Tests for artifact loader implementations."""

from pathlib import Path

import pytest

from bilancio.storage.artifact_loaders import ArtifactLoader, LocalArtifactLoader


class TestLocalArtifactLoaderLoadText:
    """Tests for LocalArtifactLoader.load_text()."""

    def test_load_text_returns_file_contents(self, tmp_path: Path):
        """load_text() returns text content of a file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        loader = LocalArtifactLoader(tmp_path)
        result = loader.load_text("test.txt")

        assert result == "Hello, World!"

    def test_load_text_handles_unicode(self, tmp_path: Path):
        """load_text() correctly handles unicode content."""
        test_file = tmp_path / "unicode.txt"
        test_file.write_text("Hello, \u4e16\u754c! \u00e9\u00e8\u00ea", encoding="utf-8")

        loader = LocalArtifactLoader(tmp_path)
        result = loader.load_text("unicode.txt")

        assert result == "Hello, \u4e16\u754c! \u00e9\u00e8\u00ea"

    def test_load_text_handles_multiline_content(self, tmp_path: Path):
        """load_text() preserves multiline content."""
        test_file = tmp_path / "multiline.txt"
        content = "line1\nline2\nline3"
        test_file.write_text(content)

        loader = LocalArtifactLoader(tmp_path)
        result = loader.load_text("multiline.txt")

        assert result == content

    def test_load_text_raises_for_missing_file(self, tmp_path: Path):
        """load_text() raises FileNotFoundError for missing files."""
        loader = LocalArtifactLoader(tmp_path)

        with pytest.raises(FileNotFoundError):
            loader.load_text("nonexistent.txt")

    def test_load_text_nested_path(self, tmp_path: Path):
        """load_text() handles nested path references."""
        nested_dir = tmp_path / "subdir" / "deeper"
        nested_dir.mkdir(parents=True)
        test_file = nested_dir / "nested.txt"
        test_file.write_text("nested content")

        loader = LocalArtifactLoader(tmp_path)
        result = loader.load_text("subdir/deeper/nested.txt")

        assert result == "nested content"


class TestLocalArtifactLoaderLoadBytes:
    """Tests for LocalArtifactLoader.load_bytes()."""

    def test_load_bytes_returns_file_contents(self, tmp_path: Path):
        """load_bytes() returns binary content of a file."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"\x00\x01\x02\x03")

        loader = LocalArtifactLoader(tmp_path)
        result = loader.load_bytes("test.bin")

        assert result == b"\x00\x01\x02\x03"

    def test_load_bytes_handles_text_files(self, tmp_path: Path):
        """load_bytes() can read text files as bytes."""
        test_file = tmp_path / "text.txt"
        test_file.write_text("Hello, World!")

        loader = LocalArtifactLoader(tmp_path)
        result = loader.load_bytes("text.txt")

        assert result == b"Hello, World!"

    def test_load_bytes_raises_for_missing_file(self, tmp_path: Path):
        """load_bytes() raises FileNotFoundError for missing files."""
        loader = LocalArtifactLoader(tmp_path)

        with pytest.raises(FileNotFoundError):
            loader.load_bytes("nonexistent.bin")

    def test_load_bytes_nested_path(self, tmp_path: Path):
        """load_bytes() handles nested path references."""
        nested_dir = tmp_path / "data"
        nested_dir.mkdir()
        test_file = nested_dir / "file.bin"
        test_file.write_bytes(b"\xff\xfe")

        loader = LocalArtifactLoader(tmp_path)
        result = loader.load_bytes("data/file.bin")

        assert result == b"\xff\xfe"


class TestLocalArtifactLoaderExists:
    """Tests for LocalArtifactLoader.exists()."""

    def test_exists_returns_true_for_existing_file(self, tmp_path: Path):
        """exists() returns True when file exists."""
        test_file = tmp_path / "exists.txt"
        test_file.write_text("content")

        loader = LocalArtifactLoader(tmp_path)
        result = loader.exists("exists.txt")

        assert result is True

    def test_exists_returns_false_for_missing_file(self, tmp_path: Path):
        """exists() returns False when file does not exist."""
        loader = LocalArtifactLoader(tmp_path)
        result = loader.exists("nonexistent.txt")

        assert result is False

    def test_exists_handles_nested_paths(self, tmp_path: Path):
        """exists() handles nested path references."""
        nested_dir = tmp_path / "subdir"
        nested_dir.mkdir()
        test_file = nested_dir / "file.txt"
        test_file.write_text("content")

        loader = LocalArtifactLoader(tmp_path)

        assert loader.exists("subdir/file.txt") is True
        assert loader.exists("subdir/missing.txt") is False

    def test_exists_returns_false_for_directory(self, tmp_path: Path):
        """exists() returns True for directory (matches Path.exists behavior)."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        loader = LocalArtifactLoader(tmp_path)
        # Note: Path.exists() returns True for directories
        assert loader.exists("subdir") is True


class TestLocalArtifactLoaderInit:
    """Tests for LocalArtifactLoader initialization."""

    def test_init_stores_base_path(self, tmp_path: Path):
        """__init__ stores base_path as Path object."""
        loader = LocalArtifactLoader(tmp_path)
        assert loader.base_path == tmp_path

    def test_init_converts_string_to_path(self, tmp_path: Path):
        """__init__ converts string path to Path object."""
        loader = LocalArtifactLoader(str(tmp_path))
        assert isinstance(loader.base_path, Path)
        assert loader.base_path == tmp_path

    def test_base_path_not_required_to_exist(self):
        """__init__ does not require base_path to exist."""
        # This should not raise
        loader = LocalArtifactLoader(Path("/nonexistent/path"))
        assert loader.base_path == Path("/nonexistent/path")


class TestLocalArtifactLoaderProtocol:
    """Tests for LocalArtifactLoader implementing ArtifactLoader protocol."""

    def test_implements_artifact_loader_protocol(self, tmp_path: Path):
        """LocalArtifactLoader implements ArtifactLoader protocol."""
        loader = LocalArtifactLoader(tmp_path)
        # ArtifactLoader is a runtime_checkable Protocol
        assert isinstance(loader, ArtifactLoader)

    def test_has_load_text_method(self, tmp_path: Path):
        """LocalArtifactLoader has load_text method."""
        loader = LocalArtifactLoader(tmp_path)
        assert hasattr(loader, "load_text")
        assert callable(loader.load_text)

    def test_has_load_bytes_method(self, tmp_path: Path):
        """LocalArtifactLoader has load_bytes method."""
        loader = LocalArtifactLoader(tmp_path)
        assert hasattr(loader, "load_bytes")
        assert callable(loader.load_bytes)

    def test_has_exists_method(self, tmp_path: Path):
        """LocalArtifactLoader has exists method."""
        loader = LocalArtifactLoader(tmp_path)
        assert hasattr(loader, "exists")
        assert callable(loader.exists)
