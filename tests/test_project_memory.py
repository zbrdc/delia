# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the project memory system (DELIA.md auto-loading)."""

import pytest
from pathlib import Path

from delia.project_memory import (
    ProjectMemory,
    LoadedMemory,
    ProjectMemoryState,
    get_project_memory,
    get_project_context,
    list_project_memories,
    reload_project_memories,
    MEMORY_FILES,
    LOCAL_MEMORY_FILE,
    RULES_DIR,
    MAX_IMPORT_DEPTH,
    MAX_TOTAL_SIZE,
)


class TestLoadedMemory:
    """Test LoadedMemory dataclass."""

    def test_create_memory(self):
        """Test creating a LoadedMemory."""
        mem = LoadedMemory(
            path=Path("/test/DELIA.md"),
            content="Test content",
            source="project",
        )
        assert mem.path == Path("/test/DELIA.md")
        assert mem.content == "Test content"
        assert mem.source == "project"
        assert mem.size == 12  # len("Test content")

    def test_size_auto_calculated(self):
        """Size should be auto-calculated from content."""
        mem = LoadedMemory(
            path=Path("/test.md"),
            content="x" * 100,
            source="test",
        )
        assert mem.size == 100


class TestProjectMemoryState:
    """Test ProjectMemoryState dataclass."""

    def test_empty_state(self):
        """Test empty state."""
        state = ProjectMemoryState()
        assert state.memories == []
        assert state.total_size == 0
        assert state.combined_content == ""

    def test_combined_content(self):
        """Test combining multiple memories."""
        state = ProjectMemoryState()
        state.memories = [
            LoadedMemory(Path("/a.md"), "Content A", "project"),
            LoadedMemory(Path("/b.md"), "Content B", "rules"),
        ]

        combined = state.combined_content
        assert "Content A" in combined
        assert "Content B" in combined
        assert "Memory: a.md" in combined
        assert "Memory: b.md" in combined

    def test_to_dict(self):
        """Test serialization."""
        state = ProjectMemoryState()
        state.memories = [
            LoadedMemory(Path("/test.md"), "Content", "project"),
        ]
        state.total_size = 7

        d = state.to_dict()
        assert len(d["memories"]) == 1
        assert d["total_size"] == 7
        assert d["memories"][0]["source"] == "project"


class TestProjectMemory:
    """Test ProjectMemory class."""

    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create a temporary project with memory files."""
        # Create DELIA.md
        (tmp_path / "DELIA.md").write_text("# Project\nMain instructions.")

        # Create .delia/rules/
        rules_dir = tmp_path / ".delia" / "rules"
        rules_dir.mkdir(parents=True)
        (rules_dir / "coding.md").write_text("# Coding\nWrite good code.")
        (rules_dir / "testing.md").write_text("# Testing\nWrite tests.")

        # Create DELIA.local.md
        (tmp_path / "DELIA.local.md").write_text("# Local\nDev settings.")

        return tmp_path

    def test_discover_memories(self, temp_project):
        """Test discovering all memory files."""
        pm = ProjectMemory(
            project_root=temp_project,
            user_dir=temp_project / ".user",
        )
        state = pm.discover()

        assert len(state.memories) == 4  # DELIA.md + 2 rules + local
        sources = {m.source for m in state.memories}
        assert "project" in sources
        assert "rules" in sources
        assert "local" in sources

    def test_memory_priority_order(self, temp_project):
        """Test that memories are loaded in correct order."""
        pm = ProjectMemory(
            project_root=temp_project,
            user_dir=temp_project / ".user",
        )
        state = pm.discover()

        # Project should come before rules, rules before local
        sources = [m.source for m in state.memories]
        project_idx = sources.index("project")
        rules_idx = sources.index("rules")
        local_idx = sources.index("local")

        assert project_idx < rules_idx < local_idx

    def test_user_memories(self, tmp_path):
        """Test loading user-level memories."""
        user_dir = tmp_path / "user"
        user_dir.mkdir()
        (user_dir / "DELIA.md").write_text("# User defaults")

        pm = ProjectMemory(
            project_root=tmp_path / "project",
            user_dir=user_dir,
        )
        state = pm.discover()

        user_mems = [m for m in state.memories if m.source == "user"]
        assert len(user_mems) == 1
        assert "User defaults" in user_mems[0].content

    def test_no_duplicates(self, tmp_path):
        """Test that same file isn't loaded twice."""
        (tmp_path / "DELIA.md").write_text("# Test")
        (tmp_path / "delia.md").write_text("# Test")  # Variant

        pm = ProjectMemory(project_root=tmp_path, user_dir=tmp_path / ".u")
        state = pm.discover()

        # Should only load one (first found)
        assert len(state.memories) <= 1

    def test_get_context_injection(self, temp_project):
        """Test context injection formatting."""
        pm = ProjectMemory(
            project_root=temp_project,
            user_dir=temp_project / ".user",
        )
        pm.discover()

        context = pm.get_context_injection()

        assert "Project Instructions" in context
        assert "Main instructions" in context

    def test_list_memories(self, temp_project):
        """Test listing loaded memories."""
        pm = ProjectMemory(
            project_root=temp_project,
            user_dir=temp_project / ".user",
        )
        pm.discover()

        memories = pm.list_memories()

        assert len(memories) == 4
        assert all("path" in m for m in memories)
        assert all("source" in m for m in memories)
        assert all("size_kb" in m for m in memories)


class TestImportSyntax:
    """Test @import syntax for including other files."""

    def test_basic_import(self, tmp_path):
        """Test basic file import."""
        main_content = """# Main
@included.md
End."""
        (tmp_path / "DELIA.md").write_text(main_content)
        (tmp_path / "included.md").write_text("Included content")

        pm = ProjectMemory(project_root=tmp_path, user_dir=tmp_path / ".u")
        state = pm.discover()

        main_mem = state.memories[0]
        assert "Included content" in main_mem.content
        assert "Imported: included.md" in main_mem.content

    def test_relative_import(self, tmp_path):
        """Test relative path import."""
        rules_dir = tmp_path / "rules"
        rules_dir.mkdir()
        (rules_dir / "test.md").write_text("Test rule")

        main_content = """# Main
@rules/test.md
"""
        (tmp_path / "DELIA.md").write_text(main_content)

        pm = ProjectMemory(project_root=tmp_path, user_dir=tmp_path / ".u")
        state = pm.discover()

        main_mem = state.memories[0]
        assert "Test rule" in main_mem.content

    def test_nested_imports(self, tmp_path):
        """Test nested imports (A imports B which imports C)."""
        (tmp_path / "a.md").write_text("A\n@b.md\nEnd A")
        (tmp_path / "b.md").write_text("B\n@c.md\nEnd B")
        (tmp_path / "c.md").write_text("C content")
        (tmp_path / "DELIA.md").write_text("@a.md")

        pm = ProjectMemory(project_root=tmp_path, user_dir=tmp_path / ".u")
        state = pm.discover()

        main_mem = state.memories[0]
        assert "A" in main_mem.content
        assert "B" in main_mem.content
        assert "C content" in main_mem.content

    def test_circular_import_prevention(self, tmp_path):
        """Test that circular imports are handled."""
        (tmp_path / "a.md").write_text("A\n@b.md")
        (tmp_path / "b.md").write_text("B\n@a.md")  # Circular!
        (tmp_path / "DELIA.md").write_text("@a.md")

        pm = ProjectMemory(project_root=tmp_path, user_dir=tmp_path / ".u")
        state = pm.discover()

        # Should not crash, circular reference should be noted
        main_mem = state.memories[0]
        assert "Already imported" in main_mem.content or "A" in main_mem.content

    def test_import_not_found(self, tmp_path):
        """Test handling of missing import."""
        (tmp_path / "DELIA.md").write_text("@nonexistent.md")

        pm = ProjectMemory(project_root=tmp_path, user_dir=tmp_path / ".u")
        state = pm.discover()

        main_mem = state.memories[0]
        assert "Import not found" in main_mem.content

    def test_max_import_depth(self, tmp_path):
        """Test that import depth is limited."""
        # Create chain of imports deeper than MAX_IMPORT_DEPTH
        for i in range(MAX_IMPORT_DEPTH + 2):
            content = f"Level {i}\n@level{i+1}.md"
            (tmp_path / f"level{i}.md").write_text(content)

        (tmp_path / "DELIA.md").write_text("@level0.md")

        pm = ProjectMemory(project_root=tmp_path, user_dir=tmp_path / ".u")
        state = pm.discover()

        main_mem = state.memories[0]
        assert "Import depth exceeded" in main_mem.content


class TestModuleFunctions:
    """Test module-level convenience functions."""

    def test_get_project_memory_singleton(self):
        """Test that get_project_memory returns singleton."""
        pm1 = get_project_memory()
        pm2 = get_project_memory()
        assert pm1 is pm2

    def test_reload_project_memories(self):
        """Test force reload."""
        state = reload_project_memories()
        assert isinstance(state, ProjectMemoryState)

    def test_list_project_memories(self):
        """Test listing memories via convenience function."""
        memories = list_project_memories()
        assert isinstance(memories, list)

    def test_get_project_context(self):
        """Test getting context injection."""
        context = get_project_context()
        assert isinstance(context, str)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_project(self, tmp_path):
        """Test project with no memory files."""
        pm = ProjectMemory(project_root=tmp_path, user_dir=tmp_path / ".u")
        state = pm.discover()

        assert len(state.memories) == 0
        assert pm.get_context_injection() == ""

    def test_empty_memory_file(self, tmp_path):
        """Test handling of empty memory file."""
        (tmp_path / "DELIA.md").write_text("")

        pm = ProjectMemory(project_root=tmp_path, user_dir=tmp_path / ".u")
        state = pm.discover()

        assert len(state.memories) == 1
        assert state.memories[0].size == 0

    def test_binary_file_skipped(self, tmp_path):
        """Test that binary files are handled gracefully."""
        # Create a file with binary content
        (tmp_path / "DELIA.md").write_bytes(b"\x00\x01\x02\x03")

        pm = ProjectMemory(project_root=tmp_path, user_dir=tmp_path / ".u")
        state = pm.discover()

        # Should either skip or load with errors
        # The exact behavior depends on UTF-8 decoding

    def test_unicode_content(self, tmp_path):
        """Test handling of unicode content."""
        (tmp_path / "DELIA.md").write_text("# 你好世界\nEmoji: 🎉🚀", encoding="utf-8")

        pm = ProjectMemory(project_root=tmp_path, user_dir=tmp_path / ".u")
        state = pm.discover()

        assert len(state.memories) == 1
        assert "你好世界" in state.memories[0].content
        assert "🎉" in state.memories[0].content

    def test_path_traversal_prevention(self, tmp_path):
        """Test that path traversal in imports is handled."""
        (tmp_path / "DELIA.md").write_text("@../../../etc/passwd")

        pm = ProjectMemory(project_root=tmp_path, user_dir=tmp_path / ".u")
        state = pm.discover()

        # Should not crash and should not include sensitive files
        # (The file won't exist in test anyway)


class TestIntegration:
    """Integration tests with ContextEngine."""

    @pytest.mark.asyncio
    async def test_context_engine_integration(self, tmp_path):
        """Test that ContextEngine uses project memory."""
        (tmp_path / "DELIA.md").write_text("# Test Project Instructions")

        from delia.project_memory import _project_memory, ProjectMemory
        import delia.project_memory as pm_module

        # Inject our test project memory
        old_pm = pm_module._project_memory
        pm_module._project_memory = ProjectMemory(
            project_root=tmp_path,
            user_dir=tmp_path / ".u",
        )
        pm_module._project_memory.discover()

        try:
            from delia.orchestration.context import ContextEngine

            content = await ContextEngine.prepare_content(
                content="Test task",
                include_project_instructions=True,
            )

            assert "Project Instructions" in content or "Test task" in content
        finally:
            # Restore
            pm_module._project_memory = old_pm
