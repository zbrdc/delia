# Copyright (C) 2023 the project owner
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import json
from pathlib import Path
import tempfile
import shutil

import pytest

from backend_manager import BackendManager, BackendConfig


class TestBackendManager:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Set up test fixtures."""
        self.test_dir = tmp_path
        self.settings_file = self.test_dir / "settings.json"

        # Create a sample settings file
        self.initial_settings = {
            "version": "1.0",
            "backends": [
                {
                    "id": "ollama-1",
                    "name": "Ollama Local",
                    "provider": "ollama",
                    "type": "local",
                    "url": "http://localhost:11434",
                    "enabled": True,
                    "priority": 1,
                    "models": {
                        "quick": "qwen2.5:14b"
                    }
                },
                {
                    "id": "llamacpp-1",
                    "name": "LlamaCpp",
                    "provider": "llamacpp",
                    "type": "local",
                    "url": "http://localhost:8080",
                    "enabled": False,
                    "priority": 2,
                    "models": {}
                }
            ],
            "routing": {"prefer_local": True}
        }

        with open(self.settings_file, "w") as f:
            json.dump(self.initial_settings, f)

        self.manager = BackendManager(settings_file=self.settings_file)

    def test_load_settings(self):
        """Test loading settings from file."""
        assert len(self.manager.backends) == 2
        assert "ollama-1" in self.manager.backends
        assert "llamacpp-1" in self.manager.backends

        ollama = self.manager.get_backend("ollama-1")
        assert ollama.provider == "ollama"
        assert ollama.enabled is True

        llamacpp = self.manager.get_backend("llamacpp-1")
        assert llamacpp.enabled is False

    def test_get_enabled_backends(self):
        """Test filtering enabled backends."""
        enabled = self.manager.get_enabled_backends()
        assert len(enabled) == 1
        assert enabled[0].id == "ollama-1"

    @pytest.mark.asyncio
    async def test_update_backend(self):
        """Test updating a backend and persistence."""
        updated = await self.manager.update_backend("ollama-1", {"priority": 10, "enabled": False})
        assert updated is not None
        assert updated.priority == 10
        assert updated.enabled is False

        # Verify persistence
        with open(self.settings_file) as f:
            data = json.load(f)
            backends = {b["id"]: b for b in data["backends"]}
            assert backends["ollama-1"]["priority"] == 10
            assert backends["ollama-1"]["enabled"] is False

    @pytest.mark.asyncio
    async def test_add_remove_backend(self):
        """Test adding and removing backends."""
        new_backend = {
            "id": "openai-1",
            "name": "OpenAI",
            "provider": "openai",
            "type": "remote",
            "url": "https://api.openai.com/v1",
            "api_key": "sk-...",
            "enabled": True
        }

        # Add
        backend = self.manager.add_backend(new_backend)
        assert backend.id == "openai-1"
        assert "openai-1" in self.manager.backends

        # Verify file update
        with open(self.settings_file) as f:
            data = json.load(f)
            assert len(data["backends"]) == 3

        # Remove (async)
        result = await self.manager.remove_backend("openai-1")
        assert result is True
        assert "openai-1" not in self.manager.backends

        # Verify file update
        with open(self.settings_file) as f:
            data = json.load(f)
            assert len(data["backends"]) == 2

    @pytest.mark.asyncio
    async def test_active_backend_selection(self):
        """Test active backend logic."""
        # Initial active backend should be the enabled one
        active = self.manager.get_active_backend()
        assert active.id == "ollama-1"

        # Manually set active
        self.manager.set_active_backend("llamacpp-1")
        assert self.manager.get_active_backend().id == "llamacpp-1"

        # Remove active backend (async)
        await self.manager.remove_backend("llamacpp-1")
        # Should fall back to the enabled one
        assert self.manager.get_active_backend().id == "ollama-1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
