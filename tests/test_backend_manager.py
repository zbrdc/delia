import unittest
import json
from pathlib import Path
import tempfile
import shutil
from unittest.mock import MagicMock, patch

from backend_manager import BackendManager, BackendConfig

class TestBackendManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.settings_file = Path(self.test_dir) / "settings.json"
        
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

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_load_settings(self):
        """Test loading settings from file."""
        self.assertEqual(len(self.manager.backends), 2)
        self.assertIn("ollama-1", self.manager.backends)
        self.assertIn("llamacpp-1", self.manager.backends)
        
        ollama = self.manager.get_backend("ollama-1")
        self.assertEqual(ollama.provider, "ollama")
        self.assertTrue(ollama.enabled)
        
        llamacpp = self.manager.get_backend("llamacpp-1")
        self.assertFalse(llamacpp.enabled)

    def test_get_enabled_backends(self):
        """Test filtering enabled backends."""
        enabled = self.manager.get_enabled_backends()
        self.assertEqual(len(enabled), 1)
        self.assertEqual(enabled[0].id, "ollama-1")

    def test_update_backend(self):
        """Test updating a backend and persistence."""
        updated = self.manager.update_backend("ollama-1", {"priority": 10, "enabled": False})
        self.assertIsNotNone(updated)
        self.assertEqual(updated.priority, 10)
        self.assertFalse(updated.enabled)
        
        # Verify persistence
        with open(self.settings_file) as f:
            data = json.load(f)
            backends = {b["id"]: b for b in data["backends"]}
            self.assertEqual(backends["ollama-1"]["priority"], 10)
            self.assertFalse(backends["ollama-1"]["enabled"])

    def test_add_remove_backend(self):
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
        self.assertEqual(backend.id, "openai-1")
        self.assertIn("openai-1", self.manager.backends)
        
        # Verify file update
        with open(self.settings_file) as f:
            data = json.load(f)
            self.assertEqual(len(data["backends"]), 3)

        # Remove
        result = self.manager.remove_backend("openai-1")
        self.assertTrue(result)
        self.assertNotIn("openai-1", self.manager.backends)
        
        # Verify file update
        with open(self.settings_file) as f:
            data = json.load(f)
            self.assertEqual(len(data["backends"]), 2)

    def test_active_backend_selection(self):
        """Test active backend logic."""
        # Initial active backend should be the enabled one
        active = self.manager.get_active_backend()
        self.assertEqual(active.id, "ollama-1")
        
        # Manually set active
        self.manager.set_active_backend("llamacpp-1")
        self.assertEqual(self.manager.get_active_backend().id, "llamacpp-1")
        
        # Remove active backend
        self.manager.remove_backend("llamacpp-1")
        # Should fall back to the enabled one
        self.assertEqual(self.manager.get_active_backend().id, "ollama-1")

if __name__ == "__main__":
    unittest.main()
