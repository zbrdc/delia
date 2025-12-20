# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import uuid
from delia.session_manager import SessionManager
from delia.types import Workspace

@pytest.fixture
def session_manager(tmp_path):
    return SessionManager(session_dir=tmp_path / "sessions")

@pytest.mark.parametrize("i", range(100))
def test_session_lifecycle_fuzzing(session_manager, i):
    """Test 100 unique session lifecycles."""
    # Create
    session = session_manager.create_session()
    session_id = session.session_id
    
    # Add messages
    session.add_message("user", f"hello {i}")
    session.add_message("assistant", f"hi {i}")
    
    # Update metadata
    session.metadata["test_run"] = i
    session_manager._save_session(session_id) # implementation detail check
    
    # Reload
    reloaded = session_manager.get_session(session_id)
    assert reloaded.messages[0].content == f"hello {i}"
    assert reloaded.metadata["test_run"] == i
    
    # Cleanup every 10th session
    if i % 10 == 0:
        session_manager.delete_session(session_id)
        assert session_manager.get_session(session_id) is None

def test_context_window_clamping_fuzzing(session_manager):
    """Test context window behavior with 50+ message turns."""
    session = session_manager.create_session()
    
    for i in range(100):
        session.add_message("user", "very long message " * 100)
        session.add_message("assistant", "response " * 50)
        
    # Verify it handles large history
    assert len(session.messages) == 200
    
def test_parallel_session_access(session_manager):
    """Test concurrent access to multiple sessions."""
    import threading
    
    def worker(idx):
        s = session_manager.create_session(client_id=f"user-{idx}")
        s.add_message("user", "ping")
        # Manager saves automatically on create, but let's force another save
        # Note: list_sessions calls load_all_from_disk
        
    threads = []
    for i in range(20):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    assert len(session_manager.list_sessions()) >= 20
