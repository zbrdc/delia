# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import tempfile
import shutil
import os
from pathlib import Path
from hypothesis import given, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule, precondition, invariant, initialize
from delia.session_manager import SessionManager
from delia.melons import MelonTracker

class DeliaStatefulFuzzer(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.temp_dir = None
        self.session_dir = None
        self.sm = None
        self.mt = None
        self.sessions = []
        self.models = ["model-a", "model-b", "model-c"]
        self.task_types = ["quick", "coder", "moe"]

    @initialize()
    def setup(self):
        self.temp_dir = tempfile.mkdtemp(prefix="delia-fuzz-stateful-")
        self.session_dir = Path(self.temp_dir) / "sessions"
        self.session_dir.mkdir()
        
        self.sm = SessionManager(session_dir=self.session_dir, max_sessions=10)
        self.mt = MelonTracker(stats_file=Path(self.temp_dir) / "melons.json")
        self.sessions = []

    @rule(client_id=st.one_of(st.none(), st.text(min_size=1, max_size=10)))
    def create_session(self, client_id):
        session = self.sm.create_session(client_id=client_id)
        self.sessions.append(session.session_id)

    @rule(session_index=st.integers(), 
          role=st.sampled_from(["user", "assistant", "system"]),
          content=st.text(),
          tokens=st.integers(min_value=0, max_value=1000))
    def add_message(self, session_index, role, content, tokens):
        if not self.sessions:
            return
        sid = self.sessions[session_index % len(self.sessions)]
        self.sm.add_to_session(sid, role, content, tokens=tokens)

    @rule(session_index=st.integers())
    def delete_session(self, session_index):
        if not self.sessions:
            return
        idx = session_index % len(self.sessions)
        sid = self.sessions.pop(idx)
        self.sm.delete_session(sid)

    @rule(model_index=st.integers(),
          task_index=st.integers(),
          savings_cents=st.integers(min_value=1, max_value=100))
    def record_savings(self, model_index, task_index, savings_cents):
        """Record savings which converts to melons (1 melon = $0.001)."""
        model = self.models[model_index % len(self.models)]
        task = self.task_types[task_index % len(self.task_types)]
        # Convert cents to USD for savings
        savings_usd = savings_cents / 100.0
        self.mt.record_savings(model, task, savings_usd=savings_usd)

    @invariant()
    def check_consistency(self):
        # Ensure session directory doesn't explode
        if self.session_dir.exists():
            files = list(self.session_dir.glob("*.json"))
            # Due to eviction, files might be fewer than total created,
            # but shouldn't exceed max_sessions + buffer
            assert len(files) <= 15 

    def teardown(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

test_delia_state = DeliaStatefulFuzzer.TestCase
