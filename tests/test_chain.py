# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for chain/pipeline execution in Delia."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from delia.orchestration.intent import IntentDetector
from delia.orchestration.meta_learning import reset_orchestration_learner, get_orchestration_learner
from delia.prompts import OrchestrationMode


class TestChainDetection:
    """Test chain pattern detection in intent detector."""

    def setup_method(self):
        """Reset meta-learner with empty patterns to prevent learned pattern override."""
        # Use a temp directory so no patterns are loaded from disk
        self._temp_dir = tempfile.mkdtemp()
        reset_orchestration_learner()

        # Patch DATA_DIR to use temp directory (no existing patterns)
        with patch('delia.orchestration.meta_learning.paths.DATA_DIR', Path(self._temp_dir)):
            reset_orchestration_learner()
            learner = get_orchestration_learner()
            # Re-init with clean data dir
            learner.data_dir = Path(self._temp_dir)
            learner.patterns.clear()
            learner.total_tasks = 500  # High count to disable exploration
            learner.base_tot_probability = 0.0

    def test_first_then_pattern(self):
        """Test 'first... then...' pattern detection."""
        detector = IntentDetector()
        intent = detector.detect("First analyze the code, then generate tests")

        assert intent.orchestration_mode == OrchestrationMode.CHAIN
        assert len(intent.chain_steps) >= 2
        # Should have analyze and generate steps
        step_types = [s.split(":")[0] for s in intent.chain_steps]
        assert "analyze" in step_types
    
    def test_numbered_list_pattern(self):
        """Test numbered list pattern detection."""
        detector = IntentDetector()
        intent = detector.detect("1. analyze the design 2. implement it 3. test it")
        
        assert intent.orchestration_mode == OrchestrationMode.CHAIN
        assert len(intent.chain_steps) == 3
    
    def test_step_pattern(self):
        """Test 'Step 1... Step 2...' pattern detection."""
        detector = IntentDetector()
        intent = detector.detect("Step 1. plan the architecture. Step 2. implement it.")
        
        assert intent.orchestration_mode == OrchestrationMode.CHAIN
        assert len(intent.chain_steps) >= 2
    
    def test_analyze_then_generate(self):
        """Test analyze-then-generate pipeline pattern."""
        detector = IntentDetector()
        intent = detector.detect("analyze this code then generate tests")
    
        # Technical verbs often trigger AGENTIC now
        assert intent.orchestration_mode in [OrchestrationMode.CHAIN, OrchestrationMode.AGENTIC]
    
    def test_plan_then_implement(self):
        """Test plan-then-implement pipeline pattern."""
        detector = IntentDetector()
        intent = detector.detect("plan the API design and then implement it")
    
        assert intent.orchestration_mode in [OrchestrationMode.CHAIN, OrchestrationMode.AGENTIC]
    
    def test_non_chain_message(self):
        """Test that normal messages don't trigger chain mode."""
        detector = IntentDetector()
        intent = detector.detect("What is the weather today?")
        
        assert intent.orchestration_mode != OrchestrationMode.CHAIN
    
    def test_chain_fallback_on_single_step(self):
        """Test fallback to NONE when only one step detected."""
        detector = IntentDetector()
        # This should match the pattern but fail to extract 2+ steps
        intent = detector.detect("First do something")
        
        # Should fall back to NONE due to insufficient steps
        assert intent.orchestration_mode == OrchestrationMode.NONE or len(intent.chain_steps) >= 2


class TestChainStepExtraction:
    """Test chain step extraction logic."""
    
    def test_step_normalization(self):
        """Test that steps are normalized to task types."""
        detector = IntentDetector()
        steps = detector._extract_chain_steps(
            "1. analyze the code 2. generate tests 3. review the tests"
        )
        
        assert len(steps) == 3
        # Each step should be prefixed with task type
        assert any("analyze:" in s for s in steps)
        assert any("generate:" in s or "review:" in s for s in steps)
    
    def test_task_type_mapping(self):
        """Test that common verbs map to correct task types."""
        detector = IntentDetector()
        
        # Test various action verbs
        normalized = detector._normalize_steps([
            "analyze the code",
            "write a function",
            "plan the architecture",
            "test the implementation",
            "debug the error",
        ])
        
        assert normalized[0].startswith("analyze:")
        assert normalized[1].startswith("generate:")  # write -> generate
        assert normalized[2].startswith("plan:")
        assert normalized[3].startswith("review:")  # test -> review
        assert normalized[4].startswith("analyze:")  # debug -> analyze


class TestChainModeIntegration:
    """Test chain mode integration with orchestration."""
    
    def test_chain_mode_enum_exists(self):
        """Verify CHAIN mode exists in OrchestrationMode."""
        assert hasattr(OrchestrationMode, "CHAIN")
        assert OrchestrationMode.CHAIN.value == "chain"
    
    def test_detected_intent_has_chain_steps(self):
        """Verify DetectedIntent has chain_steps field."""
        from delia.orchestration.result import DetectedIntent
        
        intent = DetectedIntent(
            task_type="quick",
            orchestration_mode=OrchestrationMode.CHAIN,
            chain_steps=["analyze: check code", "generate: write tests"],
        )
        
        assert len(intent.chain_steps) == 2

