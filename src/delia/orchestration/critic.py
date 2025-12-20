# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Senior QA Critic Implementation.

Uses a high-fidelity model to verify the output of the Executor
before it reaches the user.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from .result import DetectedIntent
from ..prompts import ModelRole, build_system_prompt

if TYPE_CHECKING:
    from ..backend_manager import BackendConfig

log = structlog.get_logger()

class ResponseCritic:
    """
    Acts as the final verification step in the orchestration loop.
    
    The Critic reviews the work of the Executor and either 'Approves'
    it or provides feedback for correction.
    """

    def __init__(self, call_llm_fn: Any):
        self.call_llm = call_llm_fn

    async def verify(
        self,
        original_prompt: str,
        response_to_verify: str,
        backend_obj: BackendConfig | None = None,
    ) -> tuple[bool, str]:
        """
        Verifies the response against the original prompt.
        
        Returns:
            (is_approved, feedback_or_original_text)
        """
        from ..config import config
        
        # Use 'quick' tier for speed, or a dedicated critic if configured
        critic_model = getattr(config, 'model_critic', config.model_quick).default_model
        
        system_prompt = build_system_prompt(ModelRole.CRITIC)
        
        verification_prompt = f"""### Original Request:
{original_prompt}

### Assistant Response to Review:
{response_to_verify}

---
Does this response accurately and fully address the request? 
Respond with 'APPROVED' or provide feedback."""

        log.info("critic_verification_start", model=critic_model)

        result = await self.call_llm(
            model=critic_model,
            prompt=verification_prompt,
            system=system_prompt,
            task_type="critique",
            backend_obj=backend_obj,
            temperature=0.1, # Highly deterministic
        )

        if not result.get("success"):
            log.warning("critic_failed_skipping_verification")
            return True, response_to_verify # Fail open to avoid blocking user

        critic_response = result.get("response", "").strip()
        is_approved = "APPROVED" in critic_response or "approved" in critic_response.lower()

        if is_approved:
            log.info("critic_approved_response")
            return True, response_to_verify
        else:
            log.warning("critic_rejected_response", feedback_len=len(critic_response))
            return False, critic_response
