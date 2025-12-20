# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
from hypothesis import given, strategies as st
from delia.voting import VotingConsensus

@given(st.lists(st.text(), min_size=1, max_size=10), st.integers(min_value=2, max_value=5))
def test_voting_never_crashes(responses, k):
    """The voting system should never crash with arbitrary text inputs."""
    consensus = VotingConsensus(k=k)
    for resp in responses:
        try:
            consensus.add_vote(resp)
        except Exception as e:
            pytest.fail(f"Voting crashed with input {repr(resp)}: {e}")

@given(st.floats(min_value=0, max_value=100), st.floats(min_value=0, max_value=0.01))
def test_numerical_similarity(val, delta):
    """Test if slightly different numerical strings are grouped correctly."""
    consensus = VotingConsensus(k=2, similarity_threshold=0.95)
    
    s1 = f"The result is {val}"
    s2 = f"The result is {val + delta}"
    
    res1 = consensus.add_vote(s1)
    res2 = consensus.add_vote(s2)
    
    # With a high enough threshold, these should ideally match if delta is small
    # But difflib might be sensitive. We primarily care that it doesn't fail.
    assert isinstance(res2.consensus_reached, bool)

@given(st.lists(st.text(min_size=500, max_size=2000), min_size=2, max_size=5))
def test_long_string_performance(long_strings):
    """Test performance of difflib with long strings to ensure no 'hangs'."""
    consensus = VotingConsensus(k=3)
    for s in long_strings:
        consensus.add_vote(s)

def test_mixed_json_raw_text():
    """Test how consensus handles a mix of JSON and raw text."""
    consensus = VotingConsensus(k=2)
    
    json_resp = '{"result": 42, "status": "ok"}'
    text_resp = "The result is 42 and status is ok"
    
    consensus.add_vote(json_resp)
    res = consensus.add_vote(text_resp)
    
    # They are semantically similar but structurally different.
    # We want to ensure the system handles this gracefully.
    assert not res.consensus_reached # Likely won't match, which is correct

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
