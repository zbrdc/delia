from delia.orchestration.intent import IntentDetector
import re
detector = IntentDetector()
msg = "1. analyze the design 2. implement it 3. test it"
print(f"Detecting: {msg}")
intent = detector.detect(msg)
print(f"Intent: {intent.orchestration_mode}")
print(f"Confidence: {intent.confidence}")

# Test individual patterns
for i, pat in enumerate(detector.all_patterns):
    if pat.pattern.search(msg):
        print(f"Matched pattern {i}: {pat.reasoning} -> {pat.orchestration_mode}")
