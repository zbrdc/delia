import json
from typing import Any

def process_data(data: dict[str, Any]) -> str:
    """Process incoming data."""
    if not data:
        return ""
    return json.dumps(data)

class DataProcessor:
    def __init__(self):
        self.cache = {}

    async def async_process(self, items: list) -> list:
        results = []
        for item in items:
            result = await self._handle_item(item)
            results.append(result)
        return results
