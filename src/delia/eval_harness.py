from lm_eval.api.model import LM
  import httpx

  class DeliaModel(LM):
      """Delia backend for lm-evaluation-harness."""

      def __init__(self, task_type="coder", model_tier=None):
          self.task_type = task_type
          self.model_tier = model_tier
          self.base_url = "http://localhost:8200"  # Delia SSE mode

      def generate_until(self, requests):
          # Call Delia's delegate endpoint
          results = []
          for req in requests:
              resp = httpx.post(f"{self.base_url}/delegate", json={
                  "task": self.task_type,
                  "content": req.args[0],
                  "model": self.model_tier,
              })
              results.append(resp.json()["response"])
          return results

