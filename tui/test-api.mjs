#!/usr/bin/env node
// Quick test of the API client

const API_URL = 'http://localhost:34589';

async function testHealth() {
  const res = await fetch(`${API_URL}/api/health`);
  console.log('Health:', await res.json());
}

async function testChat() {
  console.log('\n--- Testing chat endpoint ---');
  const res = await fetch(`${API_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: 'Say hello briefly', simple: true }),
  });

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let currentEventType = 'message';
  let eventCount = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.startsWith('event: ')) {
        currentEventType = line.slice(7).trim();
      } else if (line.startsWith('data: ')) {
        try {
          const data = JSON.parse(line.slice(6));
          eventCount++;
          console.log(`[${eventCount}] Event: ${currentEventType}`, JSON.stringify(data).slice(0, 100));
          currentEventType = 'message';
        } catch (e) {
          console.log('Parse error:', e.message);
        }
      }
    }
  }
  console.log(`\nTotal events received: ${eventCount}`);
}

await testHealth();
await testChat();
