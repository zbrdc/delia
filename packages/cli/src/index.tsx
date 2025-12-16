#!/usr/bin/env node
/**
 * Delia CLI - Rich terminal interface for local LLM orchestration.
 *
 * This is the TypeScript frontend that connects to the Python backend
 * via HTTP/SSE for streaming responses.
 */

import { Command } from "commander";
import { runAgent } from "./commands/agent.js";
import { runChat } from "./commands/chat.js";

const program = new Command();

program
  .name("delia-cli")
  .description("Rich CLI for Delia - local LLM orchestration")
  .version("0.1.0");

// Agent command
program
  .command("agent")
  .description("Run an autonomous agent to complete a task")
  .argument("<task>", "Task for the agent to complete")
  .option("-m, --model <model>", "Model tier (quick/coder/moe) or specific model")
  .option("-w, --workspace <path>", "Confine file operations to directory")
  .option("--max-iterations <n>", "Maximum tool call iterations", "10")
  .option("--tools <tools>", "Comma-separated tools to enable")
  .option("-b, --backend <type>", "Force backend type (local/remote)")
  .option("--api-url <url>", "Delia API URL", "http://localhost:8201")
  // Permission flags (dangerous - disabled by default)
  .option("--allow-write", "Enable file write operations (DANGEROUS)")
  .option("--allow-exec", "Enable shell command execution (DANGEROUS)")
  .option("--yolo", "Skip all confirmation prompts (USE WITH CAUTION)")
  .action(async (task: string, options) => {
    // Show warning for dangerous flags
    if (options.allowWrite || options.allowExec) {
      console.log("\x1b[33m⚠️  DANGEROUS: Permission flags enabled:\x1b[0m");
      if (options.allowWrite) console.log("   --allow-write: Agent can write/modify files");
      if (options.allowExec) console.log("   --allow-exec: Agent can execute shell commands");
      if (options.yolo) console.log("   --yolo: Skipping confirmation prompts!");
      console.log("");
    }

    await runAgent(task, {
      model: options.model,
      workspace: options.workspace,
      maxIterations: parseInt(options.maxIterations, 10),
      tools: options.tools,
      backend: options.backend,
      apiUrl: options.apiUrl,
      // Permission flags
      allowWrite: options.allowWrite ?? false,
      allowExec: options.allowExec ?? false,
      yolo: options.yolo ?? false,
    });
  });

// Chat command - interactive REPL
program
  .command("chat")
  .description("Start an interactive chat session")
  .option("-m, --model <model>", "Model tier (quick/coder/moe) or specific model")
  .option("-b, --backend <type>", "Force backend type (local/remote)")
  .option("-s, --session <id>", "Resume existing session by ID")
  .option("--api-url <url>", "Delia API URL", "http://localhost:8201")
  .action(async (options) => {
    await runChat({
      model: options.model,
      backend: options.backend,
      session: options.session,
      api: options.apiUrl,
    });
  });

// Health check command
program
  .command("health")
  .description("Check backend health")
  .option("--api-url <url>", "Delia API URL", "http://localhost:8201")
  .action(async (options) => {
    const { DeliaClient } = await import("./lib/api.js");
    const client = new DeliaClient(options.apiUrl);
    try {
      const health = await client.health();
      console.log("Status:", health.status);
      console.log("Backends:");
      for (const backend of health.backends) {
        console.log(`  - ${backend.name} (${backend.provider}, ${backend.type})`);
      }
    } catch (err) {
      console.error("Health check failed:", (err as Error).message);
      process.exit(1);
    }
  });

program.parse();
