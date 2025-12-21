#!/usr/bin/env node
import React from 'react';
import { render } from 'ink';
import meow from 'meow';
import { App } from './App.js';

const cli = meow(`
  Usage
    $ delia-tui [options]

  Options
    --server, -s        API server URL (default: http://localhost:34589)
    --task, -t          Initial task to run
    --no-allow-write    Disable file write operations
    --no-allow-exec     Disable shell execution
    --yolo              Skip security prompts (dangerous!)

  Examples
    $ delia-tui
    $ delia-tui --task "Find TODO comments"
    $ delia-tui --yolo
`, {
  importMeta: import.meta,
  flags: {
    server: {
      type: 'string',
      shortFlag: 's',
      default: 'http://localhost:34589',
    },
    task: {
      type: 'string',
      shortFlag: 't',
    },
    allowWrite: {
      type: 'boolean',
      default: true,
    },
    allowExec: {
      type: 'boolean',
      default: true,
    },
    yolo: {
      type: 'boolean',
      default: false,
    },
  },
});

render(
  <App
    serverUrl={cli.flags.server}
    initialTask={cli.flags.task}
    allowWrite={cli.flags.allowWrite}
    allowExec={cli.flags.allowExec}
    yolo={cli.flags.yolo}
  />
);
