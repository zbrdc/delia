#!/usr/bin/env node
import React from 'react';
import { render } from 'ink';
import meow from 'meow';
import { App } from './App.js';

const cli = meow(`
  Usage
    $ delia-tui [options]

  Options
    --server, -s        Delia API server URL (default: http://localhost:34589)
    --no-allow-write    Disable file write operations
    --no-allow-exec     Disable shell command execution
    --yolo              Skip all security prompts (dangerous!)
    --help              Show this help

  By default, file and shell operations are enabled but require confirmation.
  Use --yolo to skip confirmations (not recommended).

  Before running, start the Delia API server:
    $ delia api

  Examples
    $ delia-tui
    $ delia-tui --yolo
    $ delia-tui --no-allow-exec
`, {
  importMeta: import.meta,
  flags: {
    server: {
      type: 'string',
      shortFlag: 's',
      default: 'http://localhost:34589',
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
    allowWrite={cli.flags.allowWrite}
    allowExec={cli.flags.allowExec}
    yolo={cli.flags.yolo}
  />
);
