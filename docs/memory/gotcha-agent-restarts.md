---
name: gotcha-agent-restarts
description: 'A session restart kills background agents and wipes scratch; workers must save after every item'
type: gotcha
updated: 2026-10-17
---
A session restart stops background agents and clears the session scratch directory; anything not yet on disk is lost. Tool scratch lives under .agent-runs/ (ignored by git), durable outputs under data/curated/, results/ and docs/.

**Why:** a restart on 2026-10-17 lost the unsaved research of three workers and an in-flight monitor.
**How to apply:** brief workers to write results atomically after every item; after a restart notice, inspect files and agent state before re-dispatching.
