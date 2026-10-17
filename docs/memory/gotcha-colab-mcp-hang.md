---
name: gotcha-colab-mcp-hang
description: 'A dropped Colab tab hangs MCP calls for 30 min; poll in short calls and reconnect with authuser=1'
type: gotcha
updated: 2026-10-17
---
When the Colab browser tab loses its websocket, an in-flight run_code call gets no response and the MCP aborts it only after 1,828 s (the idle timeout). colab_connection_status then reports connected: false, but the VM, /content files and nohup jobs keep running. Reconnecting with open_colab_browser_connection (notebook_url ending ?authuser=1) needs the user to accept in the tab; no state was lost.

**Why:** one 30-minute stall during the 9B/greedy baseline runs; the jobs had finished while the call hung.
**How to apply:** start long jobs with nohup in the background and poll with run_code calls of at most 60 s; never wait on a single long call; on a drop, reconnect first, then read the job's files. Check df -h /content before big model downloads (the 27B FP8 checkpoint is 31 GB).

Related: [[env-colab-a100-runtime]]
