---
name: gotcha-kaggle-session-billing
description: 'Kaggle API sessions bill from creation; cancel by the id in the proxy URL (/k/<id>/), not disconnect'
type: gotcha
updated: 2026-10-17
---
A Kaggle session started through the API bills GPU quota from creation, one second per second, even when the container is the CPU image (no driver libraries, CPU-only torch). The API never returns the session id; it is the number in the proxy URL path /k/<id>/. `disconnect_kaggle` cannot stop it without that id. Cancel with kagglesdk `cancel_kernel_session(kernel_session_id=<id>)` and watch `get_accelerator_quota_statistics` stop moving.
The notebook must be saved with enable_gpu true and a machine_shape before the session starts; a second `CreateKernelSession` for the same notebook returns 409 until the first is cancelled.

**Why:** the first probe cost 14 minutes of the 6-hour weekly quota before it was cancelled.
**How to apply:** after every session start, read the id from the server URL and keep it; check the quota counter at the next step; cancel before ending any turn.
