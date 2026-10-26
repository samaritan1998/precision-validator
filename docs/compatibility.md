# Runtime compatibility and validation cost

The base package needs Linux, Python 3.10 or newer, and NumPy. Torch and training
frameworks are optional dependencies supplied by the training environment.
Dependency lower bounds describe installation constraints, not a claim that
every allowed runtime combination passed the GPU suite.

## Executed environments

The rc3 installed-wheel checks used these environments:

| Python | NumPy | Training runtime | Scope |
|---|---|---|---|
| 3.10.20 | 2.2.6 | None | Core contracts and packaging; optional tests explicitly skipped |
| 3.11.15 | 2.4.6 | None | Core contracts and packaging; optional tests explicitly skipped |
| 3.12.3 | 1.26.4 | PyTorch 2.11.0+cu129 | Full CPU tests with no skips; B200/H800/A800 integration |
| 3.13.12 | 2.4.6 | No PyTorch | Core contracts and packaging; optional tests explicitly skipped |

The tested training environment also contains CUDA 12.9, Transformer Engine
2.10.0, Triton 3.6.0, FLA 0.4.1, Megatron Core 0.16.0rc0, Transformers 5.6.0
and Safetensors 0.7.0. Exact package/runtime hashes are stored in experiment
receipts. These are recorded test versions, not dependencies that the base
package installs automatically.

The full Ray driver fixture used the public Slime source commit
`18868a79b0b083a093798a94ed0f8cb4f71b9c14`, with the bridge installed into a new
copy. It exercised TP1/DP2 versus TP2/ETP2/DP1, three steps per stage, through
`train.py`. The wider rank-loop matrix has a separate scope and source
inventory. Production Qwen validation uses a separately supplied model-enabled
Slime checkout; that checkout, pretrained weights and private input are not
distributed with this library.

The bridge recognizes specific training-entry structures and rejects unknown
ones. Passing that source check does not qualify a new model, backend or
framework revision. Run the generated integration fixtures and an appropriate
production comparison after changing those dependencies.

## Full-vector validation includes storage work

For P trainable parameters and N captured steps, FP32 gradients and FP32 updates
require approximately `24 * P * N` bytes across reference, repeat and candidate.
This excludes initialization snapshots, optimizer-state trajectories, receipts
and checkpoint-resume artifacts.

The current 26,895,998,464-parameter Qwen case writes 645,503,963,136 bytes
(about 601 GiB) of gradients and updates for one step across three stages.
Its temporary canonical initialization snapshot is another roughly 100 GiB per
active stage. Running the stages concurrently needs enough storage and I/O
capacity for those simultaneous snapshots.

The adapter hashes initialized weights, captures complete tensors, and checks
actual optimizer counters. The runner then validates saved bytes and rank
coverage before returning success. Comparison verifies the raw data again and
aligns every parameter in bounded memory. A finished GPU backward pass is
therefore earlier than completed validation. On a shared filesystem, file
hashing and many small TP fragments can take substantial time after GPU work
ends. Use the final status and successful launcher exit, not a final layer
heartbeat, as completion evidence.

Receipt timing fields distinguish native training plus capture from observed
capture time. They exclude some setup, initialization hashing and final
validation work. They are not a measurement of ordinary-training throughput
or total validator overhead; that needs a matched run without capture.

Keep raw vectors until the report and any required independent audit are
reviewed. Do not replace full gradients with sampled gradients to reduce cost
while claiming the same acceptance scope. The small independent report oracle
has a deliberate memory limit and should not be used on a production model.

Current numerical outcomes and the known BF16 MoE findings are in
[validation.md](validation.md). A supported adapter can correctly report a
model numerical failure; interface support is not a promise of numerical parity.
