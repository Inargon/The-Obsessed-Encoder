# Technical article draft

The working article is [`obsessed-encoder.md`](obsessed-encoder.md).

The draft is intentionally narrower than the paper. It tells one complete
story using the matched epoch-10 tagged-PushT experiment:

1. a predictable tag captures a standard JEPA representation;
2. control-guided encoder optimization restores physical state;
3. component ablations separate task success from nuisance suppression;
4. high-dimensional near-orthogonality is treated as a null expectation, not
   as evidence that a gradient direction is useless.

Exact numbers and protocol notes are recorded in
[`results_snapshot.json`](results_snapshot.json). Before publishing, replace
the figure callouts with exported assets and add author names.

The exact control-objective definition, design motivation, and claim boundaries
are documented separately in [`lctrl-design.md`](lctrl-design.md).
