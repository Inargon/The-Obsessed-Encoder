# Counterfactual action binding diagnostic

Question: does the action-conditioned predictor bind each candidate action
sequence to its own physical future, or is action merely decodable from a
useful representation?

Each bank row fixes one PushT anchor and evaluates 64 candidate three-step
action sequences. Eight physically diverse outcomes are retained, including a
pusher-matched but block-divergent hard pair. The 10k, seed-0 comparison uses
the same bank and the same `masked_sequence_pred03` backbone in all arms:

1. `bank_sequence_idm`: additional full-sequence IDM on the bank only.
2. `counterfactual_binding`: IDM plus forward effect regression, same-anchor
   branch retrieval, hard-pair retrieval, and invariance across two independent
   episode-constant watermark colours.
3. `binding_geometry_oracle`: arm 2 plus privileged real- and predicted-effect
   physical Gram matching. This is an upper-bound diagnostic, not the proposed
   non-privileged method.

The decisive comparisons are success rate, branch/hard-branch binding
accuracy, and the usual same-content versus same-tag pair metrics. Arm 2 must
beat arm 1 to show value beyond more IDM data; arm 3 diagnoses whether any
remaining gap is specifically geometric.
