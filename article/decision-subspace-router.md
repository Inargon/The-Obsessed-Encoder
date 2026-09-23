# Decision-Subspace Router

This prototype replaces the current per-sample one-dimensional control guide
with a bounded, history-aware subspace of recent control gradients.

For flattened control gradients (g_{c,i}), it maintains a rank-(r) sketch
of their exponentially decayed covariance using a thin SVD.  If (U) spans
the resulting subspace, the supported prediction update is

\[
g_p^{\mathrm{sup}}=UU^\top g_p.
\]

Within that subspace, a PCGrad correction removes only the component that
conflicts with the current supported control gradient.  Directions unsupported
by recent action-conditioned evidence do not enter the shared encoder.  The
prediction head still receives its original prediction gradient because the
router is implemented as a zero-valued representation-gradient surrogate.

The initial matched experiment uses rank 16, decay 0.99, updates every 20
steps, and eight deterministic control-gradient candidates per update.  It
changes no loss, data, seed, epoch count, or model architecture relative to
Full Aligned.

