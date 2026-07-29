"""Reading a trained encoder directly, with no probe and no fitted stand-in.

The measurement behind figure 10. A controlled stimulus is rendered fresh from
the simulator, encoded by a run checkpoint, and projected onto the plane of the
dataset's own dominant directions. Two sweeps share one trajectory and one
parked layout, so they differ in nothing but which tee is the moving one, and
any difference in the encoding paths belongs to the encoder.

graphs/encoder_readout.py draws it.
"""

from __future__ import annotations

import numpy as np
import torch
import stable_worldmodel as swm

from utils import get_img_preprocessor

# the PC plane comes from real RandGoal frames encoded by the same checkpoint,
# so the axes are the encoder's own dominant directions
RANDGOAL_DATASET = "pusht_scripted_goal_train.lance"
PCA_REF_SAMPLES, PCA_REF_SEED = 1024, 0

PATH_POINTS, HARMONICS, SEED = 800, 6, 0
FILL = 0.95  # fraction of the box each coordinate's wander spans

# A tee spans 120x120 about its origin (see env.add_tee), so at any angle it fits
# inside a disc of this radius. The env draws the goal outline first and the
# block and agent over it, so the green shows whole only while the moving tee
# clears the parked one by that much. test_encoder_readout checks the pixels.
TEE_REACH = float(np.hypot(15.0, 120.0))
PARKED_TEE = np.array([70.0, 12.0, 0.0])  # angle 0, footprint [10,130]x[12,132]
PARKED_AGENT = np.array([70.0, 200.0])
WANDER_LO = np.array([257.0, 130.0])  # clears that footprint by TEE_REACH in x
WANDER_HI = np.array([381.0, 381.0])  # and stays a whole tee inside the walls
PARKED_STATE = np.array([*PARKED_AGENT, *PARKED_TEE[:2], PARKED_TEE[2], 0.0, 0.0])


def load_checkpoint(run_name: str, step: int | None = None, epoch: int | None = None):
    """A run checkpoint by exact step or epoch, else the newest epoch on disk."""
    ckpt_dir = swm.data.get_cache_dir(sub_folder="checkpoints") / run_name
    if step is not None:
        path = ckpt_dir / f"weights_step_{step}.pt"
    elif epoch is not None:
        path = ckpt_dir / f"weights_epoch_{epoch}.pt"
    else:
        found = sorted(ckpt_dir.glob("weights_epoch_*.pt"),
                       key=lambda q: int(q.stem.rsplit("_", 1)[-1]))
        if not found:
            raise FileNotFoundError(f"no weights_epoch_*.pt under {ckpt_dir}")
        path = found[-1]
    if not path.exists():
        raise FileNotFoundError(path)
    model = swm.wm.utils.load_pretrained(f"{run_name}/{path.name}")
    model.eval()
    model.requires_grad_(False)
    print(f"loaded {run_name}/{path.name}")
    return model


def make_encoder(model, img_size: int = 224):
    """encode(frames) -> (N, D) projection-space embeddings of rendered frames."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    preprocess = get_img_preprocessor("pixels", "pixels", img_size=img_size)

    @torch.no_grad()
    def encode(frames: list[np.ndarray]) -> np.ndarray:
        out = []
        for i in range(0, len(frames), 64):
            px = preprocess({"pixels": np.stack(frames[i : i + 64])})["pixels"].to(device)
            cls = model.encoder(px, interpolate_pos_encoding=True).last_hidden_state[:, 0]
            out.append(model.projector(cls.float()).float().cpu().numpy())
        return np.concatenate(out)

    return encode


def _wave(rng: np.random.Generator, points: int) -> np.ndarray:
    """A smooth curve on [0, 1] spanning exactly [-1, 1]: a phase-randomized sum
    of sinusoids rescaled by its own peak, so the wander roams the whole box and
    any tangle in the encoding path comes from the encoder, not from the path."""
    s = np.linspace(0.0, 1.0, points)
    amps, phases = rng.uniform(0.5, 1.0, HARMONICS), rng.uniform(0.0, 2 * np.pi, HARMONICS)
    wave = sum(a * np.sin(2 * np.pi * (k + 1) * s + p)
               for k, (a, p) in enumerate(zip(amps, phases)))
    return wave / np.abs(wave).max()


def wander_pose(seed: int, points: int) -> np.ndarray:
    """A smooth [x, y, α] wander of the shared box. Both panels drive their
    moving tee from this one trajectory, so nothing but the identity of the
    moving tee differs between them."""
    rng = np.random.default_rng(seed)
    mid, half = (WANDER_LO + WANDER_HI) / 2, (WANDER_HI - WANDER_LO) / 2
    return np.column_stack([mid[0] + FILL * half[0] * _wave(rng, points),
                            mid[1] + FILL * half[1] * _wave(rng, points),
                            FILL * np.pi * _wave(rng, points)])


def block_states(pose: np.ndarray) -> np.ndarray:
    """Env states [agent_xy, block_xy, angle, vel(=0)] carrying that wander on
    the block, with the agent parked."""
    return np.column_stack([np.tile(PARKED_AGENT, (len(pose), 1)), pose,
                            np.zeros((len(pose), 2))])


def fit_pca_basis(encode, dataset: str, n: int = PCA_REF_SAMPLES, seed: int = PCA_REF_SEED):
    """Mean and top-2 principal directions of real dataset frames under this
    checkpoint — the shared plane both paths are drawn in."""
    ds = swm.data.load_dataset(dataset, transform=None, num_steps=4, frameskip=5,
                               keys_to_load=["pixels"])
    rng = np.random.default_rng(seed)
    frames = [np.ascontiguousarray(np.asarray(ds[int(i)]["pixels"])[0].transpose(1, 2, 0))
              for i in rng.integers(0, len(ds), n)]
    emb = encode(frames)
    mean = emb.mean(axis=0)
    print(f"fit dataset PCA basis on {n} {dataset} frames")
    return mean, np.linalg.svd(emb - mean, full_matrices=False)[2][:2].T


def encode_path(encode, frames: list[np.ndarray], mean, comps, idx: np.ndarray) -> dict:
    """One path on the shared PC plane, centred on its own mean so the centroid
    sits mid-grid. Inset frames are kept at the GIF indices to avoid re-rendering."""
    scores = (encode(frames) - mean) @ comps
    return {"scores": scores - scores.mean(axis=0), "insets": [frames[k] for k in idx]}


def readout_paths(encode, render, poses: np.ndarray, mean, comps, idx: np.ndarray) -> list[dict]:
    """The two panels, in order: the goal-T carrying the wander, then the
    block-T carrying the identical wander."""
    return [
        encode_path(encode, [render(PARKED_STATE, g) for g in poses], mean, comps, idx),
        encode_path(encode, [render(s, PARKED_TEE) for s in block_states(poses)],
                    mean, comps, idx),
    ]
