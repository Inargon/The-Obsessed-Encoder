import numpy as np


def resolve_windows(dataset, clips, episodes, steps):
    """Return raw row offsets and real episode labels for loader clip indices."""
    if dataset.frameskip != 1:
        raise ValueError('Audit requires frameskip=1')
    episodes=np.asarray(episodes).reshape(-1)
    steps=np.asarray(steps).reshape(-1)
    if len(episodes)!=len(steps): raise ValueError('Metadata length mismatch')
    rows,ids=[],[]
    for clip in clips:
        clip=int(clip)
        if not 0<=clip<len(dataset): raise ValueError('Invalid clip ID')
        ep,start=map(int,dataset.clip_indices[clip]);span=int(dataset.span)
        if not 0<=ep<len(dataset.lengths) or start<0 or start+span>int(dataset.lengths[ep]):
            raise ValueError('Clip exceeds episode boundary')
        row=int(dataset.offsets[ep])+start
        if row<0 or row+span>len(steps): raise ValueError('Invalid raw window')
        if not np.all(episodes[row:row+span]==episodes[row]): raise ValueError('Crossed episode IDs')
        np.testing.assert_array_equal(steps[row:row+span],steps[row]+np.arange(span))
        rows.append(row);ids.append(episodes[row])
    return np.asarray(rows,dtype=np.int64),np.asarray(ids)
