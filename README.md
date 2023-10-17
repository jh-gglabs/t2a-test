# text2avatar-inference
The Text2Avatar Inference Module consists of lip sync, face blendshape, and whole body gesture generative models.

### Environments
* python 3.9
* tensorflow 2.13.0
* librosa 0.10.1
* python_speech_features 0.6

### Lip-sync Usage
```python
import numpy as np
from lipsync_inference import LipSyncPredictor

single_audio_chunk = np.zeros((400, ))
lipsync_model = LipSyncPredictor()

outputs = lipsync_model.predict_outputs(single_audio_chunk)

"""
The output is the format as the below,
{
    "num_frames": int,
    "viseme": np.array [num_frames, ],
    "blendshapes: np.array [num_frames, 52]
}
"""
```