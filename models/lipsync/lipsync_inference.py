import os
import tensorflow as tf
import librosa
import python_speech_features as psf
import numpy as np
import json
from scipy import interpolate


class LipSyncPredictor(object):
    def __init__(self, model_path="./assets/lipsync/phoneme_clf_gglabs.h5", prefix=""):
        self.model = tf.keras.models.load_model(model_path)
        self.arkit_bs_mapper = json.load(open(os.path.join(prefix, 'assets/lipsync/arkit_bs_weights_mapper.json')))
        self.timit_index_mapper = json.load(open(os.path.join(prefix, 'assets/lipsync/timit_index_map.json')))
        self.viseme_char_mapper = json.load(open(os.path.join(prefix, 'assets/lipsync/viseme_char_map.json')))
        self.sample_rate = 16000
        self.num_blendshapes = 52
        self.single_chunk_size = 400

    def predict_outputs_from_audio_file(self, audio_file_path):
        x, sr = librosa.load(audio_file_path)
        if (sr != self.sample_rate):
            x = librosa.resample(x, orig_sr=sr, target_sr=self.sample_rate)
        
        mfcc_feat = psf.mfcc(
            x, samplerate=self.sample_rate, numcep=13
        )

        num_frames = mfcc_feat.shape[0]

        prediction = self.model.predict(np.expand_dims(mfcc_feat, axis=0))
        prediction = np.squeeze(prediction)

        viseme_result, bs_weights = self._make_viseme_and_bs_weights(
            prediction=prediction, num_frames=num_frames
        )

        subsampled_bs_weights = self._change_fps(
            bs_weights=bs_weights, from_fps=100, to_fps=60
        )

        num_frames = subsampled_bs_weights.shape[0]
        
        return {
            "num_frames": num_frames,
            "blendshapes": subsampled_bs_weights
        }

    def predict_outputs_from_audio_chunk(self, audio_chunk: np.array):
        if len(audio_chunk) < self.single_chunk_size:
            raise Warning("The audio chunk size is too short. Make sure the audio chunk size!")
        
        mfcc_feat = psf.mfcc(
            audio_chunk, 
            samplerate=self.sample_rate, numcep=13
        )

        num_frames = mfcc_feat.shape[0]
        
        prediction = self.model.predict(np.expand_dims(mfcc_feat, axis=0))
        prediction = np.squeeze(prediction)

        viseme_result, bs_weights = self._make_viseme_and_bs_weights(
            prediction=prediction, num_frames=num_frames
        )

        subsampled_bs_weights = self._change_fps(
            bs_weights=bs_weights, from_fps=100, to_fps=60
        )

        num_frames = subsampled_bs_weights.shape[0]
        
        return {
            "num_frames": num_frames,
            "blendshapes": subsampled_bs_weights
        }

    def _make_viseme_and_bs_weights(self, prediction, num_frames):
        # Mapping to viseme
        phoneme_result = [self.timit_index_mapper[str(np.argmax(ph))] for ph in prediction]
        viseme_result = [self.viseme_char_mapper[str(ph)] for ph in phoneme_result]
        
        # Mapping to morph-targets
        bs_weights = np.zeros((num_frames, self.num_blendshapes))
        for i in range(0, num_frames):
            viseme = str(viseme_result[i])
            bs_w = self.arkit_bs_mapper[viseme]
            bs_weights[i, ] = bs_w
        
        return viseme_result, bs_weights
    
    def _change_fps(self, bs_weights, from_fps=100, to_fps=60):
        num_frames = bs_weights.shape[0]
        frametime = from_fps / 10000
        original_times = np.linspace(0, num_frames -1, num_frames)
        sample_times = np.linspace(0, num_frames-1, int(1.0 * (num_frames * (to_fps * frametime))))

        subsampled_bs_weights = interpolate.griddata(
            original_times, bs_weights.reshape([num_frames, -1]), sample_times, method='linear'
        )

        return subsampled_bs_weights


if __name__ == "__main__":
    single_audio_chunk = np.zeros((400, ))
    lipsync_model = LipSyncPredictor("./assets/lipsync/phoneme_clf_gglabs.h5")

    outputs = lipsync_model.predict_outputs(single_audio_chunk)

    print(outputs)