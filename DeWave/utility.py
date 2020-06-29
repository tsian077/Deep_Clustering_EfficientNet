import numpy as np
import librosa
from .infer import blind_source_separation


class Util:
    def __init__(self, source1, source2, model=None):
        self.source1 = source1
        self.source2 = source2
        self.model = model

    ## source1 is the primary speaker
    def audiomixer(self, frac=0.7):
        data1, _ = librosa.load(self.source1, sr=8000)
        data2, _ = librosa.load(self.source2, sr=8000)
        m = min(len(data1), len(data2))
        mix_data = data1[:m] + data2[:m] * frac
        mix_data = mix_data / max(abs(mix_data))
        source1 = self.source1.strip().split("/")
        source2 = self.source2.strip().split("/")
        speaker1, speaker2 = source1[-2], source2[-2]
        source1_name, source2_name = source1[-1][:-4], source2[-1][:-4]
        mix_name = "_".join([speaker1, source1_name, speaker2, source2_name, "mix.wav"])
        librosa.output.write_wav(mix_name, mix_data, 8000)
        return data1[:m], data2[:m], mix_name

    def test(self, frac=0.7):
        ref1, ref2, mix_name = self.audiomixer(frac=frac)
        sources = blind_source_separation(mix_name, self.model)
        estimate1 = sources[0][0]
        estimate2 = sources[1][0]
        ref1 = ref1 / np.linalg.norm(ref1, 2)
        ref2 = ref2 / np.linalg.norm(ref2, 2)
        estimate1 = estimate1 / np.linalg.norm(estimate1, 2)
        estimate2 = estimate2 / np.linalg.norm(estimate2, 2)
       
        return np.max([abs(np.correlate(ref1, estimate1)), 
                       abs(np.correlate(ref2, estimate2)),
                       abs(np.correlate(ref1, estimate2)),
                       abs(np.correlate(ref2, estimate1))])
