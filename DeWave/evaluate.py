import DeWave
import glob
import numpy as np
import argparse
import os

def evaluation(data_dir, model, frac):
    speakers = [os.path.join(data_dir, i) for i in os.listdir(data_dir) \
        if os.path.isdir(os.path.join(data_dir, i))]
    ## decompose audios with the suffix "mix" at the end
    n_speakers = len(speakers)
    cross_cor = []
    if n_speakers <= 1:
        return None
    for speaker in speakers:
        sources = glob.glob(os.path.join(speaker, "*.wav"))
        for source1 in sources:
            speaker_id = np.random.randint(n_speakers)
            while speaker == speakers[speaker_id]:
                speaker_id = np.random.randint(n_speakers)
            sources2 = glob.glob(os.path.join(speakers[speaker_id], "*.wav"))
            source2 = sources2[np.random.randint(len(sources2))]
            Test = DeWave.utility.Util(source1, source2, model)
            result = Test.test(frac=frac)
            cross_cor.append(result)
    return np.mean(cross_cor)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("evaluation")
    parser.add_argument("--dir", type=str, help="input folder name")
    parser.add_argument("--model_dir", type=str, help="trained model")
    parser.add_argument("--frac", type=float, help="noise level", \
                        nargs="?", default=0.7)
    args = parser.parse_args()
    data_dir=args.dir
    model = args.model_dir
    frac = args.frac
    print(evaluation(data_dir, model, frac))
