import librosa
import numpy as np
from tqdm.notebook import tqdm


def extract_signal_features(
    signal, sr, n_fft=1024, hop_length=512, n_mels=64, frames=5
):
    # https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html
    mel_spectrogram = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    # https://librosa.org/doc/main/generated/librosa.power_to_db.html
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    features_vector_size = log_mel_spectrogram.shape[1] - frames + 1
    dims = frames * n_mels

    if features_vector_size < 1:
        return np.empty((0, dims), np.float32)

    features = np.zeros((features_vector_size, dims), np.float32)
    for time in range(frames):
        features[:, n_mels * time : n_mels * (time + 1)] = log_mel_spectrogram[
            :, time : time + features_vector_size
        ].T

    return features


def generate_dataset(files_list, n_fft=1024, hop_length=512, n_mels=64, frames=5):
    dims = n_mels * frames
    # https://github.com/tqdm/tqdm - progress bar
    # Iterate over files list
    for index in tqdm(range(len(files_list))):
        #Load an audio file as a floating point time series
        signal, sr = load_sound_file(files_list[index])
        # First create mel_spectogram
        # https://importchris.medium.com/how-to-create-understand-mel-spectrograms-ff7634991056
        features = extract_signal_features(
            signal,
            sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            frames=frames,
        )

        if index == 0:
            dataset = np.zeros((features.shape[0] * len(files_list), dims), np.float32)

        dataset[
            features.shape[0] * index : features.shape[0] * (index + 1), :
        ] = features

    return dataset


def load_sound_file(path, mono=False, channel=0):
    #Load an audio file as a floating point time series
    signal, sr = librosa.load(path, sr=None, mono=mono)
    # check if signal is mono or stereo
    if signal.ndim < 2:
        sound_file = signal, sr
    else:
        sound_file = signal[channel, :], sr

    return sound_file