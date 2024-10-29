# audio_processor.py

import os
import librosa
import numpy as np
import soundfile as sf
import tempfile
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.signal import butter, filtfilt
from scipy import signal
from scipy.signal import TransferFunction
import math

class AudioProcessor:
    def __init__(self, input_file):
        """
        Initialize the AudioProcessor with the input file.

        Parameters:
        - input_file (str): Path to the input audio file (MP3, AAC, FLAC, etc.)
        """
        self.input_file = input_file
        self.samples, self.sample_rate = librosa.load(self.input_file, sr=None, mono=False)
        if self.samples.ndim == 1:
            self.samples = np.expand_dims(self.samples, axis=0)
        self.num_channels = self.samples.shape[0]
        self.output_file = None  # 用于保存输出文件路径

    def get_audio_file_path(self):
        """
        Return the path to the audio file.

        Returns:
        - str: The file path of the audio file.
        """
        return self.input_file

    def convert_to_wav(self, output_file=None):
        """
        Convert the input audio file to WAV format using soundfile.

        Parameters:
        - output_file (str): Path to save the converted WAV file. If None, the output file will have the same
                             name as the input file with a .wav extension.

        Returns:
        - str: A message indicating the success of the operation.
        """
        if output_file is None:
            temp_dir = tempfile.gettempdir()
            output_file = os.path.join(temp_dir, os.path.splitext(os.path.basename(self.input_file))[0] + '.wav')

        sf.write(output_file, self.samples.T, self.sample_rate)
        self.output_file = output_file  # 保存输出文件路径

        message = f"Audio converted to WAV format."
        return message

    def print_stats(self, signal_path=None):
        if signal_path is None:
            signal_path = self.input_file

        samples, sample_rate = librosa.load(signal_path, mono=False, sr=None)
        channels = samples.shape[0] if samples.ndim > 1 else 1
        shape = samples.shape if samples.ndim > 1 else (1, samples.shape[0])
        stats = f"Channels: {channels}\nSample Rate: {sample_rate}\nShape: {shape}\nDtype: {samples.dtype}\n"
        stats += f"\nAudio Data Summary:\nMin: {np.min(samples)}\nMax: {np.max(samples)}\nMean: {np.mean(samples)}\nStd: {np.std(samples)}\n"
        return stats

    def trim_audio(self, start_time, end_time, output_dir=None):
        if output_dir is None:
            output_dir = tempfile.gettempdir()

        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        total_samples = self.samples.shape[1]

        if start_sample < 0 or end_sample > total_samples or start_sample >= end_sample:
            raise ValueError(f"Invalid start_time ({start_time} s) or end_time ({end_time} s) for audio of length {total_samples / self.sample_rate} seconds.")

        trimmed_samples = self.samples[:, start_sample:end_sample]
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(self.input_file)
        file_name, _ = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{file_name}_trimmed.wav")
        sf.write(output_path, trimmed_samples.T, self.sample_rate)
        message = f"Audio trimmed from {start_time} seconds to {end_time} seconds."
        return message, output_path

    def rechannel_audio(self, target_channels, output_dir=None):
        if output_dir is None:
            output_dir = tempfile.gettempdir()

        if self.num_channels == target_channels:
            rechanneled_audio = self.samples
        elif self.num_channels < target_channels:
            repeat_times = target_channels // self.num_channels
            remaining_channels = target_channels % self.num_channels
            rechanneled_audio = np.tile(self.samples, (repeat_times, 1))
            if remaining_channels > 0:
                rechanneled_audio = np.vstack([rechanneled_audio, self.samples[:remaining_channels, :]])
        else:
            rechanneled_audio = self.samples[:target_channels, :]

        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(self.input_file)
        file_name, _ = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{file_name}_rechanneled.wav")
        sf.write(output_path, rechanneled_audio.T, self.sample_rate)
        message = f"Audio rechanneled to {target_channels} channels."
        return message, output_path

    def resample_audio(self, target_sr, output_dir=None):
        """
        Resample the audio to a new sample rate and save it as a WAV file.

        Parameters:
        - target_sr (int): The target sample rate in Hz.
        - output_dir (str): Directory to save the resampled audio file. If None, uses the system's temporary directory.

        Returns:
        - message (str): A message indicating the success of the operation.
        - output_path (str): Path to the saved resampled audio file.
        """
        if output_dir is None:
            output_dir = tempfile.gettempdir()

        if target_sr <= 0:
            raise ValueError("Target sample rate must be positive.")

        min_acceptable_sr = self.sample_rate / 2
        if target_sr < min_acceptable_sr:
            raise ValueError(
                f"Target sample rate is too low and may cause distortion.\n"
                f"Please choose a sample rate between {int(min_acceptable_sr)} Hz and {self.sample_rate} Hz."
            )

        # Proceed with resampling
        resampled_audio = librosa.resample(self.samples, orig_sr=self.sample_rate, target_sr=target_sr, axis=1)

        # Update the sample rate
        self.sample_rate = target_sr

        # Ensure the resampled_audio is in the correct shape (channels x samples)
        if resampled_audio.ndim == 1:
            resampled_audio = np.expand_dims(resampled_audio, axis=0)

        # Save the resampled audio to a WAV file
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(self.input_file)
        file_name, _ = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{file_name}_resampled.wav")
        sf.write(output_path, resampled_audio.T, target_sr)
        message = f"Audio resampled to {target_sr} Hz."
        return message, output_path

    def play_audio(self):
        return self.samples, self.sample_rate

    def get_waveform_figure(self, xlim=None, ylim=None):
        num_channels = self.num_channels
        fig = make_subplots(rows=num_channels, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        time_axis = np.arange(self.samples.shape[1]) / self.sample_rate
        for c in range(num_channels):
            fig.add_trace(
                go.Scatter(x=time_axis, y=self.samples[c], mode='lines', name=f'Channel {c+1}'),
                row=c+1, col=1
            )
            if ylim:
                fig.update_yaxes(range=ylim, row=c+1, col=1)
        if xlim:
            fig.update_xaxes(range=xlim)
        fig.update_layout(
            dragmode='zoom',  # 添加这一行，启用框选放大功能
            height=300*num_channels,
            width=800,
            title_text="Waveform",
            showlegend=False
        )
        fig.update_xaxes(title_text='Time (s)', row=num_channels, col=1)
        fig.update_yaxes(title_text='Amplitude')
        return fig

    def RMS_energy(self, signal, frame_size, hop_length):
        res = []
        for i in range(0, len(signal), hop_length):
            cur_portion = signal[i:i + frame_size]
            rmse_val = np.sqrt(np.mean(cur_portion ** 2))
            res.append(rmse_val)
        return np.array(res)

    def get_multichannel_RMS_energy_figure(self, frame_size, hop_length):
        num_channels = self.num_channels
        fig = make_subplots(rows=num_channels, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        for c in range(num_channels):
            rmse = self.RMS_energy(self.samples[c], frame_size, hop_length)
            frames = range(len(rmse))
            time = librosa.frames_to_time(frames, sr=self.sample_rate, hop_length=hop_length)
            fig.add_trace(
                go.Scatter(x=time, y=rmse, mode='lines', name=f'Channel {c+1}'),
                row=c+1, col=1
            )
        fig.update_layout(
            dragmode='zoom',  # 添加这一行，启用框选放大功能
            height=300*num_channels,
            width=800,
            title_text="RMS Energy",
            showlegend=False
        )
        fig.update_xaxes(title_text='Time (s)', row=num_channels, col=1)
        fig.update_yaxes(title_text='RMS Energy')
        return fig

    def crest_factor(self, signal, frame_size, hop_length):
        res = []
        for i in range(0, len(signal), hop_length):
            cur_portion = signal[i:i+frame_size]
            if len(cur_portion) == 0:
                break
            rmse_val = np.sqrt(np.mean(cur_portion**2))
            crest_val = np.max(np.abs(cur_portion)) / rmse_val if rmse_val != 0 else 0
            res.append(crest_val)
        return np.array(res)

    def get_multichannel_crest_factor_figure(self, frame_size, hop_length):
        num_channels = self.num_channels
        fig = make_subplots(rows=num_channels, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        for c in range(num_channels):
            crest = self.crest_factor(self.samples[c], frame_size, hop_length)
            frames = range(len(crest))
            time = librosa.frames_to_time(frames, sr=self.sample_rate, hop_length=hop_length)
            fig.add_trace(
                go.Scatter(x=time, y=crest, mode='lines', name=f'Channel {c+1}'),
                row=c+1, col=1
            )
        fig.update_layout(
            dragmode='zoom',  # 添加这一行，启用框选放大功能
            height=300*num_channels,
            width=800,
            title_text="Crest Factor",
            showlegend=False
        )
        fig.update_xaxes(title_text='Time (s)', row=num_channels, col=1)
        fig.update_yaxes(title_text='Crest Factor')
        return fig

    def loudness_dbA(self, frame_size, hop_length, cutoff=2000, return_average=False):
        # High-pass filter to remove frequencies below the cutoff
        nyquist = 0.5 * self.sample_rate
        norm_cutoff = cutoff / nyquist
        b, a = butter(4, norm_cutoff, btype='high', analog=False)
        filtered_samples = filtfilt(b, a, self.samples, axis=1)

        # A-weighting filter coefficients (analog)
        f1 = 20.598997
        f2 = 107.65265
        f3 = 737.86223
        f4 = 12194.217

        nums = [(2 * math.pi * f4) ** 2 * (10 ** (2 / 20)), 0, 0, 0, 0]
        dens = np.polymul(
            [1, 4 * math.pi * f4, (2 * math.pi * f4) ** 2],
            [1, 4 * math.pi * f1, (2 * math.pi * f1) ** 2]
        )
        dens = np.polymul(np.polymul(dens, [1, 2 * math.pi * f3]),
                         [1, 2 * math.pi * f2])

        # Create the analog A-weighting transfer function
        a_weighting_tf = TransferFunction(nums, dens)

        # Generate frequency response of the A-weighting filter
        freqs = np.linspace(0, nyquist, frame_size // 2 + 1)
        w, mag = signal.freqs(a_weighting_tf.num, a_weighting_tf.den, worN=2 * np.pi * freqs)
        a_weighting = 10 ** (mag / 20)  # Convert from dB to linear scale

        average_loudness_values = []
        num_channels = self.num_channels
        fig = make_subplots(rows=num_channels, cols=1, shared_xaxes=True, vertical_spacing=0.02)

        for c in range(num_channels):
            res = []
            max_rms = None
            for i in range(0, self.samples.shape[1] - frame_size + 1, hop_length):
                cur_portion = filtered_samples[c][i:i + frame_size]
                spectrum = np.fft.rfft(cur_portion * np.hanning(frame_size))
                freq_spectrum = np.fft.rfftfreq(frame_size, d=1/self.sample_rate)
                a_weighting_interp = np.interp(freq_spectrum, w / (2 * np.pi), a_weighting)
                weighted_spectrum = spectrum * a_weighting_interp
                rms = np.sqrt(np.mean(np.abs(weighted_spectrum) ** 2))
                if max_rms is None or rms > max_rms:
                    max_rms = rms
                res.append(rms)

            res_dbA = 20 * np.log10(np.array(res) / max_rms)
            average_loudness = np.mean(res_dbA)
            average_loudness_values.append(average_loudness)

            frames = range(len(res_dbA))
            time_axis = librosa.frames_to_time(frames, sr=self.sample_rate, hop_length=hop_length)

            fig.add_trace(
                go.Scatter(x=time_axis, y=res_dbA, mode='lines', name=f'Channel {c+1}'),
                row=c+1, col=1
            )

        fig.update_layout(
            dragmode='zoom',  # 添加这一行，启用框选放大功能
            height=300*num_channels,
            width=800,
            title_text=f'Loudness in dB(A) (>{cutoff}Hz)',
            showlegend=False
        )
        fig.update_xaxes(title_text='Time (s)', row=num_channels, col=1)
        fig.update_yaxes(title_text='Loudness (dB(A))')

        if return_average:
            return average_loudness_values, fig
        else:
            return fig

    def check_duration(self, min_duration):
        duration = self.samples.shape[1] / self.sample_rate
        if duration >= min_duration:
            return True, duration
        else:
            return False, duration





        
           
            

    
    