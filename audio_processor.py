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
        # Load the audio file using librosa without changing the sample rate or converting to mono
        self.samples, self.sample_rate = librosa.load(self.input_file, sr=None, mono=False)
        # If the audio is mono, expand dimensions to match the expected shape
        if self.samples.ndim == 1:
            self.samples = np.expand_dims(self.samples, axis=0)
        self.num_channels = self.samples.shape[0]
        self.output_file = None  # To store the path of the output file

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

        # Write the audio data to a WAV file
        sf.write(output_file, self.samples.T, self.sample_rate)
        self.output_file = output_file  # Save the output file path

        message = f"Audio converted to WAV format."
        return message

    def print_stats(self, signal_path=None):
        """
        Print statistical information about the audio file.

        Parameters:
        - signal_path (str): Path to the audio file. If None, uses the input file.

        Returns:
        - str: A string containing the audio statistics.
        """
        if signal_path is None:
            signal_path = self.input_file

        # Load the audio file
        samples, sample_rate = librosa.load(signal_path, mono=False, sr=None)
        channels = samples.shape[0] if samples.ndim > 1 else 1
        shape = samples.shape if samples.ndim > 1 else (1, samples.shape[0])
        # Compile statistics about the audio data
        stats = f"Channels: {channels}\nSample Rate: {sample_rate}\nShape: {shape}\nDtype: {samples.dtype}\n"
        stats += f"\nAudio Data Summary:\nMin: {np.min(samples)}\nMax: {np.max(samples)}\nMean: {np.mean(samples)}\nStd: {np.std(samples)}\n"
        return stats

    def trim_audio(self, start_time, end_time, output_dir=None):
        """
        Trim the audio between specified start and end times and save it as a new WAV file.

        Parameters:
        - start_time (float): Start time in seconds.
        - end_time (float): End time in seconds.
        - output_dir (str): Directory to save the trimmed audio file. If None, uses the system's temporary directory.

        Returns:
        - message (str): A message indicating the success of the operation.
        - output_path (str): Path to the saved trimmed audio file.
        """
        if output_dir is None:
            output_dir = tempfile.gettempdir()

        # Convert start and end times to sample indices
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        total_samples = self.samples.shape[1]

        # Validate the sample indices
        if start_sample < 0 or end_sample > total_samples or start_sample >= end_sample:
            raise ValueError(f"Invalid start_time ({start_time} s) or end_time ({end_time} s) for audio of length {total_samples / self.sample_rate} seconds.")

        # Trim the samples array to get the desired segment
        trimmed_samples = self.samples[:, start_sample:end_sample]

        # NOTE: To retain complete audio information, you might need to replace the matrix windowing method here.
        # The current approach slices the numpy array, but if you require overlapping windows or a different
        # method to process the audio data, you should implement that logic accordingly.

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        # Construct the output file path
        base_name = os.path.basename(self.input_file)
        file_name, _ = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{file_name}_trimmed.wav")
        # Save the trimmed audio to a WAV file
        sf.write(output_path, trimmed_samples.T, self.sample_rate)
        message = f"Audio trimmed from {start_time} seconds to {end_time} seconds."
        return message, output_path

    def rechannel_audio(self, target_channels, output_dir=None):
        """
        Change the number of channels in the audio file and save it as a new WAV file.

        Parameters:
        - target_channels (int): The desired number of channels.
        - output_dir (str): Directory to save the rechanneled audio file. If None, uses the system's temporary directory.

        Returns:
        - message (str): A message indicating the success of the operation.
        - output_path (str): Path to the saved rechanneled audio file.
        """
        if output_dir is None:
            output_dir = tempfile.gettempdir()

        # Adjust the audio data to have the target number of channels
        if self.num_channels == target_channels:
            rechanneled_audio = self.samples
        elif self.num_channels < target_channels:
            # Repeat channels to reach the target number
            repeat_times = target_channels // self.num_channels
            remaining_channels = target_channels % self.num_channels
            rechanneled_audio = np.tile(self.samples, (repeat_times, 1))
            if remaining_channels > 0:
                rechanneled_audio = np.vstack([rechanneled_audio, self.samples[:remaining_channels, :]])
        else:
            # Truncate channels to reach the target number
            rechanneled_audio = self.samples[:target_channels, :]

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        # Construct the output file path
        base_name = os.path.basename(self.input_file)
        file_name, _ = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{file_name}_rechanneled.wav")
        # Save the rechanneled audio to a WAV file
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

        # Proceed with resampling the audio
        resampled_audio = librosa.resample(self.samples, orig_sr=self.sample_rate, target_sr=target_sr, axis=1)

        # Update the sample rate
        self.sample_rate = target_sr

        # Ensure the resampled_audio has the correct shape (channels x samples)
        if resampled_audio.ndim == 1:
            resampled_audio = np.expand_dims(resampled_audio, axis=0)

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        # Construct the output file path
        base_name = os.path.basename(self.input_file)
        file_name, _ = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{file_name}_resampled.wav")
        # Save the resampled audio to a WAV file
        sf.write(output_path, resampled_audio.T, target_sr)
        message = f"Audio resampled to {target_sr} Hz."
        return message, output_path

    def play_audio(self):
        """
        Return the audio samples and sample rate for playback.

        Returns:
        - samples (np.ndarray): The audio samples.
        - sample_rate (int): The sample rate of the audio.
        """
        return self.samples, self.sample_rate

    def get_waveform_figure(self, xlim=None, ylim=None):
        """
        Generate an interactive Plotly figure of the audio waveform.

        Parameters:
        - xlim (tuple): Limits for the x-axis (start_time, end_time) in seconds.
        - ylim (tuple): Limits for the y-axis (min_amplitude, max_amplitude).

        Returns:
        - fig (plotly.graph_objs.Figure): The Plotly figure object.
        """
        num_channels = self.num_channels
        # Create subplots for each channel
        fig = make_subplots(rows=num_channels, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        time_axis = np.arange(self.samples.shape[1]) / self.sample_rate  # Time axis in seconds
        for c in range(num_channels):
            # Add a trace for each channel
            fig.add_trace(
                go.Scatter(x=time_axis, y=self.samples[c], mode='lines', name=f'Channel {c+1}'),
                row=c+1, col=1
            )
            if ylim:
                fig.update_yaxes(range=ylim, row=c+1, col=1)
        if xlim:
            fig.update_xaxes(range=xlim)
        fig.update_layout(
            dragmode='zoom',  # Enable box zooming
            height=300*num_channels,
            width=800,
            title_text="Waveform",
            showlegend=False
        )
        fig.update_xaxes(title_text='Time (s)', row=num_channels, col=1)
        fig.update_yaxes(title_text='Amplitude')
        return fig

    def RMS_energy(self, signal, frame_size, hop_length):
        """
        Calculate the Root Mean Square (RMS) energy of the signal.

        Parameters:
        - signal (np.ndarray): The audio signal.
        - frame_size (int): The number of samples per frame.
        - hop_length (int): The number of samples between successive frames.

        Returns:
        - res (np.ndarray): The RMS energy values.
        """
        res = []
        for i in range(0, len(signal), hop_length):
            cur_portion = signal[i:i + frame_size]
            rmse_val = np.sqrt(np.mean(cur_portion ** 2))
            res.append(rmse_val)
        return np.array(res)

    def get_multichannel_RMS_energy_figure(self, frame_size, hop_length):
        """
        Generate an interactive Plotly figure of the RMS energy for each channel.

        Parameters:
        - frame_size (int): The number of samples per frame.
        - hop_length (int): The number of samples between successive frames.

        Returns:
        - fig (plotly.graph_objs.Figure): The Plotly figure object.
        """
        num_channels = self.num_channels
        # Create subplots for each channel
        fig = make_subplots(rows=num_channels, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        for c in range(num_channels):
            # Calculate RMS energy
            rmse = self.RMS_energy(self.samples[c], frame_size, hop_length)
            frames = range(len(rmse))
            time = librosa.frames_to_time(frames, sr=self.sample_rate, hop_length=hop_length)
            # Add a trace for each channel
            fig.add_trace(
                go.Scatter(x=time, y=rmse, mode='lines', name=f'Channel {c+1}'),
                row=c+1, col=1
            )
        fig.update_layout(
            dragmode='zoom',  # Enable box zooming
            height=300*num_channels,
            width=800,
            title_text="RMS Energy",
            showlegend=False
        )
        fig.update_xaxes(title_text='Time (s)', row=num_channels, col=1)
        fig.update_yaxes(title_text='RMS Energy')
        return fig

    def crest_factor(self, signal, frame_size, hop_length):
        """
        Calculate the crest factor of the signal.

        Parameters:
        - signal (np.ndarray): The audio signal.
        - frame_size (int): The number of samples per frame.
        - hop_length (int): The number of samples between successive frames.

        Returns:
        - res (np.ndarray): The crest factor values.
        """
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
        """
        Generate an interactive Plotly figure of the crest factor for each channel.

        Parameters:
        - frame_size (int): The number of samples per frame.
        - hop_length (int): The number of samples between successive frames.

        Returns:
        - fig (plotly.graph_objs.Figure): The Plotly figure object.
        """
        num_channels = self.num_channels
        # Create subplots for each channel
        fig = make_subplots(rows=num_channels, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        for c in range(num_channels):
            # Calculate crest factor
            crest = self.crest_factor(self.samples[c], frame_size, hop_length)
            frames = range(len(crest))
            time = librosa.frames_to_time(frames, sr=self.sample_rate, hop_length=hop_length)
            # Add a trace for each channel
            fig.add_trace(
                go.Scatter(x=time, y=crest, mode='lines', name=f'Channel {c+1}'),
                row=c+1, col=1
            )
        fig.update_layout(
            dragmode='zoom',  # Enable box zooming
            height=300*num_channels,
            width=800,
            title_text="Crest Factor",
            showlegend=False
        )
        fig.update_xaxes(title_text='Time (s)', row=num_channels, col=1)
        fig.update_yaxes(title_text='Crest Factor')
        return fig

    def loudness_dbA(self, frame_size, hop_length, cutoff=2000, return_average=False):
        """
        Calculate the loudness of the audio in dB(A) and generate an interactive Plotly figure.

        Parameters:
        - frame_size (int): The number of samples per frame.
        - hop_length (int): The number of samples between successive frames.
        - cutoff (float): The cutoff frequency for the high-pass filter in Hz.
        - return_average (bool): Whether to return the average loudness values per channel.

        Returns:
        - If return_average is True:
            - average_loudness_values (list): Average loudness per channel.
            - fig (plotly.graph_objs.Figure): The Plotly figure object.
        - If return_average is False:
            - fig (plotly.graph_objs.Figure): The Plotly figure object.
        """
        # Apply a high-pass filter to remove frequencies below the cutoff
        nyquist = 0.5 * self.sample_rate
        norm_cutoff = cutoff / nyquist
        b, a = butter(4, norm_cutoff, btype='high', analog=False)
        filtered_samples = filtfilt(b, a, self.samples, axis=1)

        # Define A-weighting filter coefficients (analog)
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
        # Create subplots for each channel
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

            # Add a trace for each channel
            fig.add_trace(
                go.Scatter(x=time_axis, y=res_dbA, mode='lines', name=f'Channel {c+1}'),
                row=c+1, col=1
            )

        fig.update_layout(
            dragmode='zoom',  # Enable box zooming
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
        """
        Check if the audio duration meets the minimum duration requirement.

        Parameters:
        - min_duration (float): The minimum required duration in seconds.

        Returns:
        - meets_duration (bool): True if the duration meets or exceeds the minimum, False otherwise.
        - duration (float): The actual duration of the audio in seconds.
        """
        duration = self.samples.shape[1] / self.sample_rate
        if duration >= min_duration:
            return True, duration
        else:
            return False, duration






        
           
            

    
    