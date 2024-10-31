# app.py

import streamlit as st
from audio_processor import AudioProcessor
import numpy as np
import os
import tempfile

# Set the page configuration for the Streamlit app
st.set_page_config(page_title="Audio Processing Tool", layout="wide")

# Title of the application
st.title("Audio Processing Tool")

# Sidebar section for file information
st.sidebar.header("File Information")

# File Upload widget in the sidebar
uploaded_file = st.sidebar.file_uploader("Select an audio file", type=["wav", "mp3", "flac"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Save the uploaded file to a temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_file_path = tmp_file.name

    # Initialize the AudioProcessor with the temporary file path
    processor = AudioProcessor(tmp_file_path)

    # Display file information in the sidebar
    st.sidebar.write(f"**Filename:** {uploaded_file.name}")
    st.sidebar.write(f"**Sample Rate:** {processor.sample_rate} Hz")
    st.sidebar.write(f"**Channels:** {processor.num_channels}")
    # Calculate the duration of the audio
    duration = processor.samples.shape[1] / processor.sample_rate
    st.sidebar.write(f"**Duration:** {duration:.2f} seconds")

    # Main section: Module Selection
    st.header("Select Module")
    modules = ["Data Check Module", "Data Operation Module", "Data Analysis Module", "Decision Module"]
    # Dropdown to select the module
    selected_module = st.selectbox("Select Module", modules)

    st.write("---")  # Divider line

    # Display content based on the selected module
    if selected_module == "Data Check Module":
        st.subheader("Data Check Module")
        # Input for minimum duration
        min_duration = st.number_input("Minimum Duration (seconds)", min_value=0.0, value=0.0, step=0.1)
        # Button to trigger duration check
        if st.button("Check Duration"):
            with st.spinner('Checking audio duration...'):
                # Use the processor to check if the audio meets the minimum duration
                meets_duration, actual_duration = processor.check_duration(min_duration)
                if meets_duration:
                    st.success(f"Audio duration is {actual_duration:.2f}s, meets the minimum duration requirement.")
                    st.write("The audio meets the standard.")
                else:
                    st.error(f"Audio duration is {actual_duration:.2f}s, does not meet the minimum duration requirement.")
                    st.write("The audio does not meet the standard.")

    elif selected_module == "Data Operation Module":
        st.subheader("Data Operation Module")
        # Dropdown to select the operation to perform
        operation = st.selectbox("Select Operation", ["None", "Convert to WAV", "Print Stats", "Trim Audio",
                                                      "Rechannel Audio", "Resample Audio", "Play Audio"])

        if operation == "None":
            st.info("No operation selected.")

        elif operation == "Convert to WAV":
            # Button to trigger conversion to WAV format
            if st.button("Convert to WAV"):
                with st.spinner('Converting to WAV...'):
                    try:
                        # Perform conversion
                        message = processor.convert_to_wav()
                        st.success(message)
                        # Provide a download button for the converted file
                        output_file = os.path.join(tempfile.gettempdir(), os.path.basename(processor.output_file))
                        with open(output_file, 'rb') as f:
                            st.download_button(
                                label="Download Converted WAV",
                                data=f.read(),
                                file_name=os.path.basename(output_file),
                                mime="audio/wav"
                            )
                    except Exception as e:
                        st.error(f"Error converting to WAV: {e}")

        elif operation == "Print Stats":
            # Button to display audio statistics
            if st.button("Print Stats"):
                with st.spinner('Fetching audio statistics...'):
                    try:
                        # Get statistics from the processor
                        stats = processor.print_stats()
                        # Display the statistics in a text area
                        st.text_area("Audio Statistics", stats, height=200)
                        # Provide a download button for the statistics
                        stats_file = os.path.join(tempfile.gettempdir(), "audio_stats.txt")
                        with open(stats_file, 'w') as f:
                            f.write(stats)
                        with open(stats_file, 'rb') as f:
                            st.download_button(
                                label="Download Statistics",
                                data=f.read(),
                                file_name="audio_stats.txt",
                                mime="text/plain"
                            )
                    except Exception as e:
                        st.error(f"Error fetching statistics: {e}")

        elif operation == "Trim Audio":
            st.write("### Trim Audio")
            # Inputs for start and end times for trimming
            start_time = st.number_input("Start Time (seconds)", min_value=0.0, max_value=duration, value=0.0, step=0.1)
            end_time = st.number_input("End Time (seconds)", min_value=0.0, max_value=duration, value=duration, step=0.1)
            # Button to trigger audio trimming
            if st.button("Trim Audio"):
                with st.spinner('Trimming audio...'):
                    try:
                        # Perform trimming
                        message, output_path = processor.trim_audio(start_time, end_time)
                        st.success(message)
                        # Provide audio playback of the trimmed audio
                        st.audio(output_path, format='audio/wav')
                        # Provide a download button for the trimmed audio
                        with open(output_path, 'rb') as f:
                            st.download_button(
                                label="Download Trimmed Audio",
                                data=f.read(),
                                file_name=os.path.basename(output_path),
                                mime="audio/wav"
                            )
                    except Exception as e:
                        st.error(f"Error trimming audio: {e}")

        elif operation == "Rechannel Audio":
            st.write("### Rechannel Audio")
            # Input for target number of channels
            target_channels = st.number_input("Target Channels", min_value=1, max_value=2, value=1, step=1)
            # Button to trigger rechanneling
            if st.button("Rechannel Audio"):
                with st.spinner('Rechanneling audio...'):
                    try:
                        # Perform rechanneling
                        message, output_path = processor.rechannel_audio(target_channels)
                        st.success(message)
                        # Provide audio playback of the rechanneled audio
                        st.audio(output_path, format='audio/wav')
                        # Provide a download button for the rechanneled audio
                        with open(output_path, 'rb') as f:
                            st.download_button(
                                label="Download Rechanneled Audio",
                                data=f.read(),
                                file_name=os.path.basename(output_path),
                                mime="audio/wav"
                            )
                    except Exception as e:
                        st.error(f"Error rechanneling audio: {e}")

        elif operation == "Resample Audio":
            st.write("### Resample Audio")
            # Determine minimum and maximum allowable sample rates
            min_sr = int(processor.sample_rate / 2)
            max_sr = int(processor.sample_rate)
            # Input for target sample rate within the allowed range
            target_sr = st.number_input(
                "Target Sample Rate (Hz)",
                min_value=min_sr,
                max_value=max_sr,
                value=min_sr,
                step=1000
            )
            # Button to trigger resampling
            if st.button("Resample Audio"):
                with st.spinner('Resampling audio...'):
                    try:
                        # Perform resampling
                        message, output_path = processor.resample_audio(target_sr)
                        st.success(message)
                        # Provide audio playback of the resampled audio
                        st.audio(output_path, format='audio/wav')
                        # Provide a download button for the resampled audio
                        with open(output_path, 'rb') as f:
                            st.download_button(
                                label="Download Resampled Audio",
                                data=f.read(),
                                file_name=os.path.basename(output_path),
                                mime="audio/wav"
                            )
                    except ValueError as e:
                        st.error(f"Error resampling audio: {e}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")

        elif operation == "Play Audio":
            st.write("### Play Audio")
            # Provide audio playback of the original audio
            st.audio(tmp_file_path, format='audio/wav')
            st.info("Use the audio player controls to play, pause, or stop the audio.")

    elif selected_module == "Data Analysis Module":
        st.subheader("Data Analysis Module")
        # Dropdown to select the type of analysis
        analysis_type = st.selectbox("Select Analysis Type", ["None", "Plot Waveform", "Plot RMS Energy",
                                                              "Plot Crest Factor", "Plot Loudness"])

        if analysis_type == "None":
            st.info("No analysis selected.")

        elif analysis_type == "Plot Waveform":
            st.write("### Plot Waveform")
            # Inputs for axis limits
            xlim_text = st.text_input("X-axis Limits (e.g., 0,10)", value="")
            ylim_text = st.text_input("Y-axis Limits (e.g., -1,1)", value="")
            # Button to generate waveform plot
            if st.button("Plot Waveform"):
                with st.spinner('Plotting waveform...'):
                    try:
                        # Parse axis limits if provided
                        xlim = tuple(map(float, xlim_text.split(','))) if xlim_text else None
                        ylim = tuple(map(float, ylim_text.split(','))) if ylim_text else None
                        # Get the waveform figure from the processor
                        fig = processor.get_waveform_figure(xlim, ylim)
                        # Display the figure with interactive features
                        st.plotly_chart(fig, use_container_width=True, config={
                            'scrollZoom': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                        })
                    except Exception as e:
                        st.error(f"Error plotting waveform: {e}")

        elif analysis_type == "Plot RMS Energy":
            st.write("### Plot RMS Energy")
            # Inputs for frame size and hop length
            frame_size = st.number_input("Frame Size", min_value=256, max_value=8192, value=1024, step=256)
            hop_length = st.number_input("Hop Length", min_value=128, max_value=4096, value=512, step=128)
            # Button to generate RMS energy plot
            if st.button("Plot RMS Energy"):
                with st.spinner('Plotting RMS energy...'):
                    try:
                        # Get the RMS energy figure from the processor
                        fig = processor.get_multichannel_RMS_energy_figure(frame_size, hop_length)
                        # Display the figure with interactive features
                        st.plotly_chart(fig, use_container_width=True, config={
                            'scrollZoom': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                        })
                    except Exception as e:
                        st.error(f"Error plotting RMS energy: {e}")

        elif analysis_type == "Plot Crest Factor":
            st.write("### Plot Crest Factor")
            # Inputs for frame size and hop length
            frame_size = st.number_input("Frame Size", min_value=256, max_value=8192, value=1024, step=256)
            hop_length = st.number_input("Hop Length", min_value=128, max_value=4096, value=512, step=128)
            # Button to generate crest factor plot
            if st.button("Plot Crest Factor"):
                with st.spinner('Plotting crest factor...'):
                    try:
                        # Get the crest factor figure from the processor
                        fig = processor.get_multichannel_crest_factor_figure(frame_size, hop_length)
                        # Display the figure with interactive features
                        st.plotly_chart(fig, use_container_width=True, config={
                            'scrollZoom': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                        })
                    except Exception as e:
                        st.error(f"Error plotting crest factor: {e}")

        elif analysis_type == "Plot Loudness":
            st.write("### Plot Loudness")
            # Inputs for frame size, hop length, and cutoff frequency
            frame_size = st.number_input("Frame Size", min_value=256, max_value=8192, value=1024, step=256)
            hop_length = st.number_input("Hop Length", min_value=128, max_value=4096, value=512, step=128)
            cutoff = st.number_input("Cutoff Frequency (Hz)", min_value=20.0, max_value=20000.0, value=2000.0, step=100.0)
            # Button to generate loudness plot
            if st.button("Plot Loudness"):
                with st.spinner('Plotting loudness...'):
                    try:
                        # Get the loudness figure and average loudness values from the processor
                        avg_loudness, fig = processor.loudness_dbA(frame_size, hop_length, cutoff=cutoff, return_average=True)
                        # Display the figure with interactive features
                        st.plotly_chart(fig, use_container_width=True, config={
                            'scrollZoom': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                        })
                        # Display average loudness per channel
                        st.write("### Average Loudness per Channel")
                        for idx, loudness in enumerate(avg_loudness):
                            st.write(f"**Channel {idx+1}:** {loudness:.2f} dB(A)")
                    except Exception as e:
                        st.error(f"Error plotting loudness: {e}")

    elif selected_module == "Decision Module":
        st.subheader("Decision Module")
        # Inputs for reference loudness and allowed deviation
        reference_loudness = st.number_input("Reference Loudness (dB(A))", value=70.0, step=0.1)
        tolerance = st.number_input("Allowed Deviation (dB)", value=5.0, step=0.1)
        # Inputs for frame size and hop length
        frame_size = st.number_input("Frame Size", min_value=256, max_value=8192, value=1024, step=256)
        hop_length = st.number_input("Hop Length", min_value=128, max_value=4096, value=512, step=128)
        # Button to perform evaluation
        if st.button("Evaluate"):
            with st.spinner('Evaluating audio...'):
                try:
                    # Get average loudness values from the processor
                    average_loudness_values, _ = processor.loudness_dbA(
                        frame_size, hop_length, cutoff=2000, return_average=True
                    )
                    decision = True  # Initialize decision flag
                    # Evaluate each channel's loudness against the reference
                    for idx, avg_loudness in enumerate(average_loudness_values):
                        difference = abs(avg_loudness - reference_loudness)
                        if difference <= tolerance:
                            st.success(f"Channel {idx + 1}: Loudness {avg_loudness:.2f} dB(A) is within the acceptable range.")
                        else:
                            st.error(f"Channel {idx + 1}: Loudness {avg_loudness:.2f} dB(A) is outside the acceptable range.")
                            decision = False  # Set decision flag to False if any channel is out of range
                    # Display the overall conclusion based on the decision flag
                    if decision:
                        st.write("### Conclusion: The audio meets the standard.")
                    else:
                        st.write("### Conclusion: The audio does not meet the standard.")
                except Exception as e:
                    st.error(f"An error occurred during evaluation: {e}")

    # Clean up the temporary file after processing
    if uploaded_file is not None and os.path.exists(tmp_file_path):
        os.remove(tmp_file_path)








