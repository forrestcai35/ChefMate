from transformers import WhisperProcessor, WhisperForConditionalGeneration
import whisper
import pyaudio
import wave
import os
import tempfile
import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()

class AudioTranscriber:
    def __init__(self, model_name="base"):
        # Load the Whisper model
        self.model = whisper.load_model("medium")
        # Initialize PyAudio
        self.pyaudio = pyaudio.PyAudio()

    def record_audio(self, record_seconds=5, wav_output_filename="output.wav"):
        # Audio recording parameters
        print(f"* Start recording.")

        chunk = 1024
        format = pyaudio.paInt16
        channels = 1
        rate = 44100

        stream = self.pyaudio.open(format=format,
          channels=channels,
          rate=rate,
          input=True,
          frames_per_buffer=chunk)

        print(f"* Recording audio for {record_seconds} seconds.")

        frames = []

        for _ in range(0, int(rate / chunk * record_seconds)):
            data = stream.read(chunk)
            frames.append(data)

        print("* Done recording.")

        stream.stop_stream()
        stream.close()
        self.pyaudio.terminate()

        # Save the recorded data as a WAV file
        wf = wave.open(wav_output_filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(self.pyaudio.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        return wav_output_filename

    def transcribe_audio(self, audio_path):
        # Use Whisper to transcribe the audio file
        result = self.model.transcribe(audio_path)
        transcription = result["text"]
        return transcription

    def transcribe_from_microphone(self, record_seconds=5):
        # Record audio from microphone
        with tempfile.TemporaryDirectory() as tempdir:
            temp_audio_path = os.path.join(tempdir, "temp_audio.wav")
            self.record_audio(record_seconds, temp_audio_path)
            # Transcribe the recorded audio
            transcription = self.transcribe_audio(temp_audio_path)
            print("Transcription:", transcription)
            return transcription

# Testing
# transcriber = AudioTranscriber()
# transcribe_text = transcriber.transcribe_from_microphone(5)
# print(transcribe_text)

# Note: The example usage is commented out to prevent execution in this environment.
#       To use this script, uncomment the example usage, and run the script in an environment where Whisper and PyAudio are installed.
