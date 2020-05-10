from kivy.graphics import Rectangle, Color
from kivy.uix.widget import Widget
from kivy.properties import *
import numpy as np
import pyaudio
import wave
import matplotlib.pyplot as plt
import io
import os
from google.cloud import speech_v1
import colorsys
import threading

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

class Transcriber:

    def __init__(self):
        self.job_callbacks = {}
        self.job_count = 0
        self.completed_jobs = {}
        self.lock = threading.Lock()

    def transcribe(self, audio_buffer, callback):
        self.lock.acquire()
        self.job_callbacks[self.job_count] = callback
        t = threading.Thread(target=self._transcribe_worker, args=(self.job_count, audio_buffer))
        t.start()
        self.job_count += 1
        self.lock.release()

    def _transcribe_worker(self, job_id, audio_buffer):
        client = speech_v1.SpeechClient()
        config = {
            "enable_word_time_offsets": True,
            "language_code": "en-US",
        }
        audio = {"content": audio_buffer}

        print("Recognizing...")
        response = client.recognize(config, audio)

        # The first result includes start and end time word offsets
        # First alternative is the most probable result
        # alternative = result.alternatives[0]
        transcript = ', '.join([result.alternatives[0].transcript for result in response.results])
        print(u"Transcript: {}".format(transcript))
        # plt.figure()
        # all_data = np.frombuffer(buf.getvalue(), dtype=np.int16).astype(float) / 32768
        # plt.plot(all_data)
        # plt.title(alternative.transcript)
        # plt.show()
        # Merge all results into one object
        result_obj = {'transcript': transcript}
        timestamps = []
        for result in response.results:
            for word in result.alternatives[0].words:
                timestamps.append({
                    'word': word.word,
                    'start_time': word.start_time.seconds + word.start_time.nanos / 1e9,
                    'end_time': word.end_time.seconds + word.end_time.nanos / 1e9
                })
        result_obj['timestamps'] = timestamps

        self.lock.acquire()
        self.completed_jobs[job_id] = result_obj
        self.lock.release()

    def call_callbacks(self):
        self.lock.acquire()
        for job_id, result in self.completed_jobs.items():
            self.job_callbacks[job_id](result)

        self.job_callbacks = {j: c for j, c in self.job_callbacks.items()
                              if j not in self.completed_jobs}
        self.completed_jobs = {}
        self.lock.release()

    def is_transcribing(self):
        return len(self.job_callbacks) > 0

class SpeechWidget(Widget):
    """
    Manages speech recording and transcription, as well as showing a speech indicator in the UI.
    """

    num_bars = NumericProperty(5)
    color = ListProperty([0.5, 0.5, 0.5, 1])
    bar_width = 32.0
    transcript = ObjectProperty(None, allownone=True)

    def __init__(self, **kwargs):
        super(SpeechWidget, self).__init__(**kwargs)
        self.bars = []
        self.color_elem = Color(*self.color)
        self.canvas.add(self.color_elem)
        self.frequencies = []
        self.amplitudes = []
        self.last_volumes = []
        self.is_recording = False
        self.audio = pyaudio.PyAudio()
        self.reset()
        self.on_silence = None
        self.transcriber = Transcriber()
        self.is_transcribing = False

    def reset(self):
        if self.is_recording:
            self.stop_recording(False)
        self.color = [0.5, 0.5, 0.5, 1]
        self.current_time = 0
        self.frames_since_sound = 0
        self.had_sound = False
        self.max_volume = 0
        self.should_stop_rec = False

        self.num_frames = 0
        self.buffer_str = io.BytesIO()
        self.buffer = wave.open(self.buffer_str, 'wb')
        self.buffer.setnchannels(1)
        self.buffer.setsampwidth(2)
        self.buffer.setframerate(RATE)
        self.stream = None
        self.is_recording = False
        self.is_paused = False

    def start_recording(self):
        # start Recording
        if self.num_frames:
            return
        self.is_recording = True
        print("Recording")
        self.stream = self.audio.open(format=FORMAT, channels=CHANNELS,
                                      rate=RATE, input=True,
                                      frames_per_buffer=CHUNK,
                                      stream_callback=self.update_recording)

    def pause_recording(self):
        self.is_paused = True

    def resume_recording(self):
        self.is_paused = False

    def update_recording(self, in_data, frame_count, time_info, status_flags):
        if self.is_paused:
            return (None, pyaudio.paContinue)
        arr = np.frombuffer(in_data, dtype=np.int16) / 32768
        self.buffer.writeframes(in_data)
        self.num_frames += 1
        volume = np.clip(np.sqrt(np.mean(arr ** 2)) * 6 + 0.2, 0, 1)
        self.last_volumes.append(volume)
        if len(self.last_volumes) > 10:
            self.last_volumes.pop(0)

        avg_volume = sum(self.last_volumes) / len(self.last_volumes)
        if self.had_sound and avg_volume < self.max_volume * 0.4:
            pass
            # self.frames_since_sound += 1
            # if self.frames_since_sound * CHUNK / RATE >= 2:
            #     self.should_stop_rec = True
            #     print("Silence")
            #     if self.on_silence:
            #         self.on_silence()
            #     return (None, pyaudio.paComplete)
        else:
            if avg_volume > 0.25 and not self.had_sound:
                print("Had sound")
                self.had_sound = True
            self.frames_since_sound = 0
            self.max_volume = max(avg_volume, self.max_volume)

        self.amplitudes = np.hanning(self.num_bars) * avg_volume
        if self.had_sound:
            self.color = (*colorsys.hsv_to_rgb((1 - avg_volume) * 0.7, 0.6, 0.9), 1)
        return (None, pyaudio.paContinue)

    def stop_recording(self, transcribe=True, callback=None):
        if not self.is_recording:
            return
        print("Finished recording")
        self.is_recording = False
        self.stream.stop_stream()
        self.stream.close()
        self.buffer.close()
        if transcribe:
            buf, start_time = self.trim_audio()
            value = buf.getvalue()
            assert callback, "Callback is required if transcribing"
            self.transcriber.transcribe(value, callback)
            return start_time

    def on_num_bars(self, instance, value):
        # Set the frequencies and amplitudes
        self.frequencies = [np.random.randint(10, 70) / 60.0]
        while len(self.frequencies) < value:
            self.frequencies.append(max((self.frequencies[-1] * 60.0 + np.random.randint(-10, 10)) / 60.0, 1 / 60))
        self.amplitudes = np.hanning(value) * 0.1

    def update(self, dt):
        # if self.should_stop_rec:
        #     self.stop_recording()
        #     self.should_stop_rec = False

        self.transcriber.call_callbacks()
        self.is_transcribing = self.transcriber.is_transcribing()

        self.color_elem.rgba = self.color
        center_pos = (self.pos[0] + self.size[0] / 2, self.pos[1] + self.size[1] / 2)
        origin_pos = (center_pos[0] - self.num_bars / 2 * self.bar_width, center_pos[1])
        for i in range(self.num_bars):
            if len(self.bars) <= i:
                bar = Rectangle()
                self.canvas.add(bar)
                self.bars.append(bar)
            else:
                bar = self.bars[i]
            height = self.amplitudes[i] * np.sin(2 * np.pi * self.frequencies[i] *
                                                 self.current_time) * 40.0
            if height >= 0:
                bar.pos = (origin_pos[0] + i * self.bar_width, origin_pos[1])
            else:
                bar.pos = (origin_pos[0] + i * self.bar_width, origin_pos[1] + height)
            bar.size = (self.bar_width, np.abs(height))
        self.current_time += dt

    def trim_audio(self):
        """Trims the audio to get rid of likely silence.

        Returns the new audio buffer as well as the time (seconds) that was trimmed from the start."""
        # Load a numpy array with the audio
        in_buf = io.BytesIO(self.buffer_str.getvalue())
        wav_file = wave.open(in_buf, 'rb')
        all_data = np.frombuffer(wav_file.readframes(wav_file.getnframes()),
                                 dtype=np.int16).astype(float) / 32768
        wav_file.close()

        # Divide into windows and compute energy
        window_len = 2048
        hop_size = 2048
        energies = []
        for start in range(0, len(all_data) - window_len, hop_size):
            energies.append(np.mean(all_data[start:start + window_len]) ** 2)

        # Get the likely noise threshold
        energies = np.array(energies)
        thresh = np.max(energies) * 0.1

        # Max filter
        max_window_len = int(RATE / hop_size) * 2 # 1 second windows
        filtered_energies = []
        for frame in range(len(energies)):
            filtered_energies.append(np.max(energies[max(frame - max_window_len // 2, 0):frame + max_window_len // 2]))

        # Find the first and last non-silence values and trim outside of that
        non_silence = np.nonzero(filtered_energies >= thresh)[0]
        trimmed_data = (all_data[non_silence[0] * hop_size:non_silence[-1] * hop_size] * 32768).astype(np.int16)

        buf = io.BytesIO()
        out_wav = wave.open(buf, 'wb')
        out_wav.setframerate(RATE)
        out_wav.setnchannels(CHANNELS)
        out_wav.setsampwidth(2)
        out_wav.writeframes(trimmed_data.tobytes())
        out_wav.close()
        with open("test_output.wav", "wb") as file:
            file.write(buf.getvalue())
        return buf, non_silence[0] * hop_size / RATE

