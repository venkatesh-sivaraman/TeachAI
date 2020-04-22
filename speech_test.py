from google.cloud import speech_v1
import io

def sample_long_running_recognize(storage_uri):
    """
    Print start and end time of each word spoken in audio file from Cloud Storage

    Args:
      storage_uri URI for audio file in Cloud Storage, e.g. gs://[BUCKET]/[FILE]
    """

    client = speech_v1.SpeechClient()

    # storage_uri = 'gs://cloud-samples-data/speech/brooklyn_bridge.flac'

    # When enabled, the first result returned by the API will include a list
    # of words and the start and end time offsets (timestamps) for those words.
    enable_word_time_offsets = True

    # The language of the supplied audio
    language_code = "en-US"
    config = {
        "enable_word_time_offsets": enable_word_time_offsets,
        "language_code": language_code,
    }
    with io.open(storage_uri, "rb") as f:
        content = f.read()
    audio = {"content": content}

    # audio = {"uri": storage_uri}

    response = client.recognize(config, audio)

    print(u"Waiting for operation to complete...")

    # The first result includes start and end time word offsets
    result = response.results[0]
    # First alternative is the most probable result
    alternative = result.alternatives[0]
    print(u"Transcript: {}".format(alternative.transcript))
    # Print the start and end time of each word
    for word in alternative.words:
        print(u"Word: {}".format(word.word))
        print(
            u"Start time: {} seconds {} nanos".format(
                word.start_time.seconds, word.start_time.nanos
            )
        )
        print(
            u"End time: {} seconds {} nanos".format(
                word.end_time.seconds, word.end_time.nanos
            )
        )

sample_long_running_recognize("test_query.wav")
# import deepspeech
# model_file_path = '../deepspeech/deepspeech-0.6.1-models/output_graph.pbmm'
# beam_width = 500
# model = deepspeech.Model(model_file_path, beam_width)
# 
# lm_file_path = '../deepspeech/deepspeech-0.6.1-models/lm.binary'
# trie_file_path = '../deepspeech/deepspeech-0.6.1-models/trie'
# lm_alpha = 0.75
# lm_beta = 1.85
# model.enableDecoderWithLM(lm_file_path, trie_file_path, lm_alpha, lm_beta)
# 
# import numpy as np
# import pyaudio
# import time
# 
# context = model.createStream()
# 
# text_so_far = ''
# def process_audio(in_data, frame_count, time_info, status):
#     global text_so_far
#     data16 = np.frombuffer(in_data, dtype=np.int16)
#     model.feedAudioContent(context, data16)
#     text = model.intermediateDecode(context)
#     if text != text_so_far:
#         print('Interim text = {}'.format(text))
#         text_so_far = text
#     return (in_data, pyaudio.paContinue)
# 
# audio = pyaudio.PyAudio()
# stream = audio.open(
#     format=pyaudio.paInt16,
#     channels=1,
#     rate=16000,
#     input=True,
#     frames_per_buffer=1024,
#     stream_callback=process_audio
# )
# print('Please start speaking, when done press Ctrl-C ...')
# stream.start_stream()
# 
# try:
#     while stream.is_active():
#         time.sleep(0.1)
# except KeyboardInterrupt:
#     # PyAudio
#     stream.stop_stream()
#     stream.close()
#     audio.terminate()
#     print('Finished recording.')
#     # DeepSpeech
#     text = model.finishStreamWithMetadata(context)
#     for item in text.items:
#         print(item.character, item.start_time)


# buffer_len = 44100 * 3 #len(buffer)
# offset = 0
# batch_size = 16384
# text = ''
# while offset < buffer_len:
#     end_offset = offset + batch_size
#     chunk = buffer[offset:end_offset]
#     data16 = np.frombuffer(chunk, dtype=np.int16)
#     model.feedAudioContent(context, data16)
#     text = model.intermediateDecode(context)
#     print(text)
#     offset = end_offset

# import pyaudio
# 
# p = pyaudio.PyAudio()
# for i in range(p.get_device_count()):
#     print(p.get_device_info_by_index(i))
# #otherwise you can do:
# print(p.get_default_input_device_info())
# 
# import os
# from pocketsphinx import AudioFile, get_model_path, get_data_path
# 
# model_path = get_model_path()
# data_path = get_data_path()
# print(model_path)
# 
# config = {
#     'verbose': True,
#     'buffer_size': 2048,
#     'no_search': False,
#     'full_utt': False,
#     'hmm': os.path.join(model_path, 'en-us'),
#     'lm': os.path.join(model_path, 'en-us.lm.bin'),
#     'dict': os.path.join(model_path, 'cmudict-en-us.dict')
# }
# 
# for phrase in AudioFile(audio_file="/Users/venkatesh-sivaraman/Documents/School/MIT/6-835/fp/test_query.wav"):
#     print(phrase.segments(detailed=True)) # => "[('forward', -617, 63, 121)]"
# 
# from pocketsphinx import LiveSpeech
# speech = LiveSpeech(verbose=True)
# print(repr(speech))
# for phrase in speech:
#     print(phrase.segments(detailed=True))
