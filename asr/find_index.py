import pyaudio

p = pyaudio.PyAudio()

for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f"Index {i}: {info['name']}")
    print(f"   Max input channels: {info['maxInputChannels']}")
    print(f"   Default sample rate: {info['defaultSampleRate']}\n")

p.terminate()
