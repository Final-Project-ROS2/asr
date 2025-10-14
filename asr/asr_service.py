import asyncio
import json
import sys
import time

# Fix for Windows + Python 3.13
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import pyaudio
import websockets
from .api import API_KEY_ASSEMBLY

from custom_interfaces.srv import GetTranscript  # Ensure this is the correct import path

import rclpy
from rclpy.node import Node

# Audio config
FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Silence detection config
SILENCE_THRESHOLD = 500  # Adjust based on your environment (lower = more sensitive)
SILENCE_DURATION = 3.0   # Seconds of silence before stopping

# AssemblyAI new streaming endpoint (v3)
URL = "wss://streaming.assemblyai.com/v3/ws"


class ASRService(Node):

    def __init__(self):
        super().__init__('asr_service')
        self.srv = self.create_service(GetTranscript, 'get_transcript', self.asr_callback)
        self.get_logger().info('ASR Service Initialized')

    def asr_callback(self, request, response):
        self.get_logger().info("Incoming request: duration=%d seconds" % request.duration)

        try:
            transcript = asyncio.run(self.run_asr())
            response.transcript = transcript
            response.success = True
        except Exception as e:
            self.get_logger().error(f'ASR Error: {str(e)}')
            response.transcript = ''
            response.success = False

        return response

    def is_silent(self, data):
        """Check if audio data is below silence threshold"""
        import numpy as np
        audio_data = np.frombuffer(data, dtype=np.int16)
        return np.abs(audio_data).mean() < SILENCE_THRESHOLD

    async def run_asr(self):
        """Run ASR and return list of transcripts"""
        p = pyaudio.PyAudio()

        device_index = None
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info.get('maxInputChannels') > 0:
                device_index = i
                self.get_logger().info(f"Using input device {i}: {info.get('name')}")
                break

        if device_index is None:
            raise OSError("No microphone input device found!")

        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=FRAMES_PER_BUFFER
        )

        transcripts = []
        stop_event = asyncio.Event()
        last_audio_time = time.time()

        try:
            async with websockets.connect(
                URL,
                extra_headers={"Authorization": API_KEY_ASSEMBLY},
            ) as ws:

                session_begins = await ws.recv()
                self.get_logger().info(f'ASR session started: {session_begins}')

                async def send_audio():
                    nonlocal last_audio_time
                    while not stop_event.is_set():
                        try:
                            data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                            
                            # Check for silence
                            if not self.is_silent(data):
                                last_audio_time = time.time()
                            
                            # Check if silence duration exceeded
                            if time.time() - last_audio_time > SILENCE_DURATION:
                                self.get_logger().info(f'No audio input for {SILENCE_DURATION} seconds. Stopping...')
                                stop_event.set()
                                break
                            
                            await ws.send(data)
                        except websockets.exceptions.ConnectionClosedError as e:
                            self.get_logger().error(f'Connection closed: {e}')
                            break
                        except Exception as e:
                            self.get_logger().error(f'Send error: {e}')
                            break
                        await asyncio.sleep(0.01)

                async def receive_transcripts():
                    while not stop_event.is_set():
                        try:
                            result_str = await ws.recv()
                            result = json.loads(result_str)
                            
                            if result.get("type") == "Turn":
                                text = result.get("transcript", "")
                                if text:
                                    self.get_logger().info(f'Transcript: {text}')
                                    transcripts.append(text)
                                    
                        except websockets.exceptions.ConnectionClosedError as e:
                            self.get_logger().error(f'Connection closed: {e}')
                            break
                        except Exception as e:
                            self.get_logger().error(f'Receive error: {e}')
                            break

                try:
                    await asyncio.gather(send_audio(), receive_transcripts())
                except asyncio.CancelledError:
                    pass

        finally:
            stop_event.set()
            stream.stop_stream()
            stream.close()
            p.terminate()
            self.get_logger().info('ASR stopped')

        return transcripts


def main():
    rclpy.init()

    asr_service = ASRService()

    rclpy.spin(asr_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()