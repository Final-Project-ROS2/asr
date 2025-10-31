import asyncio
import json
import sys
import re

# Fix for Windows + Python 3.13
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import pyaudio
import websockets
from .api import API_KEY_ASSEMBLY

from custom_interfaces.srv import GetTranscript
from std_msgs.msg import String, Bool

import rclpy
from rclpy.node import Node

# Audio config
FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Silence detection config
SILENCE_THRESHOLD = 500
# Removed SILENCE_DURATION - continuous listening mode

# Keyword triggers
DEACTIVATION_KEYWORDS = ["execute", "go ahead"]  # Changed to deactivation
EMERGENCY_KEYWORD = "stop"

# AssemblyAI streaming endpoint (v3)
URL = "wss://streaming.assemblyai.com/v3/ws"


class ASRService(Node):

    def __init__(self):
        super().__init__('asr_service')
        self.srv = self.create_service(GetTranscript, 'get_transcript', self.asr_callback)
        
        # Publishers
        self.prompt_publisher = self.create_publisher(String, '/prompt', 10)
        self.emergency_publisher = self.create_publisher(Bool, '/emergency', 10)
        
        self.get_logger().info('ASR Service Initialized with keyword detection')
        self.get_logger().info(f'Deactivation keywords: {DEACTIVATION_KEYWORDS}')
        self.get_logger().info(f'Emergency keyword: {EMERGENCY_KEYWORD}')
        self.get_logger().info('Mode: Continuous listening until deactivation keyword')

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

    def check_for_keywords(self, text):
        """
        Check if text contains deactivation or emergency keywords.
        Returns: ('deactivation', cleaned_text) or ('emergency', None) or (None, None)
        """
        text_lower = text.lower().strip()
        
        # Check for emergency keyword
        if EMERGENCY_KEYWORD in text_lower:
            self.get_logger().warn(f'EMERGENCY STOP detected: "{text}"')
            return ('emergency', None)
        
        # Check for deactivation keywords
        for keyword in DEACTIVATION_KEYWORDS:
            if keyword in text_lower:
                # Remove the deactivation keyword from the text
                cleaned_text = re.sub(
                    r'\b' + re.escape(keyword) + r'\b',
                    '',
                    text_lower,
                    flags=re.IGNORECASE
                ).strip()
                
                # Clean up extra spaces
                cleaned_text = ' '.join(cleaned_text.split())
                
                self.get_logger().info(f'Deactivation keyword "{keyword}" detected')
                self.get_logger().info(f'Cleaned prompt: "{cleaned_text}"')
                return ('deactivation', cleaned_text)
        
        return (None, None)

    async def run_asr(self):
        """Run ASR with keyword detection"""
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
        latest_sentence = ""
        stop_event = asyncio.Event()

        try:
            async with websockets.connect(
                URL,
                extra_headers={"Authorization": API_KEY_ASSEMBLY},
            ) as ws:

                session_begins = await ws.recv()
                self.get_logger().info(f'ASR session started: {session_begins}')

                async def send_audio():
                    while not stop_event.is_set():
                        try:
                            data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                            await ws.send(data)
                        except websockets.exceptions.ConnectionClosedError as e:
                            self.get_logger().error(f'Connection closed: {e}')
                            break
                        except Exception as e:
                            self.get_logger().error(f'Send error: {e}')
                            break
                        await asyncio.sleep(0.01)

                async def receive_transcripts():
                    nonlocal latest_sentence
                    while not stop_event.is_set():
                        try:
                            result_str = await ws.recv()
                            result = json.loads(result_str)
                            
                            # Log all received messages for debugging
                            self.get_logger().info(f'Received message type: {result.get("type")}')
                            
                            # Check for both "Turn" and "PartialTranscript" types
                            if result.get("type") == "Turn":
                                text = result.get("transcript", "")
                                if text:
                                    self.get_logger().info(f'Turn Transcript: {text}')
                                    transcripts.append(text)
                                    latest_sentence = text
                                    
                                    # Check for keywords
                                    keyword_type, cleaned_text = self.check_for_keywords(text)
                                    
                                    if keyword_type == 'emergency':
                                        # Publish emergency stop
                                        emergency_msg = Bool()
                                        emergency_msg.data = True
                                        self.emergency_publisher.publish(emergency_msg)
                                        self.get_logger().warn('Published EMERGENCY STOP to /emergency')
                                        
                                        # Stop the session immediately
                                        stop_event.set()
                                        break
                                        
                                    elif keyword_type == 'deactivation':
                                        # Publish cleaned prompt to /prompt topic
                                        if cleaned_text:  # Only publish if there's actual content
                                            prompt_msg = String()
                                            prompt_msg.data = cleaned_text
                                            self.prompt_publisher.publish(prompt_msg)
                                            self.get_logger().info(f'Published to /prompt: "{cleaned_text}"')
                                            
                                            # Stop listening after deactivation keyword
                                            self.get_logger().info('Deactivation keyword detected - stopping ASR session')
                                            stop_event.set()
                                            break
                                        else:
                                            self.get_logger().warn('Deactivation keyword detected but no command found')
                            
                            # Also show partial transcripts for debugging
                            elif result.get("type") == "PartialTranscript":
                                text = result.get("text", "")
                                if text:
                                    self.get_logger().info(f'Partial: {text}')
                                    
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