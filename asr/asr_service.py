import asyncio
import json
import sys
import re
import threading
import pyaudio
import websockets
import numpy as np
import scipy.signal
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from .api import API_KEY_ASSEMBLY

from std_srvs.srv import Trigger

# Fix for Windows + Python 3.13
# if sys.platform.startswith("win"):
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

DEACTIVATION_KEYWORDS = ["execute", "go ahead"]
EMERGENCY_STOP_KEYWORD = "stop"
EMERGENCY_START_KEYWORD = "okay"
CONFIRM_KEYWORD = "confirm"

URL = "wss://streaming.assemblyai.com/v3/ws"


class ASRPublisher(Node):
    def __init__(self):
        super().__init__('asr_publisher')

        # Publishers
        self.prompt_publisher = self.create_publisher(String, '/high_level_prompt', 10)
        self.emergency_publisher = self.create_publisher(Bool, '/emergency', 10)

        # Confirm service client
        self.confirm_service_client = self.create_client(Trigger, '/confirm')

        self.get_logger().info('✅ ASR Publisher Initialized with keyword detection')
        self.get_logger().info(f'Deactivation keywords: {DEACTIVATION_KEYWORDS}')
        self.get_logger().info(f'Emergency stop keyword: {EMERGENCY_STOP_KEYWORD}')
        self.get_logger().info(f'Emergency start keyword: {EMERGENCY_START_KEYWORD}')
        self.get_logger().info('Mode: Continuous listening until deactivation or emergency keyword')

        self._asr_started = False
        self._asr_thread = None
        
        # Start ASR immediately in background thread
        self.get_logger().info("🎙️ Starting ASR loop in background thread...")
        self._asr_thread = threading.Thread(target=self._run_asyncio_asr, daemon=True)
        self._asr_thread.start()

    def _run_asyncio_asr(self):
        """Runs the asyncio ASR loop in a dedicated thread"""
        try:
            self.get_logger().info("🔧 Setting up asyncio event loop in ASR thread...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.get_logger().info("🚀 Running ASR coroutine...")
            loop.run_until_complete(self.run_asr())
        except Exception as e:
            self.get_logger().error(f"💥 ASR thread crashed: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def check_for_keywords(self, text):
        text_lower = text.lower().strip()

        # Check for emergency stop
        if EMERGENCY_STOP_KEYWORD in text_lower:
            self.get_logger().warn(f'🚨 EMERGENCY STOP detected: "{text}"')
            return ('emergency_stop', None)
        
        # Check for emergency start (resume)
        if EMERGENCY_START_KEYWORD in text_lower:
            self.get_logger().info(f'✅ EMERGENCY START detected: "{text}"')
            return ('emergency_start', None)

        # Check for deactivation keywords
        for keyword in DEACTIVATION_KEYWORDS:
            if keyword in text_lower:
                cleaned_text = re.sub(
                    r'\b' + re.escape(keyword) + r'\b', '', text_lower, flags=re.IGNORECASE
                ).strip()
                cleaned_text = ' '.join(cleaned_text.split())
                self.get_logger().info(f'🗣️ Deactivation keyword "{keyword}" detected')
                return ('deactivation', cleaned_text)

        # Check for confirm keyword (confirm)
        if CONFIRM_KEYWORD in text_lower:
            self.get_logger().info(f'✅ CONFIRM keyword detected: "{text}"')
            return ('confirm', None)

        return (None, None)

    async def run_asr(self):
        """Continuously listen, stream audio, and publish when triggers occur"""
        self.get_logger().info("🎤 ASR run_asr() function started!")

        p = pyaudio.PyAudio()
        target_name = "ATR2500x-USB Microphone"
        device_index = None

        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            name = info.get('name')
            if target_name.lower() in name.lower() and info.get('maxInputChannels') > 0:
                device_index = i
                print(f"🎧 Selected device [{i}]: {name}")
                break

        if device_index is None:
            print("❌ Could not find matching device.")
        else:
            try:
                info = p.get_device_info_by_index(device_index)
                self.get_logger().info(f"🎧 Using input device {device_index}: {info.get('name')}")
            except Exception as e:
                self.get_logger().error(f"❌ Failed to get device info for index {device_index}: {e}")
                self.get_logger().info("Available audio devices:")
                for i in range(p.get_device_count()):
                    dev_info = p.get_device_info_by_index(i)
                    self.get_logger().info(f"  [{i}] {dev_info.get('name')} - Inputs: {dev_info.get('maxInputChannels')}")
                p.terminate()
                return

        try:
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=FRAMES_PER_BUFFER
            )
            self.get_logger().info("🎤 Audio stream opened successfully")
        except Exception as e:
            self.get_logger().error(f"❌ Failed to open audio stream: {e}")
            p.terminate()
            return

        while rclpy.ok():
            stop_event = asyncio.Event()
            try:
                self.get_logger().info(f"🔌 Connecting to AssemblyAI at {URL}...")
                async with websockets.connect(
                    URL,
                    additional_headers={"Authorization": API_KEY_ASSEMBLY},
                ) as ws:
                    session_begins = await ws.recv()
                    self.get_logger().info(f'🔗 ASR session started: {session_begins}')

                    async def send_audio():
                        while not stop_event.is_set():
                            try:
                                data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                                audio_np = np.frombuffer(data, dtype=np.int16)
                                resampled = scipy.signal.resample_poly(audio_np, 16000, RATE)
                                await ws.send(resampled.astype(np.int16).tobytes())
                            except Exception as e:
                                self.get_logger().error(f'🎧 Send error: {e}')
                                stop_event.set()
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
                                        self.get_logger().info(f'📝 Transcript: {text}')

                                        keyword_type, cleaned_text = self.check_for_keywords(text)

                                        # Emergency stop keyword → publish True
                                        if keyword_type == 'emergency_stop':
                                            msg = Bool()
                                            msg.data = True
                                            for _ in range(10):
                                                self.emergency_publisher.publish(msg)
                                                self.get_logger().warn('🚨 Published EMERGENCY STOP (True) to /emergency')
                                                await asyncio.sleep(0.01)
                                            stop_event.set()
                                            break
                                        
                                        # Emergency start keyword → publish False
                                        elif keyword_type == 'emergency_start':
                                            msg = Bool()
                                            msg.data = False
                                            self.emergency_publisher.publish(msg)
                                            self.get_logger().info('✅ Published EMERGENCY START (False) to /emergency')
                                            stop_event.set()
                                            break

                                        # Deactivation keyword → publish cleaned command
                                        elif keyword_type == 'deactivation':
                                            if cleaned_text:
                                                msg = String()
                                                msg.data = cleaned_text
                                                self.prompt_publisher.publish(msg)
                                                self.get_logger().info(f'📤 Published to /high_level_prompt: "{cleaned_text}"')
                                            else:
                                                self.get_logger().warn('⚠️ Deactivation keyword detected but no command text found')
                                            stop_event.set()
                                            break
                                        
                                        # Confirm keyword → call confirm service using threading
                                        elif keyword_type == 'confirm':
                                            # Use threading.Event to wait for service call
                                            result_event = threading.Event()
                                            result_container = [None]
                                            
                                            def call_confirm_service():
                                                try:
                                                    if self.confirm_service_client.wait_for_service(timeout_sec=5.0):
                                                        req = Trigger.Request()
                                                        
                                                        # Use callback instead of spin_until_future_complete
                                                        def response_callback(future):
                                                            try:
                                                                result_container[0] = future.result()
                                                                result_event.set()
                                                            except Exception as e:
                                                                self.get_logger().error(f'❌ Confirm callback error: {e}')
                                                                result_event.set()
                                                        
                                                        future = self.confirm_service_client.call_async(req)
                                                        future.add_done_callback(response_callback)
                                                    else:
                                                        self.get_logger().error('❌ Confirm service not available')
                                                        result_event.set()
                                                except Exception as e:
                                                    self.get_logger().error(f'❌ Confirm service call error: {e}')
                                                    result_event.set()
                                            
                                            # Call service in a separate thread
                                            service_thread = threading.Thread(target=call_confirm_service, daemon=True)
                                            service_thread.start()
                                            
                                            # Wait for result (convert to async wait)
                                            while not result_event.is_set():
                                                await asyncio.sleep(0.1)
                                            
                                            if result_container[0] is not None:
                                                self.get_logger().info(f'✅ Confirm service response: {result_container[0].message}')
                                            else:
                                                self.get_logger().error('❌ Confirm service call failed')
                                            
                                            stop_event.set()
                                            break

                            except Exception as e:
                                self.get_logger().error(f'💬 Receive error: {e}')
                                stop_event.set()
                                break

                    await asyncio.gather(send_audio(), receive_transcripts())

            except Exception as e:
                self.get_logger().error(f'❌ ASR session error: {e}')
                import traceback
                self.get_logger().error(traceback.format_exc())
                await asyncio.sleep(2.0)

        stream.stop_stream()
        stream.close()
        p.terminate()
        self.get_logger().info('🛑 ASR Publisher stopped.')


def main():
    rclpy.init()
    node = ASRPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("👋 Keyboard interrupt, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()