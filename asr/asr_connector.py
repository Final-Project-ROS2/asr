#!/usr/bin/env python3

import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.action import ActionClient
from custom_interfaces.action import Prompt

from TTS.api import TTS
import sounddevice as sd


class HighLevelPromptClient(Node):
    def __init__(self):
        super().__init__('high_level_prompt_client')

        # Subscriber to /high_level_prompt
        self.subscription = self.create_subscription(
            String,
            '/high_level_prompt',
            self.prompt_callback,
            10
        )

        # Action client for /prompt_high_level
        self.action_client = ActionClient(self, Prompt, '/prompt_high_level')

        # Load TTS once at startup
        self.get_logger().info('Loading Coqui TTS model...')
        self.tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
        self.sample_rate = self.tts.synthesizer.output_sample_rate
        self.get_logger().info(f'TTS loaded. sample_rate={self.sample_rate}')

        # Prevent overlapping speech
        self._speak_lock = threading.Lock()

    def prompt_callback(self, msg: String):
        self.get_logger().info(f'Received prompt: "{msg.data}"')
        self.send_prompt_action(msg.data)

    def send_prompt_action(self, prompt_text: str):
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server /prompt_high_level not available!')
            return

        goal_msg = Prompt.Goal()
        goal_msg.prompt = prompt_text

        self._send_goal_future = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by action server.')
            return

        self.get_logger().info('Goal accepted, waiting for result...')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'Feedback received: tools_called={feedback.tools_called}'
        )

    def get_result_callback(self, future):
        result = future.result().result
        final_response = result.final_response.strip()

        self.get_logger().info(
            f'Action finished. Success: {result.success}, Final Response: "{final_response}"'
        )

        if result.success and final_response:
            self.speak_text(final_response)

    def speak_text(self, text: str):
        # Run TTS in a background thread so ROS callbacks stay responsive
        threading.Thread(
            target=self._speak_text_worker,
            args=(text,),
            daemon=True
        ).start()

    def _speak_text_worker(self, text: str):
        with self._speak_lock:
            try:
                self.get_logger().info(f'Speaking: "{text}"')
                wav = self.tts.tts(text=text)
                sd.stop()
                sd.play(wav, samplerate=self.sample_rate)
                sd.wait()
            except Exception as e:
                self.get_logger().error(f'TTS playback failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = HighLevelPromptClient()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()