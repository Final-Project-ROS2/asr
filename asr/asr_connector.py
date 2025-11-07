#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.action import ActionClient
from custom_interfaces.action import Prompt

class HighLevelPromptClient(Node):
    def __init__(self):
        super().__init__('high_level_prompt_client')
        
        # Subscriber to /high_level_prompt (std_msgs/String)
        self.subscription = self.create_subscription(
            String,
            '/high_level_prompt',
            self.prompt_callback,
            10
        )
        self.subscription  # prevent unused variable warning

        # Action client for /prompt_high_level (custom_interfaces/Prompt)
        self.action_client = ActionClient(self, Prompt, '/prompt_high_level')

    def prompt_callback(self, msg: String):
        self.get_logger().info(f'Received prompt: "{msg.data}"')
        self.send_prompt_action(msg.data)

    def send_prompt_action(self, prompt_text: str):
        # Wait until the action server is available
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server /prompt_high_level not available!')
            return
        
        # Create goal
        goal_msg = Prompt.Goal()
        goal_msg.prompt = prompt_text
        
        # Send goal asynchronously
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
        self.get_logger().info(f'Feedback received: tools_called={feedback.tools_called}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Action finished. Success: {result.success}, Final Response: "{result.final_response}"')

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

if __name__ == '__main__':
    main()
