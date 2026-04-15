#!/usr/bin/env python3

import math

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


class TwistSafetyLimiter(object):
    def __init__(self):
        self.input_cmd_topic = rospy.get_param("~input_cmd_topic", "/cmd_vel")
        self.output_cmd_topic = rospy.get_param("~output_cmd_topic", "/cmd_vel_safe")
        self.scan_topic = rospy.get_param("~scan_topic", "/ackermann_vehicle/scan")

        self.max_linear_x = rospy.get_param("~max_linear_x", 0.28)
        self.max_angular_z = rospy.get_param("~max_angular_z", 0.45)
        self.stop_distance = rospy.get_param("~stop_distance", 0.9)
        self.slow_distance = rospy.get_param("~slow_distance", 1.8)
        self.front_angle_deg = rospy.get_param("~front_angle_deg", 40.0)
        self.min_slow_scale = rospy.get_param("~min_slow_scale", 0.35)
        self.slowdown_exponent = rospy.get_param("~slowdown_exponent", 0.6)
        self.allow_turn_in_place = rospy.get_param("~allow_turn_in_place", True)
        self.turn_assist_speed = rospy.get_param("~turn_assist_speed", 0.06)
        self.turn_assist_min_angular = rospy.get_param("~turn_assist_min_angular", 0.25)
        self.reverse_escape_distance = rospy.get_param("~reverse_escape_distance", 0.32)
        self.reverse_escape_speed = rospy.get_param("~reverse_escape_speed", 0.05)
        self.unstick_enable = rospy.get_param("~unstick_enable", True)
        self.unstick_trigger_distance = rospy.get_param("~unstick_trigger_distance", 0.36)
        self.unstick_speed_epsilon = rospy.get_param("~unstick_speed_epsilon", 0.015)
        self.unstick_timeout = rospy.get_param("~unstick_timeout", 1.5)
        self.unstick_duration = rospy.get_param("~unstick_duration", 1.1)
        self.unstick_reverse_speed = rospy.get_param("~unstick_reverse_speed", 0.10)
        self.unstick_turn_angular = rospy.get_param("~unstick_turn_angular", 0.70)

        # Keep parameters in safe numeric ranges.
        self.min_slow_scale = max(0.0, min(1.0, float(self.min_slow_scale)))
        self.slowdown_exponent = max(0.1, float(self.slowdown_exponent))
        self.slow_distance = max(float(self.slow_distance), float(self.stop_distance) + 1e-3)
        self.turn_assist_speed = max(0.0, min(float(self.turn_assist_speed), float(self.max_linear_x)))
        self.turn_assist_min_angular = max(0.0, float(self.turn_assist_min_angular))
        self.reverse_escape_distance = max(0.0, float(self.reverse_escape_distance))
        self.reverse_escape_speed = max(0.0, min(float(self.reverse_escape_speed), float(self.max_linear_x)))
        self.unstick_trigger_distance = max(0.0, float(self.unstick_trigger_distance))
        self.unstick_speed_epsilon = max(0.0, float(self.unstick_speed_epsilon))
        self.unstick_timeout = max(0.2, float(self.unstick_timeout))
        self.unstick_duration = max(0.2, float(self.unstick_duration))
        self.unstick_reverse_speed = max(0.0, min(float(self.unstick_reverse_speed), float(self.max_linear_x)))
        self.unstick_turn_angular = max(0.0, min(float(self.unstick_turn_angular), float(self.max_angular_z)))

        self.front_min_range = None
        self.current_speed = 0.0
        self.stuck_since = None
        self.unstick_active_until = rospy.Time(0)
        self.unstick_turn_sign = 1.0

        self.pub = rospy.Publisher(self.output_cmd_topic, Twist, queue_size=10)
        self.scan_sub = rospy.Subscriber(self.scan_topic, LaserScan, self.scan_cb, queue_size=10)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_cb, queue_size=10)
        self.cmd_sub = rospy.Subscriber(self.input_cmd_topic, Twist, self.cmd_cb, queue_size=10)

    def scan_cb(self, msg):
        half_angle = math.radians(self.front_angle_deg)
        angle = msg.angle_min
        min_range = None

        for r in msg.ranges:
            if -half_angle <= angle <= half_angle and math.isfinite(r):
                if msg.range_min <= r <= msg.range_max:
                    if min_range is None or r < min_range:
                        min_range = r
            angle += msg.angle_increment

        self.front_min_range = min_range

    def odom_cb(self, msg):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.current_speed = math.hypot(vx, vy)

    def maybe_start_unstick(self, out):
        if not self.unstick_enable:
            self.stuck_since = None
            return

        now = rospy.Time.now()

        if now < self.unstick_active_until:
            out.linear.x = -self.unstick_reverse_speed
            out.angular.z = self.unstick_turn_sign * self.unstick_turn_angular
            return

        if self.front_min_range is None:
            self.stuck_since = None
            return

        close_to_wall = self.front_min_range <= self.unstick_trigger_distance
        trying_to_go_forward = out.linear.x > 0.02
        almost_not_moving = self.current_speed < self.unstick_speed_epsilon

        if close_to_wall and trying_to_go_forward and almost_not_moving:
            if self.stuck_since is None:
                self.stuck_since = now
            elif (now - self.stuck_since).to_sec() >= self.unstick_timeout:
                self.unstick_active_until = now + rospy.Duration(self.unstick_duration)
                self.unstick_turn_sign *= -1.0
                out.linear.x = -self.unstick_reverse_speed
                out.angular.z = self.unstick_turn_sign * self.unstick_turn_angular
                self.stuck_since = None
        else:
            self.stuck_since = None

    def cmd_cb(self, msg):
        out = Twist()
        out.linear.x = max(min(msg.linear.x, self.max_linear_x), -self.max_linear_x)
        out.angular.z = max(min(msg.angular.z, self.max_angular_z), -self.max_angular_z)

        if self.front_min_range is not None and out.linear.x > 0.0:
            if self.front_min_range <= self.stop_distance:
                # Ackermann steering cannot rotate in place at v=0.
                # Keep a tiny crawl while turning to escape local deadlocks.
                # If too close to a wall, back up first instead of pushing forward.
                if self.allow_turn_in_place and abs(out.angular.z) >= self.turn_assist_min_angular:
                    if self.front_min_range <= self.reverse_escape_distance:
                        out.linear.x = -self.reverse_escape_speed
                    else:
                        out.linear.x = self.turn_assist_speed
                else:
                    out.linear.x = 0.0
                    out.angular.z = 0.0
            elif self.front_min_range < self.slow_distance:
                normalized = (self.front_min_range - self.stop_distance) / (self.slow_distance - self.stop_distance)
                normalized = max(0.0, min(1.0, normalized))
                # Use a nonlinear slowdown curve with a floor to avoid crawling.
                scale = self.min_slow_scale + (1.0 - self.min_slow_scale) * (normalized ** self.slowdown_exponent)
                out.linear.x = min(out.linear.x, self.max_linear_x * scale)
                if self.allow_turn_in_place and abs(out.angular.z) >= self.turn_assist_min_angular:
                    out.linear.x = max(out.linear.x, self.turn_assist_speed)

            self.maybe_start_unstick(out)

        self.pub.publish(out)


if __name__ == "__main__":
    rospy.init_node("twist_safety_limiter")
    TwistSafetyLimiter()
    rospy.spin()
