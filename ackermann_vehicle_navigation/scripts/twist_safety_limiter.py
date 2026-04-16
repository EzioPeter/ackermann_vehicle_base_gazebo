#!/usr/bin/env python3

import math

import rospy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


class TwistSafetyLimiter(object):
    def __init__(self):
        self.input_cmd_topic = rospy.get_param("~input_cmd_topic", "/cmd_vel")
        self.output_cmd_topic = rospy.get_param("~output_cmd_topic", "/cmd_vel_safe")
        self.scan_topic = rospy.get_param("~scan_topic", "/ackermann_vehicle/scan")
        self.goal_topic = rospy.get_param("~goal_topic", "/ego_goal")

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
        self.idle_escape_enable = rospy.get_param("~idle_escape_enable", True)
        self.idle_escape_timeout = rospy.get_param("~idle_escape_timeout", 2.0)
        self.idle_escape_duration = rospy.get_param("~idle_escape_duration", 1.4)
        self.idle_escape_forward_speed = rospy.get_param("~idle_escape_forward_speed", 0.12)
        self.idle_escape_reverse_speed = rospy.get_param("~idle_escape_reverse_speed", 0.10)
        self.idle_escape_turn_angular = rospy.get_param("~idle_escape_turn_angular", 0.55)
        self.idle_escape_front_clearance = rospy.get_param("~idle_escape_front_clearance", 0.95)
        self.idle_escape_cmd_epsilon = rospy.get_param("~idle_escape_cmd_epsilon", 0.02)
        self.idle_escape_goal_timeout = rospy.get_param("~idle_escape_goal_timeout", 60.0)

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
        self.idle_escape_timeout = max(0.2, float(self.idle_escape_timeout))
        self.idle_escape_duration = max(0.2, float(self.idle_escape_duration))
        self.idle_escape_forward_speed = max(0.0, min(float(self.idle_escape_forward_speed), float(self.max_linear_x)))
        self.idle_escape_reverse_speed = max(0.0, min(float(self.idle_escape_reverse_speed), float(self.max_linear_x)))
        self.idle_escape_turn_angular = max(0.0, min(float(self.idle_escape_turn_angular), float(self.max_angular_z)))
        self.idle_escape_front_clearance = max(0.0, float(self.idle_escape_front_clearance))
        self.idle_escape_cmd_epsilon = max(0.0, float(self.idle_escape_cmd_epsilon))
        self.idle_escape_goal_timeout = max(1.0, float(self.idle_escape_goal_timeout))

        self.front_min_range = None
        self.left_min_range = None
        self.right_min_range = None
        self.current_speed = 0.0
        self.last_goal_time = rospy.Time(0)
        self.stuck_since = None
        self.idle_since = None
        self.idle_escape_active_until = rospy.Time(0)
        self.idle_escape_turn_sign = 1.0
        self.idle_escape_reverse = False
        self.unstick_active_until = rospy.Time(0)
        self.unstick_turn_sign = 1.0

        self.pub = rospy.Publisher(self.output_cmd_topic, Twist, queue_size=10)
        self.scan_sub = rospy.Subscriber(self.scan_topic, LaserScan, self.scan_cb, queue_size=10)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_cb, queue_size=10)
        self.goal_sub = rospy.Subscriber(self.goal_topic, PoseStamped, self.goal_cb, queue_size=10)
        self.cmd_sub = rospy.Subscriber(self.input_cmd_topic, Twist, self.cmd_cb, queue_size=10)

    def scan_cb(self, msg):
        half_angle = math.radians(self.front_angle_deg)
        side_inner = math.radians(max(10.0, self.front_angle_deg * 0.5))
        side_outer = math.radians(115.0)
        angle = msg.angle_min
        min_range = None
        left_min = None
        right_min = None

        for r in msg.ranges:
            if math.isfinite(r) and msg.range_min <= r <= msg.range_max:
                if -half_angle <= angle <= half_angle:
                    if min_range is None or r < min_range:
                        min_range = r
                elif side_inner <= angle <= side_outer:
                    if left_min is None or r < left_min:
                        left_min = r
                elif -side_outer <= angle <= -side_inner:
                    if right_min is None or r < right_min:
                        right_min = r
            angle += msg.angle_increment

        self.front_min_range = min_range
        self.left_min_range = left_min
        self.right_min_range = right_min

    def odom_cb(self, msg):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.current_speed = math.hypot(vx, vy)

    def goal_cb(self, _msg):
        self.last_goal_time = rospy.Time.now()

    def has_recent_goal(self, now):
        if self.last_goal_time == rospy.Time(0):
            return False
        return (now - self.last_goal_time).to_sec() <= self.idle_escape_goal_timeout

    def choose_turn_sign(self):
        left = self.left_min_range if self.left_min_range is not None else 0.0
        right = self.right_min_range if self.right_min_range is not None else 0.0
        if abs(left - right) < 0.05:
            self.idle_escape_turn_sign *= -1.0
        else:
            self.idle_escape_turn_sign = 1.0 if left > right else -1.0
        return self.idle_escape_turn_sign

    def apply_idle_escape(self, out):
        turn_sign = self.idle_escape_turn_sign
        front = self.front_min_range if self.front_min_range is not None else self.idle_escape_front_clearance

        if self.idle_escape_reverse:
            out.linear.x = -self.idle_escape_reverse_speed
            out.angular.z = turn_sign * self.idle_escape_turn_angular
        elif front < self.idle_escape_front_clearance:
            out.linear.x = max(self.turn_assist_speed, self.idle_escape_forward_speed * 0.5)
            out.angular.z = turn_sign * self.idle_escape_turn_angular
        else:
            out.linear.x = self.idle_escape_forward_speed
            out.angular.z = turn_sign * self.idle_escape_turn_angular

    def maybe_start_idle_escape(self, out, raw_cmd):
        if not self.idle_escape_enable:
            self.idle_since = None
            return

        now = rospy.Time.now()

        if now < self.idle_escape_active_until:
            self.apply_idle_escape(out)
            return

        if not self.has_recent_goal(now):
            self.idle_since = None
            return

        raw_active = (
            abs(raw_cmd.linear.x) > self.idle_escape_cmd_epsilon or
            abs(raw_cmd.linear.y) > self.idle_escape_cmd_epsilon or
            abs(raw_cmd.angular.z) > self.idle_escape_cmd_epsilon
        )
        almost_not_moving = self.current_speed < self.unstick_speed_epsilon

        if raw_active or not almost_not_moving:
            self.idle_since = None
            return

        if self.idle_since is None:
            self.idle_since = now
            return

        if (now - self.idle_since).to_sec() < self.idle_escape_timeout:
            return

        front = self.front_min_range if self.front_min_range is not None else self.idle_escape_front_clearance
        self.idle_escape_turn_sign = self.choose_turn_sign()
        self.idle_escape_reverse = front <= self.reverse_escape_distance
        self.idle_escape_active_until = now + rospy.Duration(self.idle_escape_duration)
        self.idle_since = None
        self.apply_idle_escape(out)
        rospy.logwarn_throttle(
            2.0,
            "Planner command is idle while exploration has an active goal; injecting escape cmd v=%.2f w=%.2f",
            out.linear.x,
            out.angular.z,
        )

    def maybe_start_unstick(self, out, raw_forward=False):
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
        trying_to_go_forward = raw_forward or out.linear.x > 0.02
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
        raw_forward = msg.linear.x > 0.02

        self.maybe_start_idle_escape(out, msg)

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
                    if self.current_speed < self.unstick_speed_epsilon:
                        if self.front_min_range <= self.reverse_escape_distance:
                            out.linear.x = -self.reverse_escape_speed
                        else:
                            out.linear.x = self.turn_assist_speed
                        out.angular.z = self.choose_turn_sign() * max(
                            self.turn_assist_min_angular,
                            min(self.max_angular_z, self.idle_escape_turn_angular),
                        )
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

            self.maybe_start_unstick(out, raw_forward)
        elif out.linear.x <= 0.0:
            self.stuck_since = None

        self.pub.publish(out)


if __name__ == "__main__":
    rospy.init_node("twist_safety_limiter")
    TwistSafetyLimiter()
    rospy.spin()
