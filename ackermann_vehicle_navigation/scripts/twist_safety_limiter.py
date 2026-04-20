#!/usr/bin/env python3

import math

import rospy
from actionlib_msgs.msg import GoalStatusArray
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


class TwistSafetyLimiter(object):
    def __init__(self):
        self.input_cmd_topic = rospy.get_param("~input_cmd_topic", "/cmd_vel")
        self.output_cmd_topic = rospy.get_param("~output_cmd_topic", "/cmd_vel_safe")
        self.scan_topic = rospy.get_param("~scan_topic", "/ackermann_vehicle/scan")
        self.goal_topic = rospy.get_param("~goal_topic", "/move_base/current_goal")
        self.move_base_status_topic = rospy.get_param("~move_base_status_topic", "/move_base/status")
        self.external_pause_param = rospy.get_param("~external_pause_param", "/explore/suspend_until")
        self.external_brake_param = rospy.get_param("~external_brake_param", "/teb_frontier/brake_until")

        self.max_linear_x = rospy.get_param("~max_linear_x", 0.35)
        self.max_angular_z = rospy.get_param("~max_angular_z", 0.60)
        self.stop_distance = rospy.get_param("~stop_distance", 0.9)
        self.slow_distance = rospy.get_param("~slow_distance", 1.8)
        self.front_angle_deg = rospy.get_param("~front_angle_deg", 40.0)
        self.collision_half_width = rospy.get_param("~collision_half_width", 0.42)
        self.collision_min_forward = rospy.get_param("~collision_min_forward", 0.10)
        self.rear_angle_deg = rospy.get_param("~rear_angle_deg", 70.0)
        self.rear_stop_distance = rospy.get_param("~rear_stop_distance", 0.60)
        self.rear_slow_distance = rospy.get_param("~rear_slow_distance", 1.20)
        self.rear_min_slow_scale = rospy.get_param("~rear_min_slow_scale", 0.12)
        self.turning_side_stop_distance = rospy.get_param("~turning_side_stop_distance", 0.45)
        self.turning_side_slow_distance = rospy.get_param("~turning_side_slow_distance", 0.85)
        self.turning_side_min_slow_scale = rospy.get_param("~turning_side_min_slow_scale", 0.10)
        self.min_slow_scale = rospy.get_param("~min_slow_scale", 0.42)
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
        self.idle_escape_require_active_goal = rospy.get_param("~idle_escape_require_active_goal", True)

        self.min_slow_scale = max(0.0, min(1.0, float(self.min_slow_scale)))
        self.rear_min_slow_scale = max(0.0, min(1.0, float(self.rear_min_slow_scale)))
        self.turning_side_min_slow_scale = max(0.0, min(1.0, float(self.turning_side_min_slow_scale)))
        self.slowdown_exponent = max(0.1, float(self.slowdown_exponent))
        self.slow_distance = max(float(self.slow_distance), float(self.stop_distance) + 1e-3)
        self.collision_half_width = max(0.05, float(self.collision_half_width))
        self.collision_min_forward = max(0.0, float(self.collision_min_forward))
        self.rear_stop_distance = max(0.0, float(self.rear_stop_distance))
        self.rear_slow_distance = max(float(self.rear_slow_distance), float(self.rear_stop_distance) + 1e-3)
        self.turning_side_stop_distance = max(0.0, float(self.turning_side_stop_distance))
        self.turning_side_slow_distance = max(
            float(self.turning_side_stop_distance) + 1e-3,
            float(self.turning_side_slow_distance),
        )
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
        self.front_sector_min_range = None
        self.rear_min_range = None
        self.left_min_range = None
        self.right_min_range = None
        self.current_speed = 0.0
        self.last_goal_time = rospy.Time(0)
        self.last_active_goal_time = rospy.Time(0)
        self.move_base_has_active_goal = False
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
        self.status_sub = rospy.Subscriber(self.move_base_status_topic, GoalStatusArray, self.status_cb, queue_size=10)
        self.cmd_sub = rospy.Subscriber(self.input_cmd_topic, Twist, self.cmd_cb, queue_size=10)

    def scan_cb(self, msg):
        half_angle = math.radians(self.front_angle_deg)
        rear_half_angle = math.radians(self.rear_angle_deg * 0.5)
        rear_inner = math.pi - rear_half_angle
        side_inner = math.radians(max(10.0, self.front_angle_deg * 0.5))
        side_outer = math.radians(115.0)
        angle = msg.angle_min
        min_range = None
        sector_min = None
        rear_min = None
        left_min = None
        right_min = None

        for value in msg.ranges:
            if math.isfinite(value) and msg.range_min <= value <= msg.range_max:
                if -half_angle <= angle <= half_angle:
                    if sector_min is None or value < sector_min:
                        sector_min = value
                    forward = value * math.cos(angle)
                    lateral = value * math.sin(angle)
                    if forward >= self.collision_min_forward and abs(lateral) <= self.collision_half_width:
                        if min_range is None or value < min_range:
                            min_range = value
                elif abs(angle) >= rear_inner:
                    if rear_min is None or value < rear_min:
                        rear_min = value
                elif side_inner <= angle <= side_outer:
                    if left_min is None or value < left_min:
                        left_min = value
                elif -side_outer <= angle <= -side_inner:
                    if right_min is None or value < right_min:
                        right_min = value
            angle += msg.angle_increment

        self.front_min_range = min_range
        self.front_sector_min_range = sector_min
        self.rear_min_range = rear_min
        self.left_min_range = left_min
        self.right_min_range = right_min

    def odom_cb(self, msg):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.current_speed = math.hypot(vx, vy)

    def goal_cb(self, _msg):
        self.last_goal_time = rospy.Time.now()

    def status_cb(self, msg):
        active = any(status.status in (0, 1) for status in msg.status_list)
        self.move_base_has_active_goal = active
        if active:
            self.last_active_goal_time = rospy.Time.now()

    def external_recovery_paused(self, now):
        try:
            suspend_until = float(rospy.get_param(self.external_pause_param, 0.0))
        except (TypeError, ValueError):
            return False
        return suspend_until > 0.0 and now < rospy.Time(suspend_until)

    def external_motion_braked(self, now):
        try:
            brake_until = float(rospy.get_param(self.external_brake_param, 0.0))
        except (TypeError, ValueError):
            return False
        return brake_until > 0.0 and now < rospy.Time(brake_until)

    def has_recent_goal(self, now):
        if self.external_recovery_paused(now):
            return False
        if self.idle_escape_require_active_goal:
            if self.move_base_has_active_goal:
                return True
            if self.last_active_goal_time == rospy.Time(0):
                return False
            return (now - self.last_active_goal_time).to_sec() <= 0.5
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
        if self.external_recovery_paused(now):
            self.idle_since = None
            self.idle_escape_active_until = rospy.Time(0)
            return

        if now < self.idle_escape_active_until:
            if self.has_recent_goal(now):
                self.apply_idle_escape(out)
            else:
                self.idle_escape_active_until = rospy.Time(0)
            return

        if not self.has_recent_goal(now):
            self.idle_since = None
            return

        raw_active = (
            abs(raw_cmd.linear.x) > self.idle_escape_cmd_epsilon
            or abs(raw_cmd.linear.y) > self.idle_escape_cmd_epsilon
            or abs(raw_cmd.angular.z) > self.idle_escape_cmd_epsilon
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

    def reverse_is_clear(self):
        return self.rear_min_range is not None and self.rear_min_range > self.rear_stop_distance

    def turning_side_range(self, angular_z):
        if angular_z > self.turn_assist_min_angular:
            return self.left_min_range
        if angular_z < -self.turn_assist_min_angular:
            return self.right_min_range
        return None

    def apply_turning_side_limit(self, out):
        if out.linear.x <= 0.0:
            return

        turning_side = self.turning_side_range(out.angular.z)
        if turning_side is None:
            return

        if turning_side <= self.turning_side_stop_distance:
            if (
                self.front_min_range is not None
                and self.front_min_range <= self.reverse_escape_distance
                and self.current_speed < self.unstick_speed_epsilon
            ):
                out.linear.x = -self.reverse_escape_speed
            else:
                out.linear.x = 0.0
            rospy.logwarn_throttle(
                1.0,
                "Turning-side clearance %.2fm is below stop distance %.2fm; suppressing forward motion",
                turning_side,
                self.turning_side_stop_distance,
            )
            return

        if turning_side >= self.turning_side_slow_distance:
            return

        normalized = (turning_side - self.turning_side_stop_distance) / (
            self.turning_side_slow_distance - self.turning_side_stop_distance
        )
        normalized = max(0.0, min(1.0, normalized))
        scale = self.turning_side_min_slow_scale + (
            1.0 - self.turning_side_min_slow_scale
        ) * (normalized ** self.slowdown_exponent)
        out.linear.x = min(out.linear.x, self.max_linear_x * scale)

    def apply_reverse_limit(self, out):
        if out.linear.x >= 0.0:
            return

        if self.rear_min_range is None:
            out.linear.x = 0.0
            rospy.logwarn_throttle(
                1.0,
                "No rear laser sector is available; blocking blind reverse motion",
            )
            return

        if self.rear_min_range <= self.rear_stop_distance:
            out.linear.x = 0.0
            rospy.logwarn_throttle(
                1.0,
                "Rear clearance %.2fm is below stop distance %.2fm; blocking reverse motion",
                self.rear_min_range,
                self.rear_stop_distance,
            )
            return

        if self.rear_min_range >= self.rear_slow_distance:
            return

        normalized = (self.rear_min_range - self.rear_stop_distance) / (
            self.rear_slow_distance - self.rear_stop_distance
        )
        normalized = max(0.0, min(1.0, normalized))
        scale = self.rear_min_slow_scale + (
            1.0 - self.rear_min_slow_scale
        ) * (normalized ** self.slowdown_exponent)
        out.linear.x = max(out.linear.x, -self.max_linear_x * scale)

    def cmd_cb(self, msg):
        now = rospy.Time.now()
        if self.external_motion_braked(now):
            self.stuck_since = None
            self.idle_since = None
            self.unstick_active_until = rospy.Time(0)
            self.idle_escape_active_until = rospy.Time(0)
            self.pub.publish(Twist())
            rospy.logwarn_throttle(
                1.0,
                "External motion brake is active; suppressing cmd_vel output",
            )
            return

        out = Twist()
        out.linear.x = max(min(msg.linear.x, self.max_linear_x), -self.max_linear_x)
        out.angular.z = max(min(msg.angular.z, self.max_angular_z), -self.max_angular_z)
        raw_forward = msg.linear.x > 0.02

        self.maybe_start_idle_escape(out, msg)

        if self.front_min_range is not None and out.linear.x > 0.0:
            if self.front_min_range <= self.stop_distance:
                if (
                    self.front_min_range <= self.reverse_escape_distance
                    and self.current_speed < self.unstick_speed_epsilon
                    and self.reverse_is_clear()
                ):
                    out.linear.x = -self.reverse_escape_speed
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
                scale = self.min_slow_scale + (1.0 - self.min_slow_scale) * (normalized ** self.slowdown_exponent)
                out.linear.x = min(out.linear.x, self.max_linear_x * scale)

            self.apply_turning_side_limit(out)
            self.maybe_start_unstick(out, raw_forward)
        elif out.linear.x <= 0.0:
            self.stuck_since = None

        self.apply_reverse_limit(out)
        self.pub.publish(out)


if __name__ == "__main__":
    rospy.init_node("twist_safety_limiter")
    TwistSafetyLimiter()
    rospy.spin()
