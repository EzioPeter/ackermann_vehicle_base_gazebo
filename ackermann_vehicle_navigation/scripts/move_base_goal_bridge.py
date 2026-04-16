#!/usr/bin/env python3

import math

import rospy
from actionlib_msgs.msg import GoalID
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import tf2_ros
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose


class MoveBaseGoalBridge(object):
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/move_base/current_goal")
        self.output_topic = rospy.get_param("~output_topic", "/ego_goal")
        self.output_frame = rospy.get_param("~output_frame", "odom")

        self.enable_frontier_timeout_switch = rospy.get_param("~enable_frontier_timeout_switch", True)
        self.frontier_timeout = float(rospy.get_param("~frontier_timeout", 12.0))
        self.goal_reached_tolerance = float(rospy.get_param("~goal_reached_tolerance", 0.6))
        self.goal_change_tolerance = float(rospy.get_param("~goal_change_tolerance", 0.35))

        self.enable_fallback_explore = rospy.get_param("~enable_fallback_explore", True)
        self.fallback_idle_timeout = float(rospy.get_param("~fallback_idle_timeout", 3.0))
        self.fallback_period = float(rospy.get_param("~fallback_period", 3.0))
        self.fallback_goal_distance = float(rospy.get_param("~fallback_goal_distance", 5.0))
        self.fallback_goal_min_distance = float(rospy.get_param("~fallback_goal_min_distance", 1.2))
        self.fallback_goal_z = float(rospy.get_param("~fallback_goal_z", 0.0))
        self.fallback_scan_topic = rospy.get_param("~fallback_scan_topic", "/ackermann_vehicle/scan")
        self.fallback_cmd_topic = rospy.get_param("~fallback_cmd_topic", "/cmd_vel")
        self.fallback_cmd_speed_epsilon = float(rospy.get_param("~fallback_cmd_speed_epsilon", 0.03))
        self.fallback_min_clearance = float(rospy.get_param("~fallback_min_clearance", 0.75))

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.out_pub = rospy.Publisher(self.output_topic, PoseStamped, queue_size=10)
        self.cancel_pub = rospy.Publisher("/move_base/cancel", GoalID, queue_size=2)
        self.sub = rospy.Subscriber(self.input_topic, PoseStamped, self.cb, queue_size=10)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_cb, queue_size=10)
        self.cmd_sub = rospy.Subscriber(self.fallback_cmd_topic, Twist, self.cmd_cb, queue_size=10)
        self.scan_sub = rospy.Subscriber(self.fallback_scan_topic, LaserScan, self.scan_cb, queue_size=5)

        self.last_goal = None
        self.last_goal_time = rospy.Time(0)
        self.last_goal_source = None
        self.last_fallback_time = rospy.Time(0)
        self.last_nonzero_cmd_time = rospy.Time(0)
        self.fallback_count = 0
        self.latest_odom = None
        self.latest_scan = None
        self.check_timer = rospy.Timer(rospy.Duration(1.0), self.check_timeout)

    def odom_cb(self, msg):
        self.latest_odom = msg

    def scan_cb(self, msg):
        self.latest_scan = msg

    def cmd_cb(self, msg):
        linear = math.hypot(msg.linear.x, msg.linear.y)
        angular = abs(msg.angular.z)
        if linear > self.fallback_cmd_speed_epsilon or angular > self.fallback_cmd_speed_epsilon:
            self.last_nonzero_cmd_time = rospy.Time.now()

    def _is_new_goal(self, goal_msg):
        if self.last_goal is None:
            return True

        dx = goal_msg.pose.position.x - self.last_goal.pose.position.x
        dy = goal_msg.pose.position.y - self.last_goal.pose.position.y
        return math.hypot(dx, dy) > self.goal_change_tolerance

    def _publish_goal(self, goal_msg, force=False, source="frontier"):
        is_new_goal = self._is_new_goal(goal_msg)
        if force or is_new_goal or self.last_goal is None:
            self.out_pub.publish(goal_msg)
            self.last_goal = goal_msg
            self.last_goal_time = rospy.Time.now()
            self.last_goal_source = source

    def cb(self, msg):
        if not self.output_frame or msg.header.frame_id == self.output_frame:
            out_msg = PoseStamped()
            out_msg.header = msg.header
            out_msg.pose = msg.pose
            if self.output_frame:
                out_msg.header.frame_id = self.output_frame
            self._publish_goal(out_msg, source="frontier")
            return

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.output_frame,
                msg.header.frame_id,
                rospy.Time(0),
                rospy.Duration(0.2),
            )
            transformed = do_transform_pose(msg, tf_msg)
            transformed.header.frame_id = self.output_frame
            transformed.header.stamp = rospy.Time.now()
            self._publish_goal(transformed, source="frontier")
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(
                2.0,
                "move_base_goal_bridge transform %s -> %s failed: %s",
                msg.header.frame_id,
                self.output_frame,
                str(exc),
            )

    def _robot_pose_in_output_frame(self):
        if self.latest_odom is None:
            return None

        robot = PoseStamped()
        robot.header = self.latest_odom.header
        robot.pose = self.latest_odom.pose.pose

        if robot.header.frame_id == self.output_frame:
            return robot

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.output_frame,
                robot.header.frame_id,
                rospy.Time(0),
                rospy.Duration(0.2),
            )
            transformed = do_transform_pose(robot, tf_msg)
            transformed.header.frame_id = self.output_frame
            transformed.header.stamp = rospy.Time.now()
            return transformed
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return None

    def _yaw_from_pose(self, pose):
        q = pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _angle_delta(self, a, b):
        return math.atan2(math.sin(a - b), math.cos(a - b))

    def _scan_clearance(self, target_angle):
        if self.latest_scan is None or not self.latest_scan.ranges:
            return self.fallback_goal_distance

        window = math.radians(14.0)
        best_clearance = self.latest_scan.range_max
        found = False
        angle = self.latest_scan.angle_min
        for value in self.latest_scan.ranges:
            if math.isfinite(value) and value > self.latest_scan.range_min:
                if abs(self._angle_delta(angle, target_angle)) <= window:
                    best_clearance = min(best_clearance, value)
                    found = True
            angle += self.latest_scan.angle_increment

        if not found:
            return self.latest_scan.range_max
        return best_clearance

    def _make_fallback_goal(self, robot_pose):
        robot_yaw = self._yaw_from_pose(robot_pose.pose)
        candidate_angles = [
            0.0,
            math.radians(25.0),
            math.radians(-25.0),
            math.radians(50.0),
            math.radians(-50.0),
            math.radians(80.0),
            math.radians(-80.0),
            math.radians(115.0),
            math.radians(-115.0),
            math.radians(150.0),
            math.radians(-150.0),
        ]
        phase = self.fallback_count % len(candidate_angles)

        best_angle = candidate_angles[0]
        best_clearance = 0.0
        best_score = -1e9
        for idx, angle in enumerate(candidate_angles):
            clearance = self._scan_clearance(angle)
            forward_bias = 0.45 * math.cos(angle)
            cycle_bias = 0.08 if idx == phase else 0.0
            score = clearance + forward_bias + cycle_bias
            if score > best_score:
                best_score = score
                best_angle = angle
                best_clearance = clearance

        travel_distance = min(self.fallback_goal_distance, max(self.fallback_goal_min_distance, best_clearance - 0.45))
        if best_clearance < self.fallback_min_clearance:
            travel_distance = self.fallback_goal_min_distance

        yaw = robot_yaw + best_angle
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = self.output_frame or robot_pose.header.frame_id
        goal.pose.position.x = robot_pose.pose.position.x + travel_distance * math.cos(yaw)
        goal.pose.position.y = robot_pose.pose.position.y + travel_distance * math.sin(yaw)
        goal.pose.position.z = self.fallback_goal_z
        goal.pose.orientation.z = math.sin(yaw * 0.5)
        goal.pose.orientation.w = math.cos(yaw * 0.5)
        return goal, best_angle, travel_distance, best_clearance

    def _maybe_request_frontier_switch(self, now):
        if not self.enable_frontier_timeout_switch:
            return
        if self.last_goal is None or self.latest_odom is None:
            return
        if self.last_goal_source != "frontier":
            return

        if (now - self.last_goal_time).to_sec() < self.frontier_timeout:
            return

        robot_pose = self._robot_pose_in_output_frame()
        if robot_pose is None:
            return

        dx = self.last_goal.pose.position.x - robot_pose.pose.position.x
        dy = self.last_goal.pose.position.y - robot_pose.pose.position.y
        dist_to_goal = math.hypot(dx, dy)
        if dist_to_goal <= self.goal_reached_tolerance:
            return

        self.cancel_pub.publish(GoalID())
        self.last_goal_time = now
        rospy.logwarn(
            "move_base_goal_bridge timeout %.1fs reached, requesting frontier switch via /move_base/cancel",
            self.frontier_timeout,
        )

    def _maybe_publish_fallback_goal(self, now):
        if not self.enable_fallback_explore or self.latest_odom is None:
            return

        if (now - self.last_fallback_time).to_sec() < self.fallback_period:
            return

        goal_age = float("inf") if self.last_goal is None else (now - self.last_goal_time).to_sec()
        moving_age = float("inf") if self.last_nonzero_cmd_time == rospy.Time(0) else (now - self.last_nonzero_cmd_time).to_sec()
        if goal_age < self.fallback_idle_timeout or moving_age < self.fallback_idle_timeout:
            return

        robot_pose = self._robot_pose_in_output_frame()
        if robot_pose is None:
            return

        goal, angle, distance, clearance = self._make_fallback_goal(robot_pose)
        self._publish_goal(goal, force=True, source="fallback")
        self.last_fallback_time = now
        self.fallback_count += 1
        rospy.logwarn(
            "No active motion for %.1fs; publishing fallback exploration goal %.2fm away at %.1fdeg (scan clearance %.2fm)",
            min(goal_age, moving_age),
            distance,
            math.degrees(angle),
            clearance,
        )

    def check_timeout(self, _event):
        now = rospy.Time.now()
        if now == rospy.Time(0):
            return

        self._maybe_request_frontier_switch(now)
        self._maybe_publish_fallback_goal(now)


if __name__ == "__main__":
    rospy.init_node("move_base_goal_bridge")
    MoveBaseGoalBridge()
    rospy.spin()
