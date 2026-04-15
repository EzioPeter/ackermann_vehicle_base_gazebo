#!/usr/bin/env python3

import math

import rospy
from actionlib_msgs.msg import GoalID
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
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

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.out_pub = rospy.Publisher(self.output_topic, PoseStamped, queue_size=10)
        self.cancel_pub = rospy.Publisher("/move_base/cancel", GoalID, queue_size=2)
        self.sub = rospy.Subscriber(self.input_topic, PoseStamped, self.cb, queue_size=10)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_cb, queue_size=10)

        self.last_goal = None
        self.last_goal_time = rospy.Time(0)
        self.latest_odom = None
        self.check_timer = rospy.Timer(rospy.Duration(1.0), self.check_timeout)

    def odom_cb(self, msg):
        self.latest_odom = msg

    def _is_new_goal(self, goal_msg):
        if self.last_goal is None:
            return True

        dx = goal_msg.pose.position.x - self.last_goal.pose.position.x
        dy = goal_msg.pose.position.y - self.last_goal.pose.position.y
        return math.hypot(dx, dy) > self.goal_change_tolerance

    def _publish_goal(self, goal_msg):
        is_new_goal = self._is_new_goal(goal_msg)
        if is_new_goal or self.last_goal is None:
            self.out_pub.publish(goal_msg)
            self.last_goal = goal_msg
            self.last_goal_time = rospy.Time.now()

    def cb(self, msg):
        if not self.output_frame or msg.header.frame_id == self.output_frame:
            out_msg = PoseStamped()
            out_msg.header = msg.header
            out_msg.pose = msg.pose
            if self.output_frame:
                out_msg.header.frame_id = self.output_frame
            self._publish_goal(out_msg)
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
            self._publish_goal(transformed)
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

    def check_timeout(self, _event):
        if not self.enable_frontier_timeout_switch:
            return
        if self.last_goal is None or self.latest_odom is None:
            return

        now = rospy.Time.now()
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

        # Request a frontier switch by preempting current move_base goal.
        # explore_lite will then re-plan and pick another frontier target.
        self.cancel_pub.publish(GoalID())
        self.last_goal_time = now
        rospy.logwarn(
            "move_base_goal_bridge timeout %.1fs reached, requesting frontier switch via /move_base/cancel",
            self.frontier_timeout,
        )


if __name__ == "__main__":
    rospy.init_node("move_base_goal_bridge")
    MoveBaseGoalBridge()
    rospy.spin()
