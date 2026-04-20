#!/usr/bin/env python3

import math

import rospy
import tf2_ros
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion


class GazeboOdomBridge:
    def __init__(self):
        self.model_name = rospy.get_param("~model_name", "ackermann_vehicle")
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.secondary_odom_topic = rospy.get_param("~secondary_odom_topic", "")
        self.publish_tf = rospy.get_param("~publish_tf", True)
        self.publish_rate = max(rospy.get_param("~publish_rate", 30.0), 1.0)
        self.planar_mode = rospy.get_param("~planar_mode", True)
        self.rollover_guard_enable = rospy.get_param("~rollover_guard_enable", True)
        self.rollover_max_tilt_deg = max(0.0, float(rospy.get_param("~rollover_max_tilt_deg", 18.0)))
        self.rollover_stable_tilt_deg = max(
            0.0,
            min(self.rollover_max_tilt_deg, float(rospy.get_param("~rollover_stable_tilt_deg", 8.0))),
        )
        self.rollover_hold_time = max(0.0, float(rospy.get_param("~rollover_hold_time", 0.35)))
        self.rollover_reset_cooldown = max(
            0.1, float(rospy.get_param("~rollover_reset_cooldown", 2.0))
        )
        self.suppress_odom_when_tilted = rospy.get_param("~suppress_odom_when_tilted", True)
        self.reset_reference_frame = rospy.get_param("~reset_reference_frame", "world")
        self.reset_pose_x = float(rospy.get_param("~reset_pose_x", 0.0))
        self.reset_pose_y = float(rospy.get_param("~reset_pose_y", 0.0))
        self.reset_pose_z = float(rospy.get_param("~reset_pose_z", 0.2))
        self.reset_pose_yaw = float(rospy.get_param("~reset_pose_yaw", 0.0))
        self.set_model_state_service = rospy.get_param("~set_model_state_service", "/gazebo/set_model_state")

        self._last_pub_time = rospy.Time(0)
        self._min_period = rospy.Duration.from_sec(1.0 / self.publish_rate)
        self._warned = False
        self._last_stable_pose = None
        self._tilt_started_at = None
        self._last_reset_time = rospy.Time(0)
        self._last_rollover_warn_time = rospy.Time(0)

        self.odom_pub = rospy.Publisher(self.odom_topic, Odometry, queue_size=10)
        self.secondary_odom_pub = None
        if self.secondary_odom_topic:
            self.secondary_odom_pub = rospy.Publisher(self.secondary_odom_topic, Odometry, queue_size=10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster() if self.publish_tf else None
        self.set_model_state = None
        if self.rollover_guard_enable:
            try:
                rospy.wait_for_service(self.set_model_state_service, timeout=5.0)
                self.set_model_state = rospy.ServiceProxy(self.set_model_state_service, SetModelState)
            except rospy.ROSException:
                rospy.logwarn(
                    "Failed to connect to %s; disabling rollover guard",
                    self.set_model_state_service,
                )
                self.rollover_guard_enable = False

        rospy.Subscriber("/gazebo/model_states", ModelStates, self._model_states_cb, queue_size=1)

    @staticmethod
    def _yaw_to_quaternion(yaw):
        return math.sin(0.5 * yaw), math.cos(0.5 * yaw)

    @staticmethod
    def _tilt_deg_from_orientation(orientation):
        body_z_world_z = 1.0 - 2.0 * (
            orientation.x * orientation.x + orientation.y * orientation.y
        )
        body_z_world_z = max(-1.0, min(1.0, body_z_world_z))
        return math.degrees(math.acos(body_z_world_z))

    def _update_last_stable_pose(self, pose, yaw):
        self._last_stable_pose = {
            "x": pose.position.x,
            "y": pose.position.y,
            "z": pose.position.z,
            "yaw": yaw,
        }

    def _build_reset_state(self):
        target = self._last_stable_pose or {
            "x": self.reset_pose_x,
            "y": self.reset_pose_y,
            "z": self.reset_pose_z,
            "yaw": self.reset_pose_yaw,
        }

        quat_z, quat_w = self._yaw_to_quaternion(target["yaw"])

        state = ModelState()
        state.model_name = self.model_name
        state.reference_frame = self.reset_reference_frame
        state.pose.position.x = target["x"]
        state.pose.position.y = target["y"]
        state.pose.position.z = target["z"]
        state.pose.orientation.x = 0.0
        state.pose.orientation.y = 0.0
        state.pose.orientation.z = quat_z
        state.pose.orientation.w = quat_w
        state.twist.linear.x = 0.0
        state.twist.linear.y = 0.0
        state.twist.linear.z = 0.0
        state.twist.angular.x = 0.0
        state.twist.angular.y = 0.0
        state.twist.angular.z = 0.0
        return state

    def _maybe_guard_rollover(self, now, pose, yaw):
        if not self.rollover_guard_enable:
            return False

        tilt_deg = self._tilt_deg_from_orientation(pose.orientation)
        if tilt_deg <= self.rollover_stable_tilt_deg:
            self._update_last_stable_pose(pose, yaw)
            self._tilt_started_at = None
            return False

        if tilt_deg < self.rollover_max_tilt_deg:
            self._tilt_started_at = None
            return False

        if self._tilt_started_at is None:
            self._tilt_started_at = now

        if (
            self._last_rollover_warn_time == rospy.Time(0)
            or (now - self._last_rollover_warn_time).to_sec() >= 2.0
        ):
            rospy.logwarn(
                "Detected vehicle tilt %.1f deg (threshold %.1f deg); suppressing planar odom until reset",
                tilt_deg,
                self.rollover_max_tilt_deg,
            )
            self._last_rollover_warn_time = now

        if (now - self._tilt_started_at).to_sec() < self.rollover_hold_time:
            return self.suppress_odom_when_tilted

        if (now - self._last_reset_time).to_sec() < self.rollover_reset_cooldown:
            return self.suppress_odom_when_tilted

        if self.set_model_state is None:
            return self.suppress_odom_when_tilted

        state = self._build_reset_state()
        target_yaw = self._last_stable_pose["yaw"] if self._last_stable_pose is not None else self.reset_pose_yaw
        try:
            response = self.set_model_state(state)
        except rospy.ServiceException as exc:
            rospy.logwarn("Failed to reset %s after rollover: %s", self.model_name, exc)
            return self.suppress_odom_when_tilted

        if not response.success:
            rospy.logwarn("Gazebo rejected rollover reset for %s: %s", self.model_name, response.status_message)
            return self.suppress_odom_when_tilted

        self._last_reset_time = now
        self._tilt_started_at = None
        self._last_pub_time = rospy.Time(0)
        self._last_stable_pose = {
            "x": state.pose.position.x,
            "y": state.pose.position.y,
            "z": state.pose.position.z,
            "yaw": target_yaw,
        }
        rospy.logwarn(
            "Reset %s after rollover to x=%.2f y=%.2f z=%.2f yaw=%.2f",
            self.model_name,
            state.pose.position.x,
            state.pose.position.y,
            state.pose.position.z,
            target_yaw,
        )
        return True

    def _model_states_cb(self, msg):
        now = rospy.Time.now()
        try:
            index = msg.name.index(self.model_name)
        except ValueError:
            if not self._warned:
                rospy.logwarn("Model '%s' is not present in /gazebo/model_states yet", self.model_name)
                self._warned = True
            return

        self._warned = False

        pose = msg.pose[index]
        twist = msg.twist[index]

        _, _, yaw = euler_from_quaternion(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]
        )

        if self._maybe_guard_rollover(now, pose, yaw):
            return

        if now - self._last_pub_time < self._min_period:
            return

        self._last_pub_time = now

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        linear_x = cos_yaw * twist.linear.x + sin_yaw * twist.linear.y
        linear_y = -sin_yaw * twist.linear.x + cos_yaw * twist.linear.y

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = pose.position.x
        odom.pose.pose.position.y = pose.position.y
        odom.pose.pose.position.z = 0.0 if self.planar_mode else pose.position.z
        odom.pose.pose.orientation.x = 0.0 if self.planar_mode else pose.orientation.x
        odom.pose.pose.orientation.y = 0.0 if self.planar_mode else pose.orientation.y
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = self._yaw_to_quaternion(yaw)
        odom.twist.twist.linear.x = linear_x
        odom.twist.twist.linear.y = linear_y
        odom.twist.twist.linear.z = 0.0 if self.planar_mode else twist.linear.z
        odom.twist.twist.angular.x = 0.0 if self.planar_mode else twist.angular.x
        odom.twist.twist.angular.y = 0.0 if self.planar_mode else twist.angular.y
        odom.twist.twist.angular.z = twist.angular.z
        if rospy.is_shutdown():
            return

        try:
            self.odom_pub.publish(odom)
            if self.secondary_odom_pub is not None:
                self.secondary_odom_pub.publish(odom)
        except rospy.ROSException:
            return

        if self.tf_broadcaster is None:
            return

        transform = TransformStamped()
        transform.header.stamp = now
        transform.header.frame_id = self.odom_frame
        transform.child_frame_id = self.base_frame
        transform.transform.translation.x = pose.position.x
        transform.transform.translation.y = pose.position.y
        transform.transform.translation.z = 0.0 if self.planar_mode else pose.position.z
        transform.transform.rotation.x = 0.0 if self.planar_mode else pose.orientation.x
        transform.transform.rotation.y = 0.0 if self.planar_mode else pose.orientation.y
        transform.transform.rotation.z, transform.transform.rotation.w = self._yaw_to_quaternion(yaw)
        try:
            self.tf_broadcaster.sendTransform(transform)
        except rospy.ROSException:
            return


if __name__ == "__main__":
    rospy.init_node("gazebo_odom_bridge")
    GazeboOdomBridge()
    rospy.spin()
