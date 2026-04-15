#!/usr/bin/env python3

import rospy
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
import tf2_ros
import geometry_msgs.msg


class GazeboModelStateToOdom(object):
    def __init__(self):
        self.model_name = rospy.get_param("~model_name", "ackermann_vehicle")
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.base_frame = rospy.get_param("~base_frame", "base_link")

        self.odom_pub = rospy.Publisher("/odom", Odometry, queue_size=20)
        self.odom_pub_ego = rospy.Publisher("/Odometry", Odometry, queue_size=20)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.last_stamp = None

        self.sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.cb, queue_size=10)

    def cb(self, msg):
        if self.model_name not in msg.name:
            return

        idx = msg.name.index(self.model_name)
        pose = msg.pose[idx]
        twist = msg.twist[idx]
        if hasattr(msg, "header"):
            now = msg.header.stamp
        else:
            now = rospy.Time.now()

        if self.last_stamp is not None and now == self.last_stamp:
            return
        self.last_stamp = now

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose = pose
        odom.twist.twist = twist

        if rospy.is_shutdown():
            return

        try:
            self.odom_pub.publish(odom)
            self.odom_pub_ego.publish(odom)
        except rospy.ROSException:
            return

        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = now
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = pose.position.x
        t.transform.translation.y = pose.position.y
        t.transform.translation.z = pose.position.z
        t.transform.rotation = pose.orientation
        try:
            self.tf_broadcaster.sendTransform(t)
        except rospy.ROSException:
            return


if __name__ == "__main__":
    rospy.init_node("gazebo_modelstate_to_odom")
    GazeboModelStateToOdom()
    rospy.spin()
