#!/usr/bin/env python3

import math

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path


class OdomPathPublisher(object):
    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.path_topic = rospy.get_param("~path_topic", "/odom_visualization/path")
        self.min_step = float(rospy.get_param("~min_step", 0.05))
        self.max_poses = int(rospy.get_param("~max_poses", 2000))

        self.path = Path()
        self.path_pub = rospy.Publisher(self.path_topic, Path, queue_size=10)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=20)

    def odom_cb(self, msg):
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose

        if not self.path.header.frame_id:
            self.path.header.frame_id = pose.header.frame_id
        elif self.path.header.frame_id != pose.header.frame_id:
            self.path = Path()
            self.path.header.frame_id = pose.header.frame_id

        if self.path.poses:
            prev = self.path.poses[-1].pose.position
            curr = pose.pose.position
            if math.hypot(curr.x - prev.x, curr.y - prev.y) < self.min_step:
                self.path.header.stamp = pose.header.stamp
                self.path_pub.publish(self.path)
                return

        self.path.poses.append(pose)
        if len(self.path.poses) > self.max_poses:
            self.path.poses = self.path.poses[-self.max_poses :]
        self.path.header.stamp = pose.header.stamp
        self.path_pub.publish(self.path)


if __name__ == "__main__":
    rospy.init_node("odom_path_publisher")
    OdomPathPublisher()
    rospy.spin()
