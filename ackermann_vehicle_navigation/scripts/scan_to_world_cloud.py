#!/usr/bin/env python3

import rospy
import laser_geometry.laser_geometry as lg
from sensor_msgs.msg import LaserScan, PointCloud2
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
import tf.transformations as tft


class ScanToWorldCloud(object):
    def __init__(self):
        self.scan_topic = rospy.get_param("~scan_topic", "/ackermann_vehicle/scan")
        self.cloud_topic = rospy.get_param("~cloud_topic", "/cloud_registered")
        self.target_frame = rospy.get_param("~target_frame", "world")

        self.projector = lg.LaserProjection()
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pub = rospy.Publisher(self.cloud_topic, PointCloud2, queue_size=10)
        self.sub = rospy.Subscriber(self.scan_topic, LaserScan, self.scan_cb, queue_size=10)

    def scan_cb(self, scan_msg):
        cloud = self.projector.projectLaser(scan_msg)
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.target_frame,
                cloud.header.frame_id,
                rospy.Time(0),
                rospy.Duration(0.1),
            )

            q = tf_msg.transform.rotation
            t = tf_msg.transform.translation
            rot_mat = tft.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
            tx, ty, tz = t.x, t.y, t.z

            transformed_points = []
            for point in pc2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True):
                x, y, z = point
                px = rot_mat[0, 0] * x + rot_mat[0, 1] * y + rot_mat[0, 2] * z + tx
                py = rot_mat[1, 0] * x + rot_mat[1, 1] * y + rot_mat[1, 2] * z + ty
                pz = rot_mat[2, 0] * x + rot_mat[2, 1] * y + rot_mat[2, 2] * z + tz
                transformed_points.append((px, py, pz))

            header = cloud.header
            header.stamp = scan_msg.header.stamp
            header.frame_id = self.target_frame
            self.pub.publish(pc2.create_cloud_xyz32(header, transformed_points))
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "scan_to_world_cloud transform failed: %s", str(exc))


if __name__ == "__main__":
    rospy.init_node("scan_to_world_cloud")
    ScanToWorldCloud()
    rospy.spin()
