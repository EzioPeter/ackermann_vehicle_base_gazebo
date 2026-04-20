#!/usr/bin/env python3

import rospy
from actionlib_msgs.msg import GoalID
from geometry_msgs.msg import Twist


class ManualNavigationMode(object):
    def __init__(self):
        self.manual_mode_param = rospy.get_param("~manual_mode_param", "/teb_frontier/manual_mode")
        self.explore_pause_param = rospy.get_param("~explore_pause_param", "/explore/suspend_until")
        self.motion_brake_param = rospy.get_param("~motion_brake_param", "/teb_frontier/brake_until")
        self.pause_horizon = max(5.0, float(rospy.get_param("~pause_horizon", 30.0)))
        self.refresh_period = max(0.2, float(rospy.get_param("~refresh_period", 1.0)))
        self.cancel_retries = max(1, int(rospy.get_param("~cancel_retries", 3)))
        self.cancel_period = max(0.1, float(rospy.get_param("~cancel_period", 0.4)))
        self.publish_stop_on_switch = rospy.get_param("~publish_stop_on_switch", True)
        self.stop_cmd_topics = rospy.get_param("~stop_cmd_topics", ["/cmd_vel_nav_raw", "/cmd_vel"])
        if isinstance(self.stop_cmd_topics, str):
            self.stop_cmd_topics = [self.stop_cmd_topics]

        self.cancel_pub = rospy.Publisher("/move_base/cancel", GoalID, queue_size=2)
        self.stop_pubs = [
            rospy.Publisher(topic, Twist, queue_size=1)
            for topic in self.stop_cmd_topics
            if topic
        ]

        self._cancel_count = 0
        self._activate_manual_mode()
        self.refresh_timer = rospy.Timer(rospy.Duration(self.refresh_period), self._refresh)
        self.cancel_timer = rospy.Timer(rospy.Duration(self.cancel_period), self._startup_cancel)
        rospy.on_shutdown(self._deactivate_manual_mode)

    def _publish_stop(self):
        if not self.publish_stop_on_switch:
            return
        stop = Twist()
        for pub in self.stop_pubs:
            pub.publish(stop)

    def _publish_cancel(self):
        self.cancel_pub.publish(GoalID())

    def _activate_manual_mode(self):
        rospy.set_param(self.manual_mode_param, True)
        rospy.set_param(self.motion_brake_param, 0.0)
        rospy.set_param(
            self.explore_pause_param,
            (rospy.Time.now() + rospy.Duration(self.pause_horizon)).to_sec(),
        )
        self._publish_cancel()
        self._publish_stop()
        rospy.loginfo(
            "Manual navigation mode enabled; automatic exploration outputs are paused"
        )

    def _refresh(self, _event):
        rospy.set_param(self.manual_mode_param, True)
        rospy.set_param(self.motion_brake_param, 0.0)
        rospy.set_param(
            self.explore_pause_param,
            (rospy.Time.now() + rospy.Duration(self.pause_horizon)).to_sec(),
        )

    def _startup_cancel(self, _event):
        if self._cancel_count >= self.cancel_retries:
            self.cancel_timer.shutdown()
            return
        self._cancel_count += 1
        self._publish_cancel()
        self._publish_stop()

    def _deactivate_manual_mode(self):
        rospy.set_param(self.manual_mode_param, False)
        rospy.set_param(self.explore_pause_param, 0.0)
        rospy.set_param(self.motion_brake_param, 0.0)
        self._publish_cancel()
        self._publish_stop()
        rospy.loginfo(
            "Manual navigation mode disabled; automatic exploration takeover restored"
        )


if __name__ == "__main__":
    rospy.init_node("manual_navigation_mode")
    ManualNavigationMode()
    rospy.spin()
