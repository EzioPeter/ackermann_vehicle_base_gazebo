#!/usr/bin/env python3

import sys
import time

import rospy
from controller_manager_msgs.srv import (
    ListControllers,
    LoadController,
    SwitchController,
)
from rosgraph_msgs.msg import Clock


class ControllerBootstrap:
    def __init__(self):
        rospy.init_node("controller_bootstrap")

        self.controller_manager_ns = rospy.get_param("~controller_manager_ns", "/controller_manager").rstrip("/")
        self.controllers = list(rospy.get_param("~controllers", []))
        self.clock_wait_timeout = float(rospy.get_param("~clock_wait_timeout", 20.0))
        self.start_timeout = float(rospy.get_param("~start_timeout", 30.0))
        self.retry_period = max(0.1, float(rospy.get_param("~retry_period", 0.5)))

        if not self.controllers:
            rospy.logerr("No controllers configured for bootstrap")
            raise RuntimeError("no controllers configured")

        self.list_srv_name = self.controller_manager_ns + "/list_controllers"
        self.load_srv_name = self.controller_manager_ns + "/load_controller"
        self.switch_srv_name = self.controller_manager_ns + "/switch_controller"

    def _wait_for_clock(self):
        if not rospy.get_param("/use_sim_time", False):
            return True

        deadline = None if self.clock_wait_timeout <= 0.0 else time.time() + self.clock_wait_timeout
        while not rospy.is_shutdown():
            try:
                rospy.wait_for_message("/clock", Clock, timeout=self.retry_period)
                rospy.loginfo("Received /clock; proceeding with controller bootstrap")
                return True
            except rospy.ROSException:
                if deadline is not None and time.time() >= deadline:
                    rospy.logerr("Timed out waiting for /clock")
                    return False

        return False

    def _wait_for_service(self, name):
        deadline = None if self.start_timeout <= 0.0 else time.time() + self.start_timeout
        while not rospy.is_shutdown():
            try:
                rospy.wait_for_service(name, timeout=self.retry_period)
                return True
            except rospy.ROSException:
                if deadline is not None and time.time() >= deadline:
                    rospy.logerr("Timed out waiting for service %s", name)
                    return False

        return False

    @staticmethod
    def _controller_states(list_srv):
        return {controller.name: controller.state for controller in list_srv().controller}

    def run(self):
        if not self._wait_for_clock():
            return 1

        for service_name in (self.list_srv_name, self.load_srv_name, self.switch_srv_name):
            if not self._wait_for_service(service_name):
                return 1

        list_srv = rospy.ServiceProxy(self.list_srv_name, ListControllers)
        load_srv = rospy.ServiceProxy(self.load_srv_name, LoadController)
        switch_srv = rospy.ServiceProxy(self.switch_srv_name, SwitchController)

        deadline = None if self.start_timeout <= 0.0 else time.time() + self.start_timeout

        while not rospy.is_shutdown():
            states = self._controller_states(list_srv)

            for controller in self.controllers:
                if controller not in states:
                    response = load_srv(controller)
                    if response.ok:
                        rospy.loginfo("Loaded controller %s", controller)
                    else:
                        rospy.logwarn("controller_manager rejected load for %s", controller)

            states = self._controller_states(list_srv)
            not_running = [controller for controller in self.controllers if states.get(controller) != "running"]

            if not not_running:
                rospy.loginfo("All requested controllers are running")
                return 0

            response = switch_srv(
                start_controllers=not_running,
                stop_controllers=[],
                strictness=2,
                start_asap=False,
                timeout=self.retry_period,
            )

            if response.ok:
                rospy.loginfo("Requested start for controllers: %s", ", ".join(not_running))
            else:
                rospy.logwarn("Failed to start controllers: %s", ", ".join(not_running))

            states = self._controller_states(list_srv)
            not_running = [controller for controller in self.controllers if states.get(controller) != "running"]
            if not not_running:
                rospy.loginfo("All requested controllers are running")
                return 0

            if deadline is not None and time.time() >= deadline:
                rospy.logerr("Timed out waiting for controllers to reach running state: %s", ", ".join(not_running))
                return 1

            time.sleep(self.retry_period)

        return 1


if __name__ == "__main__":
    try:
        sys.exit(ControllerBootstrap().run())
    except Exception as exc:
        rospy.logerr("controller_bootstrap failed: %s", exc)
        sys.exit(1)
