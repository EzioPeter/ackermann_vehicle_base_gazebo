#!/usr/bin/env python3

import math

import rospy
import tf2_ros
from actionlib_msgs.msg import GoalID, GoalStatusArray
from geometry_msgs.msg import PoseStamped, Twist
from move_base_msgs.msg import MoveBaseActionGoal
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import LaserScan
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose


class FrontierGoalSupervisor(object):
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/move_base/current_goal")
        self.fallback_goal_topic = rospy.get_param("~fallback_goal_topic", "/move_base_simple/goal")
        self.output_frame = rospy.get_param("~output_frame", "map")

        self.enable_frontier_timeout_switch = rospy.get_param("~enable_frontier_timeout_switch", True)
        self.frontier_timeout = float(rospy.get_param("~frontier_timeout", 12.0))
        self.goal_reached_tolerance = float(rospy.get_param("~goal_reached_tolerance", 0.6))
        self.goal_change_tolerance = float(rospy.get_param("~goal_change_tolerance", 0.35))
        self.frontier_progress_epsilon = float(rospy.get_param("~frontier_progress_epsilon", 0.35))
        self.frontier_motion_epsilon = float(rospy.get_param("~frontier_motion_epsilon", 0.25))
        self.force_escape_enable = rospy.get_param("~force_escape_enable", True)
        self.force_escape_hold = float(rospy.get_param("~force_escape_hold", 10.0))
        self.force_escape_cooldown = float(rospy.get_param("~force_escape_cooldown", 8.0))
        self.force_escape_on_abort = rospy.get_param("~force_escape_on_abort", True)
        self.explore_pause_param = rospy.get_param("~explore_pause_param", "/explore/suspend_until")
        self.motion_brake_param = rospy.get_param("~motion_brake_param", "/teb_frontier/brake_until")
        self.manual_mode_param = rospy.get_param("~manual_mode_param", "/teb_frontier/manual_mode")

        self.enable_fallback_explore = rospy.get_param("~enable_fallback_explore", True)
        self.fallback_idle_timeout = float(rospy.get_param("~fallback_idle_timeout", 3.0))
        self.fallback_period = float(rospy.get_param("~fallback_period", 3.0))
        self.fallback_startup_grace = float(rospy.get_param("~fallback_startup_grace", 10.0))
        self.fallback_goal_distance = float(rospy.get_param("~fallback_goal_distance", 5.0))
        self.fallback_goal_min_distance = float(rospy.get_param("~fallback_goal_min_distance", 1.2))
        self.fallback_goal_z = float(rospy.get_param("~fallback_goal_z", 0.0))
        self.local_plan_escape_goal_distance = float(rospy.get_param("~local_plan_escape_goal_distance", 1.4))
        self.fallback_scan_topic = rospy.get_param("~fallback_scan_topic", "/ackermann_vehicle/scan")
        self.fallback_cmd_topic = rospy.get_param("~fallback_cmd_topic", "/cmd_vel")
        self.fallback_cmd_speed_epsilon = float(rospy.get_param("~fallback_cmd_speed_epsilon", 0.03))
        self.fallback_min_clearance = float(rospy.get_param("~fallback_min_clearance", 0.75))
        self.local_plan_guard_enable = rospy.get_param("~local_plan_guard_enable", True)
        self.local_plan_topic = rospy.get_param("~local_plan_topic", "/move_base/TebLocalPlannerROS/local_plan")
        self.local_plan_visual_topic = rospy.get_param("~local_plan_visual_topic", "/teb_frontier/local_plan_safe")
        self.local_plan_guard_lookahead = float(rospy.get_param("~local_plan_guard_lookahead", 1.3))
        self.local_plan_guard_margin = float(rospy.get_param("~local_plan_guard_margin", 0.35))
        self.local_plan_guard_window_deg = float(rospy.get_param("~local_plan_guard_window_deg", 12.0))
        self.local_plan_guard_min_distance = float(rospy.get_param("~local_plan_guard_min_distance", 0.25))
        self.local_plan_guard_cooldown = float(rospy.get_param("~local_plan_guard_cooldown", 1.0))
        self.local_plan_guard_immediate = rospy.get_param("~local_plan_guard_immediate", True)
        self.local_plan_guard_clear_visual_on_block = rospy.get_param("~local_plan_guard_clear_visual_on_block", True)
        self.local_plan_guard_segment_step = float(rospy.get_param("~local_plan_guard_segment_step", 0.05))
        self.local_plan_guard_brake_duration = float(rospy.get_param("~local_plan_guard_brake_duration", 1.0))
        self.local_plan_guard_map_start_ignore = float(
            rospy.get_param("~local_plan_guard_map_start_ignore", 0.55)
        )
        self.local_plan_guard_map_topic = rospy.get_param("~local_plan_guard_map_topic", "/map")
        self.local_plan_guard_map_clearance = float(rospy.get_param("~local_plan_guard_map_clearance", 0.42))
        self.local_plan_guard_map_occupied_threshold = int(
            rospy.get_param("~local_plan_guard_map_occupied_threshold", 50)
        )
        self.local_plan_guard_unknown_is_blocked = rospy.get_param("~local_plan_guard_unknown_is_blocked", False)
        self.local_plan_guard_blocked_angle_ttl = float(
            rospy.get_param("~local_plan_guard_blocked_angle_ttl", 18.0)
        )
        self.local_plan_guard_blocked_angle_window = math.radians(
            float(rospy.get_param("~local_plan_guard_blocked_angle_window_deg", 35.0))
        )
        self.local_plan_guard_max_blocked_angles = int(
            rospy.get_param("~local_plan_guard_max_blocked_angles", 8)
        )
        self.stop_cmd_topics = rospy.get_param("~stop_cmd_topics", ["/cmd_vel_nav_raw", "/cmd_vel"])
        if isinstance(self.stop_cmd_topics, str):
            self.stop_cmd_topics = [self.stop_cmd_topics]

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.fallback_pub = rospy.Publisher(self.fallback_goal_topic, PoseStamped, queue_size=10)
        self.cancel_pub = rospy.Publisher("/move_base/cancel", GoalID, queue_size=2)
        self.local_plan_visual_pub = rospy.Publisher(self.local_plan_visual_topic, Path, queue_size=3)
        self.stop_pubs = [
            rospy.Publisher(topic, Twist, queue_size=1)
            for topic in self.stop_cmd_topics
            if topic
        ]
        rospy.Subscriber(self.input_topic, PoseStamped, self.goal_cb, queue_size=10)
        rospy.Subscriber("/move_base/goal", MoveBaseActionGoal, self.action_goal_cb, queue_size=10)
        rospy.Subscriber("/odom", Odometry, self.odom_cb, queue_size=10)
        rospy.Subscriber(self.fallback_cmd_topic, Twist, self.cmd_cb, queue_size=10)
        rospy.Subscriber(self.fallback_scan_topic, LaserScan, self.scan_cb, queue_size=5)
        rospy.Subscriber("/move_base/status", GoalStatusArray, self.status_cb, queue_size=10)
        rospy.Subscriber(self.local_plan_topic, Path, self.local_plan_cb, queue_size=3)
        rospy.Subscriber(self.local_plan_guard_map_topic, OccupancyGrid, self.map_cb, queue_size=1)

        self.last_goal = None
        self.last_goal_time = rospy.Time(0)
        self.last_goal_source = None
        self.last_fallback_time = rospy.Time(0)
        self.last_nonzero_cmd_time = rospy.Time(0)
        self.last_force_escape_time = rospy.Time(0)
        self.startup_time = rospy.Time(0)
        self.last_progress_time = rospy.Time(0)
        self.best_goal_distance = None
        self.last_progress_pose = None
        self.pending_abort_goal_id = None
        self.last_aborted_goal_id = None
        self.fallback_count = 0
        self.latest_odom = None
        self.latest_scan = None
        self.latest_local_plan = None
        self.latest_map = None
        self.last_local_plan_guard_time = rospy.Time(0)
        self.blocked_escape_directions = []

        self.check_timer = rospy.Timer(rospy.Duration(1.0), self.check_timeout)

    def _manual_mode_active(self):
        try:
            return bool(rospy.get_param(self.manual_mode_param, False))
        except (TypeError, ValueError):
            return False

    def odom_cb(self, msg):
        self.latest_odom = msg

    def scan_cb(self, msg):
        self.latest_scan = msg

    def local_plan_cb(self, msg):
        self.latest_local_plan = msg
        if self._manual_mode_active():
            self.local_plan_visual_pub.publish(msg)
            return
        blocked = None
        if self.local_plan_guard_enable:
            robot_odom_pose = self._robot_pose_in_odom_frame()
            if robot_odom_pose is not None:
                blocked = self._local_plan_blocked_by_scan(robot_odom_pose)

        if blocked is None:
            self.local_plan_visual_pub.publish(msg)
            return

        self._publish_empty_local_plan(msg.header)
        if self.local_plan_guard_immediate:
            self._handle_blocked_local_plan(rospy.Time.now(), blocked)

    def map_cb(self, msg):
        self.latest_map = msg

    def action_goal_cb(self, msg):
        goal = msg.goal.target_pose
        if goal.header.stamp == rospy.Time(0):
            goal.header.stamp = rospy.Time.now()
        self.goal_cb(goal)

    def status_cb(self, msg):
        if not self.force_escape_on_abort or not msg.status_list:
            return

        latest_status = max(
            msg.status_list,
            key=lambda item: (
                item.goal_id.stamp.secs,
                item.goal_id.stamp.nsecs,
                item.goal_id.id,
            ),
        )

        if latest_status.status != 4:
            return

        goal_id = latest_status.goal_id.id
        if goal_id == self.last_aborted_goal_id:
            return

        self.last_aborted_goal_id = goal_id
        self.pending_abort_goal_id = goal_id

    def cmd_cb(self, msg):
        linear = math.hypot(msg.linear.x, msg.linear.y)
        angular = abs(msg.angular.z)
        if linear > self.fallback_cmd_speed_epsilon or angular > self.fallback_cmd_speed_epsilon:
            self.last_nonzero_cmd_time = rospy.Time.now()

    def _distance_between_goals(self, first, second):
        if first is None or second is None:
            return float("inf")
        dx = first.pose.position.x - second.pose.position.x
        dy = first.pose.position.y - second.pose.position.y
        return math.hypot(dx, dy)

    def _reset_goal_progress(self):
        self.last_progress_time = rospy.Time.now()
        self.best_goal_distance = None
        self.last_progress_pose = None

    def _is_new_goal(self, goal_msg):
        if self.last_goal is None:
            return True
        if goal_msg.header.frame_id != self.last_goal.header.frame_id:
            return True
        return self._distance_between_goals(goal_msg, self.last_goal) > self.goal_change_tolerance

    def _transform_pose(self, pose_msg, target_frame):
        out_msg = PoseStamped()
        out_msg.header = pose_msg.header
        out_msg.pose = pose_msg.pose
        if not target_frame or pose_msg.header.frame_id == target_frame:
            if target_frame:
                out_msg.header.frame_id = target_frame
            return out_msg

        tf_msg = self.tf_buffer.lookup_transform(
            target_frame,
            pose_msg.header.frame_id,
            rospy.Time(0),
            rospy.Duration(0.2),
        )
        transformed = do_transform_pose(out_msg, tf_msg)
        transformed.header.frame_id = target_frame
        transformed.header.stamp = rospy.Time.now()
        return transformed

    def goal_cb(self, msg):
        manual_mode = self._manual_mode_active()
        try:
            goal_msg = self._transform_pose(msg, self.output_frame)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(
                2.0,
                "frontier_goal_supervisor transform %s -> %s failed: %s",
                msg.header.frame_id,
                self.output_frame,
                str(exc),
            )
            return

        if (
            self.last_goal is not None
            and self.last_goal_source == "fallback"
            and self._distance_between_goals(goal_msg, self.last_goal) <= self.goal_change_tolerance
        ):
            self.last_goal_time = rospy.Time.now()
            return

        if self._is_new_goal(goal_msg):
            self.last_goal = goal_msg
            self.last_goal_time = rospy.Time.now()
            self.last_goal_source = "manual" if manual_mode else "frontier"
            self._reset_goal_progress()

    def _robot_pose_in_output_frame(self):
        if self.latest_odom is None:
            return None

        robot = PoseStamped()
        robot.header = self.latest_odom.header
        robot.pose = self.latest_odom.pose.pose
        try:
            return self._transform_pose(robot, self.output_frame)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return None

    def _yaw_from_pose(self, pose):
        q = pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _pose_xy_yaw(self, pose):
        return pose.position.x, pose.position.y, self._yaw_from_pose(pose)

    def _angle_delta(self, first, second):
        return math.atan2(math.sin(first - second), math.cos(first - second))

    def _scan_clearance(self, target_angle, window=None):
        if self.latest_scan is None or not self.latest_scan.ranges:
            return self.fallback_goal_distance

        if window is None:
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

    def _prune_blocked_escape_directions(self, now=None):
        if now is None:
            now = rospy.Time.now()
        self.blocked_escape_directions = [
            item for item in self.blocked_escape_directions
            if item[1] > now
        ]
        max_items = max(0, self.local_plan_guard_max_blocked_angles)
        if max_items and len(self.blocked_escape_directions) > max_items:
            self.blocked_escape_directions = self.blocked_escape_directions[-max_items:]

    def _remember_blocked_escape_direction(self, robot_pose, bearing_deg):
        ttl = max(0.0, self.local_plan_guard_blocked_angle_ttl)
        if ttl <= 0.0:
            return
        robot_yaw = self._yaw_from_pose(robot_pose.pose)
        absolute_yaw = robot_yaw + math.radians(bearing_deg)
        expires = rospy.Time.now() + rospy.Duration(ttl)
        self.blocked_escape_directions.append((absolute_yaw, expires))
        self._prune_blocked_escape_directions()

    def _escape_direction_recently_blocked(self, absolute_yaw, now=None):
        if self.local_plan_guard_blocked_angle_window <= 0.0:
            return False
        self._prune_blocked_escape_directions(now)
        for blocked_yaw, _expires in self.blocked_escape_directions:
            if abs(self._angle_delta(absolute_yaw, blocked_yaw)) <= self.local_plan_guard_blocked_angle_window:
                return True
        return False

    def _max_map_safe_distance(self, x, y, yaw, max_distance):
        if self.latest_map is None:
            return max_distance

        step = max(0.05, self.local_plan_guard_segment_step)
        ignore_start = max(0.25, self.local_plan_guard_min_distance, self.local_plan_guard_map_start_ignore)
        distance = step
        safe_distance = 0.0
        while distance <= max_distance:
            sample_x = x + distance * math.cos(yaw)
            sample_y = y + distance * math.sin(yaw)
            if distance >= ignore_start and self._map_near_blocked(sample_x, sample_y):
                return max(0.0, safe_distance - 0.15)
            safe_distance = distance
            distance += step
        return max_distance

    def _make_fallback_goal(self, robot_pose, max_distance=None):
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
            math.radians(180.0),
        ]
        phase = self.fallback_count % len(candidate_angles)
        if max_distance is None:
            max_distance = self.fallback_goal_distance
        max_distance = max(self.fallback_goal_min_distance, min(self.fallback_goal_distance, max_distance))

        best_angle = None
        best_clearance = 0.0
        best_distance = 0.0
        best_score = -1e9
        blocked_best_angle = None
        blocked_best_clearance = 0.0
        blocked_best_distance = 0.0
        blocked_best_score = -1e9
        now = rospy.Time.now()
        for index, angle in enumerate(candidate_angles):
            clearance = self._scan_clearance(angle)
            if clearance < self.fallback_min_clearance:
                continue
            yaw = robot_yaw + angle
            travel_distance = min(max_distance, max(self.fallback_goal_min_distance, clearance - 0.45))
            travel_distance = self._max_map_safe_distance(
                robot_pose.pose.position.x,
                robot_pose.pose.position.y,
                yaw,
                travel_distance,
            )
            if travel_distance < self.fallback_goal_min_distance:
                continue

            forward_bias = 0.45 * math.cos(angle)
            cycle_bias = 0.08 if index == phase else 0.0
            score = clearance + travel_distance + forward_bias + cycle_bias
            if self._escape_direction_recently_blocked(yaw, now):
                if score > blocked_best_score:
                    blocked_best_score = score
                    blocked_best_angle = angle
                    blocked_best_clearance = clearance
                    blocked_best_distance = travel_distance
                continue
            if score > best_score:
                best_score = score
                best_angle = angle
                best_clearance = clearance
                best_distance = travel_distance

        if best_angle is None:
            return None, 0.0, 0.0, best_clearance

        yaw = robot_yaw + best_angle
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = self.output_frame or robot_pose.header.frame_id
        goal.pose.position.x = robot_pose.pose.position.x + best_distance * math.cos(yaw)
        goal.pose.position.y = robot_pose.pose.position.y + best_distance * math.sin(yaw)
        goal.pose.position.z = self.fallback_goal_z
        goal.pose.orientation.z = math.sin(yaw * 0.5)
        goal.pose.orientation.w = math.cos(yaw * 0.5)
        return goal, best_angle, best_distance, best_clearance

    def _distance_to_goal(self, robot_pose, goal_pose):
        dx = goal_pose.pose.position.x - robot_pose.pose.position.x
        dy = goal_pose.pose.position.y - robot_pose.pose.position.y
        return math.hypot(dx, dy)

    def _goal_bearing_deg(self, robot_pose, goal_pose):
        robot_yaw = self._yaw_from_pose(robot_pose.pose)
        dx = goal_pose.pose.position.x - robot_pose.pose.position.x
        dy = goal_pose.pose.position.y - robot_pose.pose.position.y
        goal_yaw = math.atan2(dy, dx)
        return math.degrees(self._angle_delta(goal_yaw, robot_yaw))

    def _set_explore_pause(self, now, duration):
        pause_until = now + rospy.Duration(max(0.0, duration))
        rospy.set_param(self.explore_pause_param, pause_until.to_sec())
        return pause_until

    def _set_motion_brake(self, now, duration):
        brake_until = now + rospy.Duration(max(0.0, duration))
        rospy.set_param(self.motion_brake_param, brake_until.to_sec())
        return brake_until

    def _motion_brake_active(self, now):
        try:
            brake_until = float(rospy.get_param(self.motion_brake_param, 0.0))
        except (TypeError, ValueError):
            return False
        return brake_until > 0.0 and now < rospy.Time(brake_until)

    def _publish_empty_local_plan(self, header=None):
        if not self.local_plan_guard_clear_visual_on_block:
            return
        empty_plan = Path()
        if header is not None:
            empty_plan.header = header
        empty_plan.header.stamp = rospy.Time.now()
        if not empty_plan.header.frame_id:
            empty_plan.header.frame_id = self.output_frame
        self.local_plan_visual_pub.publish(empty_plan)

    def _publish_stop_cmd(self):
        stop = Twist()
        for pub in self.stop_pubs:
            pub.publish(stop)

    def _trigger_forced_escape(self, now, robot_pose, reason, dist_to_goal=None):
        if not self.force_escape_enable:
            return False
        if (
            self.last_force_escape_time != rospy.Time(0)
            and (now - self.last_force_escape_time).to_sec() < self.force_escape_cooldown
        ):
            return False

        pause_until = self._set_explore_pause(now, self.force_escape_hold)
        self.cancel_pub.publish(GoalID())
        if reason == "local_plan_hits_obstacle":
            self._set_motion_brake(now, self.local_plan_guard_brake_duration)
        max_escape_distance = None
        if reason == "local_plan_hits_obstacle":
            max_escape_distance = self.local_plan_escape_goal_distance
        goal_msg, angle, distance, clearance = self._make_fallback_goal(robot_pose, max_escape_distance)
        if goal_msg is None:
            self.last_force_escape_time = now
            self.pending_abort_goal_id = None
            rospy.logwarn(
                "frontier_goal_supervisor forcing escape recovery (%s); pausing frontier planning for %.1fs, "
                "but no map-safe fallback goal was available",
                reason,
                max(0.0, (pause_until - now).to_sec()),
            )
            return True

        self._publish_fallback_goal(goal_msg)
        self.last_force_escape_time = now
        self.pending_abort_goal_id = None

        if dist_to_goal is None:
            dist_text = "unknown"
        else:
            dist_text = "%.2f" % dist_to_goal

        rospy.logwarn(
            "frontier_goal_supervisor forcing escape recovery (%s); pausing frontier planning for %.1fs, "
            "goal distance %s, fallback %.2fm at %.1fdeg (clearance %.2fm)",
            reason,
            max(0.0, (pause_until - now).to_sec()),
            dist_text,
            distance,
            math.degrees(angle),
            clearance,
        )
        return True

    def _update_progress(self, now, robot_pose, dist_to_goal):
        progress_detected = False

        if self.best_goal_distance is None:
            self.best_goal_distance = dist_to_goal
            progress_detected = True
        elif dist_to_goal < self.best_goal_distance - self.frontier_progress_epsilon:
            self.best_goal_distance = dist_to_goal
            progress_detected = True

        if self.last_progress_pose is None:
            self.last_progress_pose = robot_pose
            progress_detected = True
        elif self._distance_between_goals(robot_pose, self.last_progress_pose) >= self.frontier_motion_epsilon:
            self.last_progress_pose = robot_pose
            progress_detected = True

        if progress_detected or self.last_progress_time == rospy.Time(0):
            self.last_progress_time = now

    def _robot_pose_in_odom_frame(self):
        if self.latest_odom is None:
            return None

        robot = PoseStamped()
        robot.header = self.latest_odom.header
        if not robot.header.frame_id:
            robot.header.frame_id = "odom"
        robot.pose = self.latest_odom.pose.pose
        return robot

    def _transform_plan_pose(self, pose_msg, plan_frame, target_frame):
        stamped = PoseStamped()
        stamped.header = pose_msg.header
        if not stamped.header.frame_id:
            stamped.header.frame_id = plan_frame
        if stamped.header.stamp == rospy.Time(0):
            stamped.header.stamp = rospy.Time.now()
        stamped.pose = pose_msg.pose
        if stamped.header.frame_id == target_frame:
            return stamped
        return self._transform_pose(stamped, target_frame)

    def _map_world_to_cell(self, x, y):
        if self.latest_map is None:
            return None

        info = self.latest_map.info
        if info.resolution <= 0.0:
            return None

        origin_x = info.origin.position.x
        origin_y = info.origin.position.y
        mx = int((x - origin_x) / info.resolution)
        my = int((y - origin_y) / info.resolution)
        if mx < 0 or my < 0 or mx >= info.width or my >= info.height:
            return None
        return mx, my

    def _map_value_is_blocked(self, value):
        if value < 0:
            return self.local_plan_guard_unknown_is_blocked
        return value >= self.local_plan_guard_map_occupied_threshold

    def _map_near_blocked(self, x, y):
        cell = self._map_world_to_cell(x, y)
        if cell is None or self.latest_map is None:
            return False

        info = self.latest_map.info
        mx, my = cell
        radius_cells = int(math.ceil(self.local_plan_guard_map_clearance / info.resolution))
        radius_sq = radius_cells * radius_cells
        width = info.width
        height = info.height
        data = self.latest_map.data

        min_y = max(0, my - radius_cells)
        max_y = min(height - 1, my + radius_cells)
        min_x = max(0, mx - radius_cells)
        max_x = min(width - 1, mx + radius_cells)

        for cy in range(min_y, max_y + 1):
            dy = cy - my
            for cx in range(min_x, max_x + 1):
                dx = cx - mx
                if dx * dx + dy * dy > radius_sq:
                    continue
                value = data[cy * width + cx]
                if self._map_value_is_blocked(value):
                    return True
        return False

    def _segment_sample_count(self, x0, y0, x1, y1):
        length = math.hypot(x1 - x0, y1 - y0)
        step = max(0.02, self.local_plan_guard_segment_step)
        return max(1, int(math.ceil(length / step)))

    def _local_plan_blocked_by_scan(self, robot_pose):
        if self.latest_local_plan is None or not self.latest_local_plan.poses:
            return None
        if self.latest_scan is None and self.latest_map is None:
            return None

        target_frame = robot_pose.header.frame_id
        plan_frame = self.latest_local_plan.header.frame_id or target_frame
        rx, ry, robot_yaw = self._pose_xy_yaw(robot_pose.pose)
        scan_window = math.radians(self.local_plan_guard_window_deg)
        map_start_ignore = max(self.local_plan_guard_min_distance, self.local_plan_guard_map_start_ignore)
        blocked_hits = 0
        checked = 0
        last_target_xy = (rx, ry)
        last_target_dist = 0.0
        last_map_xy = None
        last_blocked = None

        if self.latest_map is not None:
            try:
                robot_map_pose = self._transform_pose(robot_pose, self.output_frame)
                last_map_xy = (robot_map_pose.pose.position.x, robot_map_pose.pose.position.y)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                last_map_xy = None

        for pose_msg in self.latest_local_plan.poses[1:]:
            try:
                plan_pose = self._transform_plan_pose(pose_msg, plan_frame, target_frame)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                continue

            px = plan_pose.pose.position.x
            py = plan_pose.pose.position.y
            dist = math.hypot(px - rx, py - ry)
            if dist < self.local_plan_guard_min_distance:
                last_target_xy = (px, py)
                last_target_dist = dist
                continue
            if dist > self.local_plan_guard_lookahead:
                break

            checked += 1
            rel_angle = self._angle_delta(math.atan2(py - ry, px - rx), robot_yaw)

            map_pose = None
            if self.latest_map is not None:
                try:
                    map_pose = self._transform_plan_pose(pose_msg, plan_frame, self.output_frame)
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    map_pose = None
                if map_pose is not None:
                    map_x = map_pose.pose.position.x
                    map_y = map_pose.pose.position.y
                    if last_map_xy is not None:
                        sample_count = self._segment_sample_count(last_map_xy[0], last_map_xy[1], map_x, map_y)
                        for index in range(1, sample_count + 1):
                            ratio = float(index) / float(sample_count)
                            sample_dist = last_target_dist + ratio * (dist - last_target_dist)
                            if sample_dist < map_start_ignore:
                                continue
                            if sample_dist > self.local_plan_guard_lookahead:
                                break
                            sample_x = last_map_xy[0] + ratio * (map_x - last_map_xy[0])
                            sample_y = last_map_xy[1] + ratio * (map_y - last_map_xy[1])
                            if self._map_near_blocked(sample_x, sample_y):
                                return (
                                    sample_dist,
                                    0.0,
                                    math.degrees(rel_angle),
                                    "map_segment_occupied_or_too_close",
                                )
                    elif dist >= map_start_ignore and self._map_near_blocked(map_x, map_y):
                        return (dist, 0.0, math.degrees(rel_angle), "map_occupied_or_too_close")
                    last_map_xy = (map_x, map_y)

            if self.latest_scan is None:
                last_target_xy = (px, py)
                last_target_dist = dist
                continue

            clearance = self._scan_clearance(rel_angle, scan_window)
            if clearance <= dist + self.local_plan_guard_margin:
                blocked_hits += 1
                last_blocked = (dist, clearance, math.degrees(rel_angle), "scan_clearance")
                if blocked_hits >= 2:
                    return last_blocked
            else:
                blocked_hits = 0

            last_target_xy = (px, py)
            last_target_dist = dist

        if checked == 0:
            return None
        return None

    def _handle_blocked_local_plan(self, now, blocked):
        if (
            self.last_local_plan_guard_time != rospy.Time(0)
            and (now - self.last_local_plan_guard_time).to_sec() < self.local_plan_guard_cooldown
        ):
            return

        robot_pose = self._robot_pose_in_output_frame()
        dist_to_goal = None
        if robot_pose is not None and self.last_goal is not None:
            dist_to_goal = self._distance_to_goal(robot_pose, self.last_goal)

        self.last_local_plan_guard_time = now
        self.cancel_pub.publish(GoalID())
        self._publish_stop_cmd()
        self._set_motion_brake(now, self.local_plan_guard_brake_duration)
        self._set_explore_pause(now, min(self.force_escape_hold, 3.0))
        if robot_pose is not None:
            self._remember_blocked_escape_direction(robot_pose, blocked[2])
            if self.last_goal is not None:
                self._remember_blocked_escape_direction(
                    robot_pose,
                    self._goal_bearing_deg(robot_pose, self.last_goal),
                )
        rospy.logwarn(
            "Local TEB plan appears unsafe (%s): plan_dist %.2fm, clearance %.2fm, "
            "bearing %.1fdeg; canceling current goal and clearing RViz local plan",
            blocked[3],
            blocked[0],
            blocked[1],
            blocked[2],
        )

        if robot_pose is not None:
            self._trigger_forced_escape(now, robot_pose, "local_plan_hits_obstacle", dist_to_goal)

    def _maybe_guard_local_plan(self, now):
        if self._manual_mode_active():
            return
        if not self.local_plan_guard_enable:
            return
        if self.last_goal is None:
            return

        robot_odom_pose = self._robot_pose_in_odom_frame()
        if robot_odom_pose is None:
            return

        blocked = self._local_plan_blocked_by_scan(robot_odom_pose)
        if blocked is None:
            return

        self._publish_empty_local_plan(self.latest_local_plan.header)
        self._handle_blocked_local_plan(now, blocked)

    def _maybe_request_frontier_switch(self, now):
        if self._manual_mode_active():
            return
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

        dist_to_goal = self._distance_to_goal(robot_pose, self.last_goal)
        if dist_to_goal <= self.goal_reached_tolerance:
            return

        self._update_progress(now, robot_pose, dist_to_goal)
        if (now - self.last_progress_time).to_sec() < self.frontier_timeout:
            return

        if self._trigger_forced_escape(now, robot_pose, "frontier_stalled", dist_to_goal):
            return

        self.cancel_pub.publish(GoalID())
        self.last_goal_time = now
        self.last_progress_time = now
        self.best_goal_distance = dist_to_goal
        self.last_progress_pose = robot_pose
        rospy.logwarn(
            "frontier_goal_supervisor stalled for %.1fs with distance %.2fm; requesting frontier switch",
            self.frontier_timeout,
            dist_to_goal,
        )

    def _maybe_handle_aborted_goal(self, now):
        if self._manual_mode_active():
            self.pending_abort_goal_id = None
            return
        if not self.force_escape_on_abort or self.pending_abort_goal_id is None:
            return
        if self.last_goal_source != "frontier":
            self.pending_abort_goal_id = None
            return

        robot_pose = self._robot_pose_in_output_frame()
        if robot_pose is None:
            return

        dist_to_goal = None
        if self.last_goal is not None:
            dist_to_goal = self._distance_to_goal(robot_pose, self.last_goal)

        if self._trigger_forced_escape(now, robot_pose, "move_base_aborted", dist_to_goal):
            return

    def _publish_fallback_goal(self, goal_msg):
        self.fallback_pub.publish(goal_msg)
        self.last_goal = goal_msg
        self.last_goal_time = rospy.Time.now()
        self.last_goal_source = "fallback"
        self.last_fallback_time = self.last_goal_time
        self._reset_goal_progress()
        self.fallback_count += 1

    def _maybe_publish_fallback_goal(self, now):
        if self._manual_mode_active():
            return
        if not self.enable_fallback_explore or self.latest_odom is None:
            return
        if (now - self.startup_time).to_sec() < self.fallback_startup_grace:
            return
        if (now - self.last_fallback_time).to_sec() < self.fallback_period:
            return
        if self._motion_brake_active(now):
            return

        goal_age = float("inf") if self.last_goal is None else (now - self.last_goal_time).to_sec()
        moving_age = float("inf") if self.last_nonzero_cmd_time == rospy.Time(0) else (now - self.last_nonzero_cmd_time).to_sec()
        progress_age = float("inf") if self.last_progress_time == rospy.Time(0) else (now - self.last_progress_time).to_sec()
        # Give an active frontier goal a fair chance before fallback exploration
        # starts taking over. Ackermann vehicles often need extra time to arc
        # through narrow passages even when the current global plan is valid.
        if self.last_goal_source == "frontier" and goal_age < self.frontier_timeout:
            return
        if progress_age < self.fallback_idle_timeout:
            return
        if goal_age < self.fallback_idle_timeout or moving_age < self.fallback_idle_timeout:
            return

        robot_pose = self._robot_pose_in_output_frame()
        if robot_pose is None:
            return

        goal_msg, angle, distance, clearance = self._make_fallback_goal(robot_pose)
        if goal_msg is None:
            rospy.logwarn_throttle(
                3.0,
                "No active motion for %.1fs, but no map-safe fallback goal is available",
                min(goal_age, moving_age),
            )
            return
        self._publish_fallback_goal(goal_msg)
        rospy.logwarn(
            "No active motion for %.1fs; publishing fallback goal %.2fm away at %.1fdeg (scan clearance %.2fm)",
            min(goal_age, moving_age),
            distance,
            math.degrees(angle),
            clearance,
        )

    def check_timeout(self, _event):
        now = rospy.Time.now()
        if now == rospy.Time(0):
            return
        if self.startup_time == rospy.Time(0):
            self.startup_time = now

        if self._manual_mode_active():
            self.pending_abort_goal_id = None
            return

        self._maybe_handle_aborted_goal(now)
        self._maybe_guard_local_plan(now)
        self._maybe_request_frontier_switch(now)
        self._maybe_publish_fallback_goal(now)


if __name__ == "__main__":
    rospy.init_node("frontier_goal_supervisor")
    FrontierGoalSupervisor()
    rospy.spin()
