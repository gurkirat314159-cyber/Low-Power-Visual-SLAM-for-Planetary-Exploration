#!/usr/bin/env python3
"""
imu_node.py  -  rover_pipeline  (Node 3: IMU pose tracker)

Dead-reckons a 6-DOF pose from a 200 Hz sensor_msgs/Imu stream and publishes
  * nav_msgs/Odometry on /odom   (frame 'odom' -> child 'base_link')
  * the matching TF transform odom -> base_link

How it works
  1. CALIBRATION: for the first `calib_samples` messages the rover must be
     STILL. The node averages the gyro (gyro bias), measures gravity
     magnitude, and sets the initial roll/pitch from the accelerometer.
  2. ORIENTATION: integrates the bias-corrected gyro into a quaternion
     (or uses the orientation in the message if use_msg_orientation is true
     and the message really contains one).
  3. POSITION: rotates body acceleration into the world frame, subtracts
     gravity, then integrates twice (3D, including z).
  4. ZUPT: a sliding window of the last `zupt_window` samples detects when
     the rover is stationary and then zeroes velocity to limit drift.

HONEST LIMITS: double-integrating an IMU drifts quadratically with time
(metres within tens of seconds). Yaw also drifts (no magnetometer). Treat
this as a short-term motion estimate and fuse it with wheel odometry or
visual odometry (e.g. robot_localization) for anything longer.

Parameters (override with --ros-args -p name:=value):
  imu_topic            (str)   /imu/data
  odom_frame           (str)   odom
  base_frame           (str)   base_link
  calib_samples        (int)   200     (~1 s at 200 Hz, keep rover still)
  use_msg_orientation  (bool)  false
  zupt                 (bool)  true
  zupt_window          (int)   40      samples in the sliding window
  zupt_gyro_thr        (float) 0.03    rad/s
  zupt_accel_thr       (float) 0.30    m/s^2 (gravity-removed, world frame)
  publish_tf           (bool)  true
"""

import math
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster


# ---------------------------------------------------------------------- #
# Quaternion helpers, quaternion layout is [x, y, z, w], body -> world.
# ---------------------------------------------------------------------- #
def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def quat_normalize(q):
    return q / np.linalg.norm(q)


def quat_from_rotvec(v):
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return quat_normalize(np.array([0.5 * v[0], 0.5 * v[1], 0.5 * v[2], 1.0]))
    axis = v / angle
    s = math.sin(angle / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2.0)])


def quat_to_rotmat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_from_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


class IMUPoseTrackerNode(Node):
    def __init__(self):
        super().__init__('imu_pose_tracker')

        self.declare_parameter('imu_topic', '/imu/data')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('calib_samples', 200)
        self.declare_parameter('use_msg_orientation', False)
        self.declare_parameter('zupt', True)
        self.declare_parameter('zupt_window', 40)
        self.declare_parameter('zupt_gyro_thr', 0.03)
        self.declare_parameter('zupt_accel_thr', 0.30)
        self.declare_parameter('publish_tf', True)

        p = self.get_parameter
        imu_topic = p('imu_topic').value
        self.odom_frame = p('odom_frame').value
        self.base_frame = p('base_frame').value
        self.calib_samples = int(p('calib_samples').value)
        self.use_msg_orientation = bool(p('use_msg_orientation').value)
        self.zupt_enabled = bool(p('zupt').value)
        self.zupt_gyro_thr = float(p('zupt_gyro_thr').value)
        self.zupt_accel_thr = float(p('zupt_accel_thr').value)
        self.publish_tf = bool(p('publish_tf').value)

        # Sliding window of the last N (gyro_norm, world_accel_norm) samples,
        # used for stationary detection. deque(maxlen) keeps memory flat.
        self.window = deque(maxlen=int(p('zupt_window').value))

        # State
        self.q = np.array([0.0, 0.0, 0.0, 1.0])   # body -> world
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)                      # world frame
        self.gyro_bias = np.zeros(3)
        self.g_mag = 9.80665
        self.calibrated = False
        self.calib_a = []
        self.calib_w = []
        self.last_t = None

        # IMUs normally publish BEST_EFFORT. A RELIABLE subscriber would
        # never connect to them, so use the sensor-data profile.
        self.create_subscription(Imu, imu_topic, self.imu_callback,
                                 qos_profile_sensor_data)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info(
            f'Waiting for {imu_topic}. Keep the rover STILL for the first '
            f'{self.calib_samples} samples (calibration).')

    # ------------------------------------------------------------------ #
    def _stamp_and_time(self, msg):
        s = msg.header.stamp
        t = s.sec + s.nanosec * 1e-9
        if t == 0.0:  # driver left the stamp empty -> use node clock
            now = self.get_clock().now()
            return now.to_msg(), now.nanoseconds * 1e-9
        return s, t

    @staticmethod
    def _msg_orientation(msg):
        """Return the message quaternion if it is genuinely valid, else None."""
        o = msg.orientation
        q = np.array([o.x, o.y, o.z, o.w])
        n = np.linalg.norm(q)
        if msg.orientation_covariance[0] == -1.0 or not (0.9 < n < 1.1):
            return None
        return q / n

    # ------------------------------------------------------------------ #
    def _finish_calibration(self, t):
        mean_a = np.mean(self.calib_a, axis=0)
        self.gyro_bias = np.mean(self.calib_w, axis=0)
        self.g_mag = float(np.linalg.norm(mean_a))

        if not (7.0 < self.g_mag < 12.5):
            self.get_logger().warn(
                f'Measured gravity is {self.g_mag:.2f}, expected about 9.8 m/s^2. '
                'Check that the IMU publishes m/s^2 (not g) and that the '
                'rover was still during calibration.')

        ax, ay, az = mean_a
        roll = math.atan2(ay, az)
        pitch = math.atan2(-ax, math.hypot(ay, az))
        self.q = quat_from_rpy(roll, pitch, 0.0)

        self.last_t = t
        self.calibrated = True
        self.calib_a.clear()
        self.calib_w.clear()
        self.get_logger().info(
            f'Calibrated: gyro bias={np.round(self.gyro_bias, 4)} rad/s, '
            f'gravity={self.g_mag:.3f} m/s^2, '
            f'roll={math.degrees(roll):.1f} deg, pitch={math.degrees(pitch):.1f} deg')

    # ------------------------------------------------------------------ #
    def imu_callback(self, msg: Imu):
        stamp, t = self._stamp_and_time(msg)

        a = np.array([msg.linear_acceleration.x,
                      msg.linear_acceleration.y,
                      msg.linear_acceleration.z])
        w = np.array([msg.angular_velocity.x,
                      msg.angular_velocity.y,
                      msg.angular_velocity.z])

        if not self.calibrated:
            self.calib_a.append(a)
            self.calib_w.append(w)
            if len(self.calib_a) >= self.calib_samples:
                self._finish_calibration(t)
            return

        dt = t - self.last_t
        self.last_t = t
        if dt <= 0.0 or dt > 0.5:   # duplicate stamp or data gap: skip a step
            return

        w_c = w - self.gyro_bias

        # Orientation
        q_msg = self._msg_orientation(msg) if self.use_msg_orientation else None
        if q_msg is not None:
            self.q = q_msg
        else:
            self.q = quat_normalize(quat_mul(self.q, quat_from_rotvec(w_c * dt)))
        R = quat_to_rotmat(self.q)

        # Specific force -> world acceleration (remove gravity)
        a_world = R @ a - np.array([0.0, 0.0, self.g_mag])

        # Sliding window + zero-velocity update
        self.window.append((float(np.linalg.norm(w_c)),
                            float(np.linalg.norm(a_world))))
        stationary = (
            self.zupt_enabled
            and len(self.window) == self.window.maxlen
            and max(g for g, _ in self.window) < self.zupt_gyro_thr
            and max(c for _, c in self.window) < self.zupt_accel_thr)

        if stationary:
            self.vel[:] = 0.0
        else:
            self.vel += a_world * dt
            self.pos += self.vel * dt

        self._publish(stamp, R, w_c)

    # ------------------------------------------------------------------ #
    def _publish(self, stamp, R, w_body):
        qx, qy, qz, qw = (float(v) for v in self.q)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = float(self.pos[0])
        odom.pose.pose.position.y = float(self.pos[1])
        odom.pose.pose.position.z = float(self.pos[2])
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        # Odometry twist is expressed in the child (body) frame.
        v_body = R.T @ self.vel
        odom.twist.twist.linear.x = float(v_body[0])
        odom.twist.twist.linear.y = float(v_body[1])
        odom.twist.twist.linear.z = float(v_body[2])
        odom.twist.twist.angular.x = float(w_body[0])
        odom.twist.twist.angular.y = float(w_body[1])
        odom.twist.twist.angular.z = float(w_body[2])
        self.odom_pub.publish(odom)

        if self.publish_tf:
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self.odom_frame
            tf.child_frame_id = self.base_frame
            tf.transform.translation.x = float(self.pos[0])
            tf.transform.translation.y = float(self.pos[1])
            tf.transform.translation.z = float(self.pos[2])
            tf.transform.rotation.x = qx
            tf.transform.rotation.y = qy
            tf.transform.rotation.z = qz
            tf.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = IMUPoseTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
