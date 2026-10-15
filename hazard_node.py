#!/usr/bin/env python3
"""
hazard_node.py  -  rover_pipeline  (Node 2: hazard mapping grid)

Publishes a nav_msgs/OccupancyGrid on /hazard_map:
    0   = safe        100 = hazardous        -1 = unknown

Two modes:
  * DEMO mode (default, cloud_topic == ''):
        publishes a fixed 5x5-cell hazard block, same as the original sketch.
  * REAL mode (cloud_topic set, e.g. /camera/depth/points):
        builds a height grid from each PointCloud2 and flags cells that
        exceed the rover's step-height or slope limits.

IMPORTANT for REAL mode: the cloud must already be expressed in the grid's
frame (frame_id, default 'map') with the rover at the grid centre.
This node does NOT do TF transforms. If your cloud is in a camera frame,
transform it first (or publish a static transform and a transformed cloud).

Parameters (override with --ros-args -p name:=value):
  cloud_topic      (str)   ''       empty = demo mode
  frame_id         (str)   map
  width, height    (int)   50, 50   cells
  resolution       (float) 0.2      metres per cell
  max_step         (float) 0.15     metres (Agilex Scout limit)
  max_slope_deg    (float) 40.0     degrees
  min_pts_per_cell (int)   3        cells with fewer points stay unknown
  publish_rate     (float) 5.0      Hz
"""

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy,
                       qos_profile_sensor_data)
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class HazardMappingNode(Node):
    def __init__(self):
        super().__init__('hazard_mapping_node')

        self.declare_parameter('cloud_topic', '')
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('width', 50)
        self.declare_parameter('height', 50)
        self.declare_parameter('resolution', 0.2)
        self.declare_parameter('max_step', 0.15)
        self.declare_parameter('max_slope_deg', 40.0)
        self.declare_parameter('min_pts_per_cell', 3)
        self.declare_parameter('publish_rate', 5.0)

        p = self.get_parameter
        self.cloud_topic = p('cloud_topic').value
        self.frame_id = p('frame_id').value
        self.width = int(p('width').value)
        self.height = int(p('height').value)
        self.resolution = float(p('resolution').value)
        self.max_step = float(p('max_step').value)
        self.max_slope = float(p('max_slope_deg').value)
        self.min_pts = int(p('min_pts_per_cell').value)

        # Grid is centred on the rover/origin.
        self.origin_x = -self.width * self.resolution / 2.0
        self.origin_y = -self.height * self.resolution / 2.0

        # Latest grid, shape (height, width), int8.
        self.grid = self._demo_grid() if not self.cloud_topic else \
            np.full((self.height, self.width), -1, dtype=np.int8)

        # Map-style QoS so RViz shows the last grid even if it subscribes late.
        map_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.grid_pub = self.create_publisher(OccupancyGrid, '/hazard_map', map_qos)

        if self.cloud_topic:
            self.create_subscription(
                PointCloud2, self.cloud_topic, self.cloud_callback,
                qos_profile_sensor_data)
            self.get_logger().info(f'REAL mode: reading {self.cloud_topic}')
        else:
            self.get_logger().info('DEMO mode: publishing a fixed hazard block')

        self.create_timer(1.0 / float(p('publish_rate').value), self.publish_map)

    # ------------------------------------------------------------------ #
    def _demo_grid(self):
        data = np.zeros((self.height, self.width), dtype=np.int8)
        data[20:25, 20:25] = 100  # simulated hazard block
        return data

    # ------------------------------------------------------------------ #
    @staticmethod
    def _cloud_to_xyz(msg):
        arr = point_cloud2.read_points(
            msg, field_names=('x', 'y', 'z'), skip_nans=True)
        if isinstance(arr, np.ndarray) and arr.dtype.names:
            return np.column_stack([arr['x'], arr['y'], arr['z']]).astype(np.float64)
        return np.array(list(arr), dtype=np.float64).reshape(-1, 3)

    # ------------------------------------------------------------------ #
    def cloud_callback(self, msg: PointCloud2):
        if msg.header.frame_id and msg.header.frame_id != self.frame_id:
            self.get_logger().warn(
                f'Cloud frame "{msg.header.frame_id}" != grid frame "{self.frame_id}". '
                'No TF is applied; results will be wrong unless they are aligned.',
                throttle_duration_sec=10.0)

        pts = self._cloud_to_xyz(msg)
        if pts.size == 0:
            return

        ix = np.floor((pts[:, 0] - self.origin_x) / self.resolution).astype(np.int64)
        iy = np.floor((pts[:, 1] - self.origin_y) / self.resolution).astype(np.int64)
        ok = (ix >= 0) & (ix < self.width) & (iy >= 0) & (iy < self.height)
        if not np.any(ok):
            return
        ix, iy, z = ix[ok], iy[ok], pts[ok, 2]

        n = self.width * self.height
        flat = iy * self.width + ix

        count = np.bincount(flat, minlength=n).astype(np.float64)
        zsum = np.bincount(flat, weights=z, minlength=n)
        zmin = np.full(n, np.inf)
        zmax = np.full(n, -np.inf)
        np.minimum.at(zmin, flat, z)
        np.maximum.at(zmax, flat, z)

        observed = count >= self.min_pts
        mean_z = np.where(observed, zsum / np.maximum(count, 1.0), np.nan)
        step = np.where(observed, zmax - zmin, 0.0)

        mean_z = mean_z.reshape(self.height, self.width)
        step = step.reshape(self.height, self.width)
        observed = observed.reshape(self.height, self.width)

        # Slope from the gradient of the mean-height grid (NaN-safe).
        gy, gx = np.gradient(mean_z, self.resolution)
        slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
        slope_bad = np.nan_to_num(slope_deg, nan=0.0) > self.max_slope

        hazard = observed & ((step > self.max_step) | slope_bad)

        grid = np.full((self.height, self.width), -1, dtype=np.int8)
        grid[observed] = 0
        grid[hazard] = 100
        self.grid = grid

    # ------------------------------------------------------------------ #
    def publish_map(self):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.orientation.w = 1.0  # valid quaternion (default is all zeros)

        msg.data = self.grid.flatten().tolist()
        self.grid_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = HazardMappingNode()
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
