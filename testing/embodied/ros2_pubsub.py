#!/usr/bin/env python3
"""Two-process ROS2 transport test: publisher and subscriber in separate
processes so the DDS path is exercised the way a real robot does it."""
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray

MODE = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
SZ = int(sys.argv[3]) if len(sys.argv) > 3 else 4096


class Pub(Node):
    def __init__(self):
        super().__init__("bench_pub")
        self.pub = self.create_publisher(UInt8MultiArray, "bench", 1000)

    def go(self):
        m = UInt8MultiArray()
        time.sleep(1.0)  # let discovery settle
        t0 = time.perf_counter()
        for i in range(N):
            m.data = i.to_bytes(8, "little") + bytes(SZ - 8)
            self.pub.publish(m)
        dt = time.perf_counter() - t0
        print(f"pub: {N} msgs of {SZ}B in {dt:.3f}s -> {N/dt:.0f} msg/s "
              f"({N*SZ/dt/1e6:.1f} MB/s)")
        time.sleep(1.0)


class Sub(Node):
    def __init__(self):
        super().__init__("bench_sub")
        self.n = 0
        self.t0 = None
        self.t1 = None
        self.create_subscription(UInt8MultiArray, "bench", self.cb, 1000)

    def cb(self, msg):
        now = time.perf_counter()
        if self.t0 is None:
            self.t0 = now
        self.n += 1
        self.t1 = now


def main():
    rclpy.init()
    if MODE == "pub":
        n = Pub()
        n.go()
    else:
        n = Sub()
        t_start = time.perf_counter()
        while time.perf_counter() - t_start < 12:
            rclpy.spin_once(n, timeout_sec=0.05)
        if n.n > 1:
            dt = n.t1 - n.t0
            print(f"sub: {n.n} msgs in {dt:.3f}s -> {n.n/dt:.0f} msg/s")
        else:
            print(f"sub: only {n.n} msgs")
    n.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
