"""Minimal stand-in for the `carla` Python package on machines without CARLA (aarch64 training nodes).
Only the plain data types that team_code imports at module level or uses in geometry helpers. Never use for simulation."""
import math


class Color:
    def __init__(self, r=0, g=0, b=0, a=255):
        self.r, self.g, self.b, self.a = r, g, b, a


class Vector3D:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __add__(self, o): return Vector3D(self.x + o.x, self.y + o.y, self.z + o.z)
    def __sub__(self, o): return Vector3D(self.x - o.x, self.y - o.y, self.z - o.z)
    def __mul__(self, s): return Vector3D(self.x * s, self.y * s, self.z * s)
    def length(self): return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)


class Location(Vector3D):
    def distance(self, o): return (self - o).length()


class Rotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch, self.yaw, self.roll = float(pitch), float(yaw), float(roll)


class Transform:
    def __init__(self, location=None, rotation=None):
        self.location = location or Location(); self.rotation = rotation or Rotation()
