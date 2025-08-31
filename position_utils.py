# +-----------------------------------------------------------------------------------------------------------------+
# | Adapted from Enzo De Sena's Scattering Delay Network (SDN) repository: https://github.com/enzodesena/sdn-matlab |
# +-----------------------------------------------------------------------------------------------------------------+

import math
import torch


class Cuboid:
    """https://github.com/enzodesena/sdn-matlab/blob/master/Cuboid.m"""
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Room:
    """https://github.com/enzodesena/sdn-matlab/blob/master/Room.m"""
    def __init__(self, shape):
        self.shape = Cuboid(*shape)


class Position:
    """https://github.com/enzodesena/sdn-matlab/blob/master/Position.m"""
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def r(self):
        return torch.sqrt(self.x) + torch.sqrt(self.y)

    def theta(self):
        return self.x + torch.mul(self.y, torch.pi)

    def is_equal(self, pos):
        tol = 1e-10
        if torch.abs(self.x - pos.x) < tol and torch.abs(self.y - pos.y) < tol and torch.abs(self.z - pos.z) < tol:
            return True
        return False

    @staticmethod
    def get_angle(pos1, pos2):
        pos_x = pos2.x - pos1.x
        pos_y = pos2.y - pos1.y
        t = torch.complex(pos_x, pos_y)
        return float(torch.angle(t))

    @staticmethod
    def get_distance(pos1, pos2):
        return float(
            math.sqrt(math.pow(pos1.x - pos2.x, 2) + math.pow(pos1.y - pos2.y, 2) + math.pow(pos1.z - pos2.z, 2)))


def get_reflect_pos_one_dim(x1, y1, x2, y2):
    """https://github.com/enzodesena/sdn-matlab/blob/01d15dd626f1d2a8f42cf1f65f1f1895f23a67a6/private/getReflectPosOneDim.m"""
    if x1 == x2:
        return x1
    else:
        return (x1 * y2 + x2 * y1) / (y1 + y2)


def get_reflect_pos_face1(source_pos, observ_pos):
    """https://github.com/enzodesena/sdn-matlab/blob/01d15dd626f1d2a8f42cf1f65f1f1895f23a67a6/private/getReflectPosFace1.m"""
    pos = Position(0, 0, 0)
    pos.x = get_reflect_pos_one_dim(source_pos.x, source_pos.y, observ_pos.x, observ_pos.y)
    pos.z = get_reflect_pos_one_dim(source_pos.z, source_pos.y, observ_pos.z, observ_pos.y)
    return pos


def get_reflect_pos(room, face_index, source_pos, observ_pos):
    """https://github.com/enzodesena/sdn-matlab/blob/01d15dd626f1d2a8f42cf1f65f1f1895f23a67a6/private/getReflectPos.m"""
    pos = Position(0, 0, 0)

    if face_index == 0:
        pos = get_reflect_pos_face1(source_pos, observ_pos)

    if face_index == 1:
        source_pos_t = Position(source_pos.y, room.shape.x - source_pos.x, source_pos.z)
        observ_pos_t = Position(observ_pos.y, room.shape.x - observ_pos.x, observ_pos.z)
        pos_t = get_reflect_pos_face1(source_pos_t, observ_pos_t)
        pos = Position(room.shape.x - pos_t.y, pos_t.x, pos_t.z)

    if face_index == 2:
        source_pos_t = Position(room.shape.x - source_pos.x, room.shape.y - source_pos.y, source_pos.z)
        observ_pos_t = Position(room.shape.x - observ_pos.x, room.shape.y - observ_pos.y, observ_pos.z)
        pos_t = get_reflect_pos_face1(source_pos_t, observ_pos_t)
        pos = Position(room.shape.x - pos_t.x, room.shape.y - pos_t.y, pos_t.z)

    if face_index == 3:
        source_pos_t = Position(room.shape.y - source_pos.y, source_pos.x, source_pos.z)
        observ_pos_t = Position(room.shape.y - observ_pos.y, observ_pos.x, observ_pos.z)
        pos_t = get_reflect_pos_face1(source_pos_t, observ_pos_t)
        pos = Position(pos_t.y, room.shape.y - pos_t.x, pos_t.z)

    if face_index == 4:
        source_pos_t = Position(source_pos.x, room.shape.z - source_pos.z, source_pos.y)
        observ_pos_t = Position(observ_pos.x, room.shape.z - observ_pos.z, observ_pos.y)
        pos_t = get_reflect_pos_face1(source_pos_t, observ_pos_t)
        pos = Position(pos_t.x, pos_t.z, room.shape.z - pos_t.y)

    if face_index == 5:
        source_pos_t = Position(source_pos.x, source_pos.z, room.shape.y - source_pos.y)
        observ_pos_t = Position(observ_pos.x, observ_pos.z, room.shape.y - observ_pos.y)
        pos_t = get_reflect_pos_face1(source_pos_t, observ_pos_t)
        pos = Position(pos_t.x, room.shape.y - pos_t.z, pos_t.y)

    return pos


def get_node_pos(room_dim, src_position, mic_position):
    return [get_reflect_pos(Room(room_dim), i, Position(*src_position), Position(*mic_position)) for i in
            range(6)]


def get_distances(n_junctions, room_dim, src_pos, mic_pos):
    node_pos = get_node_pos(room_dim, src_pos, mic_pos)

    dist_nodes, dist_src_nodes, dist_nodes_mic = [], [], []

    for i in range(n_junctions):
        for j in range(n_junctions):
            if i != j:
                dist_nodes.append(Position.get_distance(node_pos[i], node_pos[j]))
        dist_src_nodes.append(Position.get_distance(Position(*src_pos), node_pos[i]))
        dist_nodes_mic.append(Position.get_distance(node_pos[i], Position(*mic_pos)))

    dist_src_mic = Position.get_distance(Position(*src_pos), Position(*mic_pos))

    return dist_src_nodes, dist_nodes, dist_nodes_mic, [dist_src_mic]