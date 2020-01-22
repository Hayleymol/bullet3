from robot_bases import XmlBasedRobot, MJCFBasedRobot, URDFBasedRobot
import numpy as np
import pybullet
import os
import pybullet_data
from robot_bases import BodyPart


class WalkerBase(MJCFBasedRobot):

    def __init__(self, fn, robot_name, action_dim, obs_dim, power):
        MJCFBasedRobot.__init__(self, fn, robot_name, action_dim, obs_dim)
        self.power = power
        self.camera_x = 0
        self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0
        self.walk_target_x = 1e3  # kilometer away
        self.walk_target_y = 0
        self.body_xyz = [0, 0, 0]


    def robot_specific_reset(self, bullet_client):
        self._p = bullet_client
        for j in self.ordered_joints:
            j.reset_current_position(self.np_random.uniform(low=-0.1, high=0.1), 0)


        self.feet = [self.parts[f] for f in self.foot_list]

        self.feet_contact = np.array([0.0 for f in self.foot_list], dtype=np.float32)

        self.scene.actor_introduce(self)
        self.initial_z = None

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        for n, j in enumerate(self.ordered_joints):
            j.set_motor_torque(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))
            # 더 작아져야하나?
            #print(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))

    # 여기서 moving 을 결정 -> 지금 여기 문제있음
    def calc_state(self):
        j = np.array([j.current_relative_position() for j in self.ordered_joints],
                     dtype=np.float32).flatten()
        # even elements [0::2] position, scaled to -1..+1 between limits
        # odd elements  [1::2] angular speed, scaled to show -1..+1
        self.joint_speeds = j[1::2]

        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)


        body_pose = self.robot_body.pose()
        # 여기 찍어보면 지금 움직이지를 않음
        #print(body_pose.xyz())

        parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()]).flatten()
        self.body_xyz = (parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2]
                         )  # torso z is more informative than mean z
        self.body_rpy = body_pose.rpy()
        z = self.body_xyz[2]

        if self.initial_z == None:
            self.initial_z = z
        r, p, yaw = self.body_rpy
        self.walk_target_theta = np.arctan2(self.walk_target_y - self.body_xyz[1],
                                            self.walk_target_x - self.body_xyz[0])
        self.walk_target_dist = np.linalg.norm(
            [self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0]])
        angle_to_target = self.walk_target_theta - yaw

        rot_speed = np.array([[np.cos(-yaw), -np.sin(-yaw), 0], [np.sin(-yaw),
                                                                 np.cos(-yaw), 0], [0, 0, 1]])
        vx, vy, vz = np.dot(rot_speed,
                            self.robot_body.speed())  # rotate speed back to body point of view

        more = np.array(
            [
                z - self.initial_z,
                np.sin(angle_to_target),
                np.cos(angle_to_target),
                0.3 * vx,
                0.3 * vy,
                0.3 * vz,  # 0.3 is just scaling typical speed into -1..+1, no physical sense here
                r,
                p
            ],
            dtype=np.float32)
        return np.clip(np.concatenate([more] + [j] + [self.feet_contact]), -5, +5)

        def calc_potential(self):
            # progress in potential field is speed*dt, typical speed is about 2-3 meter per second, this potential will change 2-3 per frame (not per second),
            # all rewards have rew/frame units and close to 1.0
            debugmode = 0
            if (debugmode):
                print("calc_potential: self.walk_target_dist")
                print(self.walk_target_dist)
                print("self.scene.dt")
                print(self.scene.dt)
                print("self.scene.frame_skip")
                print(self.scene.frame_skip)
                print("self.scene.timestep")
                print(self.scene.timestep)

            return -self.walk_target_dist / self.scene.dt


class Ant(WalkerBase):
    foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']

    def __init__(self):
        WalkerBase.__init__(self, "ant.xml", "torso", action_dim=8, obs_dim=28, power=2.5)

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground


class Laikago(WalkerBase):
    foot_list = ['FR_lower_leg', 'FL_lower_leg', 'RR_lower_leg', 'RL_lower_leg']

    def __init__(self):
        WalkerBase.__init__(self, "laikago.urdf", "chassis", action_dim=12, obs_dim=36, power=0.2)

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.54 else -1
        # 몸통이 땅에 닿으면 죽게 만들어줘야 하고

