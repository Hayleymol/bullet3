import pybullet
import gym, gym.spaces, gym.utils
import numpy as np
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import pybullet_data


class XmlBasedRobot:
    """
	Base class for mujoco .xml based agents.
	"""

    self_collision = True

    def __init__(self, robot_name, action_dim, obs_dim, self_collision):
        self.parts = None
        self.objects = []
        self.jdict = None
        self.ordered_joints = None
        self.robot_body = None

        high = np.ones([action_dim])
        self.action_space = gym.spaces.Box(-high, high)
        high = np.inf * np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)

        #self.model_xml = model_xml
        self.robot_name = robot_name
        self.self_collision = self_collision

    def addToScene(self, bullet_client, bodies):
        self._p = bullet_client

        if self.parts is not None:
            parts = self.parts
        else:
            parts = {}

        if self.jdict is not None:
            joints = self.jdict
        else:
            joints = {}

        if self.ordered_joints is not None:
            ordered_joints = self.ordered_joints
        else:
            ordered_joints = []

        if np.isscalar(bodies):  # streamline the case where bodies is actually just one body
            bodies = [bodies]

        dump = 0

        for i in range(len(bodies)):
            if self._p.getNumJoints(bodies[i]) == 0:
                part_name, robot_name = self._p.getBodyInfo(bodies[i])

                #print(part_name) floor
                #print(robot_name) floor_obj

                self.robot_name = robot_name.decode("utf8")
                part_name = part_name.decode("utf8")
                parts[part_name] = BodyPart(self._p, part_name, bodies, i, -1)

            for j in range(self._p.getNumJoints(bodies[i])):

                # joint 12개가 돌면서 걔들 위치를 reset 시킨다

                #print(self._p.getBasePositionAndOrientation(1))
                #self._p.resetBasePositionAndOrientation(bodies[0], [0, 0, .5], [0, 0.5, 0.5, 0])
                #print(bodies[i])
                self._p.setJointMotorControl2(bodies[i],
                                              j,
                                              pybullet.POSITION_CONTROL,
                                              positionGain=0.1,
                                              velocityGain=0.1,
                                              force=0)
                jointInfo = self._p.getJointInfo(bodies[i], j)

                joint_name = jointInfo[1]
                part_name = jointInfo[12]

                joint_name = joint_name.decode("utf8")
                part_name = part_name.decode("utf8")

                #print("Joint Name :" + joint_name)
                #print("Part Name : " + part_name)

                if dump: print("ROBOT PART '%s'" % part_name)
                if dump:
                    print(
                      "ROBOT JOINT '%s'" % joint_name
                    )  # limits = %+0.2f..%+0.2f effort=%0.3f speed=%0.3f" % ((joint_name,) + j.limits()) )
                    #print(j.limits())

                parts[part_name] = BodyPart(self._p, part_name, bodies, i, j)


                if part_name == self.robot_name:
                    self.robot_body = parts[part_name]


                if i == 0 and j == 0 and self.robot_body is None:  # if nothing else works, we take this as robot_body
                    parts[self.robot_name] = BodyPart(self._p, self.robot_name, bodies, 0, -1)
                    self.robot_body = parts[self.robot_name]


                if joint_name[:6] == "ignore":
                    Joint(self._p, joint_name, bodies, i, j).disable_motor()
                    continue


                # ordered joint 에서 8개의 진짜 joint 만 빼냄
                if joint_name[:8] != "jointfix":
                    joints[joint_name] = Joint(self._p, joint_name, bodies, i, j)
                    ordered_joints.append(joints[joint_name])

                    joints[joint_name].power_coef = 100.0


        #print(self.robot_body)
        return parts, joints, ordered_joints, self.robot_body

    def reset_pose(self, position, orientation):
        self.parts[self.robot_name].reset_pose(position, orientation)


class MJCFBasedRobot(XmlBasedRobot):
    """
	Base class for mujoco .xml based agents.
	"""

    def __init__(self, model_xml, robot_name, action_dim, obs_dim, self_collision=True):
        XmlBasedRobot.__init__(self, robot_name, action_dim, obs_dim, self_collision)
        self.model_xml = model_xml
        self.doneLoading = 0


    def reset(self, bullet_client):

        self._p = bullet_client
        #print("Created bullet_client with id=", self._p._client)
        if (self.doneLoading == 0):
            self.ordered_joints = []
            self.doneLoading = 1

            print(os.path.join(os.path.dirname(__file__), "data", self.model_xml))


            if self.self_collision:

                self.objects = self._p.loadMJCF(os.path.join(pybullet_data.getDataPath(), "mjcf",
                                                         self.model_xml),
                                            flags=pybullet.URDF_USE_SELF_COLLISION |
                                            pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(self._p, self.objects)

                print(self._p.getBasePositionAndOrientation(0))  # floor 값이고
                print(self._p.getBasePositionAndOrientation(1))  # robot 맞는거같은데


            else:

                self.objects = self._p.loadMJCF(
                    os.path.join(pybullet_data.getDataPath(), "mjcf", self.model_xml))


                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(self._p, self.objects)


        self.robot_specific_reset(self._p)

        s = self.calc_state()  # optimization: calc_state() can calculate something in self.* for calc_potential() to use

        return s

    def calc_potential(self):
        return 0


class URDFBasedRobot(XmlBasedRobot):
    """
	Base class for URDF .xml based robots.
	"""
    
    def __init__(self,
               model_urdf,
               robot_name,
               action_dim,
               obs_dim,
               basePosition=[0, 0, .5],
               baseOrientation=[0,0.5,0.5,0],
               fixed_base=False,
               self_collision=True):
        XmlBasedRobot.__init__(self, robot_name, action_dim, obs_dim, self_collision)


        self.model_urdf = model_urdf
        self.basePosition = basePosition
        self.baseOrientation = baseOrientation
        self.fixed_base = fixed_base
        self.doneLoading = 0

    def reset(self, bullet_client):
        self._p = bullet_client

        if (self.doneLoading == 0):
            self.ordered_joints = []
            self.doneLoading = 1

            print(os.path.join(os.path.dirname(__file__), "laikago", self.model_urdf))

            '''
            if self.self_collision:

                # base position, orientation 문제네
                self.objects = self._p.loadURDF(os.path.join("/home/hayley/.mujoco/stable-baseline/laikago/", self.model_urdf),
                                                useFixedBase=self.fixed_base,
                                                flags=pybullet.URDF_USE_SELF_COLLISION |
                                                      pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)


                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(self._p, self.objects)
                self._p.resetBasePositionAndOrientation(self.robot_body, [0, 0, .5], [0,0.5,0.5,0])
            '''

            # 로딩 들어올 때 처음부터 시작해야함
            if self.self_collision:
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
                    self._p,
                    self._p.loadURDF(os.path.join("/home/hayley/.mujoco/stable-baseline/laikago/", self.model_urdf),
                                     basePosition=self.basePosition,
                                     baseOrientation=self.baseOrientation,
                                     useFixedBase=self.fixed_base,
                                     flags=pybullet.URDF_USE_SELF_COLLISION))

                #self._p.resetBasePositionAndOrientation(1, [0, 0, .5], [0, 0.5, 0.5, 0])

                #print(self._p.getNumJoints(1)) # 이렇게 주면 12
                print(self._p.getBasePositionAndOrientation(0)) # floor 값이고
                print(self._p.getBasePositionAndOrientation(1)) # robot 맞는거같은데

                #print(self._p.getBodyInfo(0)) # (b'floor', b'floor_obj')
                #print(self._p.getBodyInfo(1)) # (b'chassis', b'plane')
                print(len(self.parts)) # 13


            else:

                self.objects = self._p.loadURDF(
                    os.path.join("/home/hayley/.mujoco/stable-baseline/laikago/", self.model_urdf))
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(self._p, self.objects)


        self.robot_specific_reset(self._p)

        s = self.calc_state(
        )  # optimization: calc_state() can calculate something in self.* for calc_potential() to use
        self.potential = self.calc_potential()

        return s

    def calc_potential(self):
        return 0


class Pose_Helper:  # dummy class to comply to original interface

    def __init__(self, body_part):
        self.body_part = body_part

    def xyz(self):
        return self.body_part.current_position()

    def rpy(self):
        return pybullet.getEulerFromQuaternion(self.body_part.current_orientation())

    def orientation(self):
        return self.body_part.current_orientation()


class BodyPart:

    def __init__(self, bullet_client, body_name, bodies, bodyIndex, bodyPartIndex):
        self.bodies = bodies
        self._p = bullet_client
        self.bodyIndex = bodyIndex
        self.bodyPartIndex = bodyPartIndex
        self.initialPosition = self.current_position()

        # 여기서 얘 정확한 위치좀 뽑아보자
        #print(self.bodyPartIndex)
        #print(self.initialPosition)
        self.initialOrientation = self.current_orientation()
        self.bp_pose = Pose_Helper(self)

    def state_fields_of_pose_of(
        self, body_id,
        link_id=-1):  # a method you will most probably need a lot to get pose and orientation
        if link_id == -1:
          (x, y, z), (a, b, c, d) = self._p.getBasePositionAndOrientation(body_id)
        else:
          (x, y, z), (a, b, c, d), _, _, _, _ = self._p.getLinkState(body_id, link_id)
        return np.array([x, y, z, a, b, c, d])

    def get_position(self):
        return self.current_position()

    def get_pose(self):
        return self.state_fields_of_pose_of(self.bodies[self.bodyIndex], self.bodyPartIndex)

    def speed(self):
        if self.bodyPartIndex == -1:
            (vx, vy, vz), _ = self._p.getBaseVelocity(self.bodies[self.bodyIndex])
        else:
            (x, y, z), (a, b, c, d), _, _, _, _, (vx, vy, vz), (vr, vp, vy) = self._p.getLinkState(
                self.bodies[self.bodyIndex], self.bodyPartIndex, computeLinkVelocity=1)
        return np.array([vx, vy, vz])

    def current_position(self):
        return self.get_pose()[:3]

    def current_orientation(self):
        return self.get_pose()[3:]

    def get_orientation(self):
        return self.current_orientation()

    def reset_position(self, position):
        self._p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], position,
                                                self.get_orientation())

    def reset_orientation(self, orientation):
        self._p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], self.get_position(),
                                                orientation)

    def reset_velocity(self, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0]):
        self._p.resetBaseVelocity(self.bodies[self.bodyIndex], linearVelocity, angularVelocity)

    def reset_pose(self, position, orientation):
        self._p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], position, orientation)

    def pose(self):
        return self.bp_pose

    def contact_list(self):
        #print(self._p.getContactPoints(self.))
        return self._p.getContactPoints(self.bodies[self.bodyIndex], -1, self.bodyPartIndex, -1)


class Joint:

    def __init__(self, bullet_client, joint_name, bodies, bodyIndex, jointIndex):
        self.bodies = bodies
        self._p = bullet_client
        self.bodyIndex = bodyIndex
        self.jointIndex = jointIndex
        self.joint_name = joint_name

        jointInfo = self._p.getJointInfo(self.bodies[self.bodyIndex], self.jointIndex)
        self.lowerLimit = jointInfo[8]
        self.upperLimit = jointInfo[9]

        #print(self.lowerLimit)
        #print(self.upperLimit)

        self.power_coeff = 0

    def set_state(self, x, vx):
        self._p.resetJointState(self.bodies[self.bodyIndex], self.jointIndex, x, vx)

    def current_position(self):  # just some synonyme method
        return self.get_state()

    def current_relative_position(self):
        pos, vel = self.get_state()
        pos_mid = 0.5 * (self.lowerLimit + self.upperLimit)
        return (2 * (pos - pos_mid) / (self.upperLimit - self.lowerLimit), 0.1 * vel)

    def get_state(self):
        x, vx, _, _ = self._p.getJointState(self.bodies[self.bodyIndex], self.jointIndex)
        return x, vx

    def get_position(self):
        x, _ = self.get_state()
        return x

    def get_orientation(self):
        _, r = self.get_state()
        return r

    def get_velocity(self):
        _, vx = self.get_state()
        return vx

    def set_position(self, position):
        self._p.setJointMotorControl2(self.bodies[self.bodyIndex],
                                      self.jointIndex,
                                      pybullet.POSITION_CONTROL,
                                      targetPosition=position)

    def set_velocity(self, velocity):
        self._p.setJointMotorControl2(self.bodies[self.bodyIndex],
                                      self.jointIndex,
                                      pybullet.VELOCITY_CONTROL,
                                      targetVelocity=velocity)

    def set_motor_torque(self, torque):  # just some synonyme method
        self.set_torque(torque)

    def set_torque(self, torque):
        self._p.setJointMotorControl2(bodyIndex=self.bodies[self.bodyIndex],
                                      jointIndex=self.jointIndex,
                                      controlMode=pybullet.TORQUE_CONTROL,
                                      force=torque)  #, positionGain=0.1, velocityGain=0.1)

    def reset_current_position(self, position, velocity):  # just some synonyme method
        self.reset_position(position, velocity)

    def reset_position(self, position, velocity):
        self._p.resetJointState(self.bodies[self.bodyIndex],
                                self.jointIndex,
                                targetValue=position,
                                targetVelocity=velocity)
        self.disable_motor()

    def disable_motor(self):
        self._p.setJointMotorControl2(self.bodies[self.bodyIndex],
                                      self.jointIndex,
                                      controlMode=pybullet.POSITION_CONTROL,
                                      targetPosition=0,
                                      targetVelocity=0,
                                      positionGain=0.1,
                                      velocityGain=0.1,
                                      force=0)