import numpy as np
from gym.envs.registration import register
from numpy.lib.function_base import piecewise

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.vehicle.objects import Obstacle



class MergeEnv(AbstractEnv):

    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    RIGHT_LANE_REWARD: float = 0.1
    HIGH_SPEED_REWARD: float = 0.4
    STEERING_REWARD: float = -0.2
    LANE_CENTER_REWARD = -0.1
    DISTANCE_REWARD = -0.1   # 为了加大车距可能要调这个值
    MERGING_SPEED_REWARD = -0.4
    DISTANCE_RANGE = [20, 40]

    seed_value = 1

    ramp_length = 80
    ab_end = np.random.uniform(100, 200)  # jk段长度为随机值，使匝道的出现位置有随机性
    bc_end = ab_end + ramp_length
    cd_end = bc_end + np.random.uniform(100, 200)
    de_end = cd_end + ramp_length
    ef_end = de_end + np.random.uniform(100, 200)
    fg_end = ef_end + ramp_length
    gh_end = fg_end + 5000

    def default_config(self) -> dict:
        config = super().default_config()
        # 下面的make_road代码是针对3车道特别设计的，在其他数量车道的情况下会由于车道线型line_type = [[c, s],  [n, s], [n, c]]只有3个而报错
        # 原因是gym.make时会先生成个默认config的环境，此时如果默认车道设为其他值，用这个特定生成3车道的代码就会报错，之后env.reset才会用自定义的config覆盖默认config
        # 为了方便就把默认config和自定义config设成一样的
        config.update({
            "observation": {
                "type": "Kinematics"  # 没有指定obs中具体需要的参数的话就按observation.py中相应类型的默认设置
            },
            "action": {
                "type": "ContinuousAction"
            },
            "lanes_count": 3,
            "vehicles_count": 30,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 400,
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,  # The reward received when colliding with a vehicle.
            "reward_speed_range": [20, 30],
            "reward_steering_range" : [-90,90],
            "offroad_terminal": False
        })
        return config

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        merging_speed_reward = 0
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]  # 获取_id
        
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])  # 把速度归一化，但没裁减
        scaled_steer = utils.lmap(np.abs(self.vehicle.action["steering"]), self.config["reward_steering_range"], [0, 1])  # 转向角归一化，但没裁减（转向角是DDPG输出动作，会在主程序中进行裁减，不会超过范围）
        longitudinal, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self.vehicle)  # 获取前车
        if front_vehicle and not isinstance(front_vehicle, Obstacle):
            longitudinal_front, lateral_front = front_vehicle.lane.local_coordinates(front_vehicle.position)
            scaled_distance = utils.lmap(longitudinal_front - longitudinal, self.DISTANCE_RANGE, [0, 1])
        else:
            scaled_distance = 0

        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.RIGHT_LANE_REWARD * lane / (self.config["lanes_count"] - 1) \
            + self.HIGH_SPEED_REWARD * (1 - np.clip(np.abs(1 - scaled_speed), 0, 1)) \
            + self.STEERING_REWARD * scaled_steer \
            + self.LANE_CENTER_REWARD * (np.abs(lateral) / 2) \
            + self.DISTANCE_REWARD * np.clip(np.abs(scaled_distance), 0, 1)

        # Altruistic penalty  让道奖励
        for car in self.road.vehicles:
            if (((car.lane_index == ("b", "c", 3)  and self.vehicle.lane_index == ("b", "c", 2)) or
               (car.lane_index == ("d", "e", 3)  and self.vehicle.lane_index == ("d", "e", 2)) or
               (car.lane_index == ("f", "g", 3)  and self.vehicle.lane_index == ("f", "g", 2)))) and isinstance(car, ControlledVehicle):  # 当有2条直道时，直道编号为0-1，匝道编号为2
                # 当自车处在匝道相邻车道时才对匝道上车辆产生的规避让道奖励进行计算，自车不在相邻车道而由于他车占道产生的匝道车辆速度下降不对自车的奖励产生影响
                merging_speed_reward = self.MERGING_SPEED_REWARD * \
                          np.clip((car.target_speed - car.speed) / car.target_speed,0,1)  # MERGING_SPEED_REWARD: float = -0.4
                reward += merging_speed_reward
            # 匝道上车辆的目标车速是固定值=30（merging_v.target_speed），如果自车挡住匝道车辆的合流空间，匝道车就不得不减速，偏离目标车速，而自车由于没有给匝道车提供合流空间使其被迫减速而获得负的奖励

        self.right_lane_reward = self.RIGHT_LANE_REWARD * lane / (self.config["lanes_count"] - 1)
        self.high_speed_reward = self.HIGH_SPEED_REWARD * (1 - np.clip(np.abs(1 - scaled_speed), 0, 1))
        self.steering_reward = self.STEERING_REWARD * scaled_steer
        self.lane_center_reward = self.LANE_CENTER_REWARD * (np.abs(lateral) / 2)
        self.distance_reward = self.DISTANCE_REWARD * np.clip(np.abs(scaled_distance), 0, 1)
        self.merging_speed_reward = merging_speed_reward
        self.collision_reward = self.config["collision_reward"] * self.vehicle.crashed

        reward = utils.lmap(reward,  # 变道奖励+碰撞奖励+右车道奖励+高速奖励+合流奖励
                          [self.config["collision_reward"] + self.STEERING_REWARD + self.LANE_CENTER_REWARD + self.DISTANCE_REWARD + self.MERGING_SPEED_REWARD,
                           self.HIGH_SPEED_REWARD + self.RIGHT_LANE_REWARD ], [0, 1])

        # 偏离车道和跑到匝道时的奖励
        reward = 0 if not self.vehicle.on_road or self.vehicle.lane_index == ('b','c', 3) or  self.vehicle.lane_index == ('d','e', 3) or self.vehicle.lane_index == ('f','g', 3) \
                else reward

        return reward

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road) or \
            self.vehicle.lane_index == ('b','c', 3) or  self.vehicle.lane_index == ('d','e', 3) or self.vehicle.lane_index == ('f','g', 3)
    # 碰撞或行驶超过一定距离就结束
    # self.steps指决策步数（频率为5）

    def _info(self, obs: np.ndarray, action: int) -> dict:
        info = super()._info(obs, action)
        info["right_lane_reward"] = self.right_lane_reward
        info["high_speed_reward"] = self.high_speed_reward
        info["steering_reward"] = self.steering_reward
        info["lane_center_reward"] = self.lane_center_reward
        info["distance_reward"] = self.distance_reward
        info["merging_speed_reward"] = self.merging_speed_reward
        info["collision_reward"] = self.collision_reward
        return info

    def _reset(self) -> None:
        np.random.seed(self.seed_value)  # 让不同算法下每个epi生成的环境是一样的，可以用来测试对比
        self._make_road()
        self._make_vehicles()
        self.seed_value += 1

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        # 改直道数量的话要改起止点纵坐标y，在线型line_type和line_type_merge中添加中间车道的线型，for i in range(2)中的车道数要改，ljk = StraightLane中的起止点纵坐标要加上车道宽（+4）
        # 自车生成的车道要改road.network.get_lane(("a", "b", 1)，奖励中匝道车和自车的车道id要改if vehicle.lane_index == ("b", "c", 2)，self.vehicle.lane_index == ("b", "c", 1)

        net = RoadNetwork()

        # Highway lanes


        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH, 2*StraightLane.DEFAULT_WIDTH]  # DEFAULT_WIDTH = 4
        line_type = [[c, s],  [n, s], [n, c]]  # [[实线，虚线]，[无，虚线]，[无，实线]]
        line_type_merge = [[c, s],  [n, s], [n, s]]  # [[实线，虚线]，[无，虚线]，[无，虚线]]
        # 生成3条直道
        for i in range(self.config["lanes_count"]):  # 0, 1, 2
            net.add_lane("a", "b", StraightLane([0, y[i]], [self.ab_end, y[i]], line_types=line_type[i]))
            net.add_lane("b", "c", StraightLane([self.ab_end, y[i]], [self.bc_end, y[i]], line_types=line_type_merge[i]))
            net.add_lane("c", "d", StraightLane([self.bc_end, y[i]], [self.cd_end, y[i]], line_types=line_type[i]))
            net.add_lane("d", "e", StraightLane([self.cd_end, y[i]], [self.de_end, y[i]], line_types=line_type_merge[i]))
            net.add_lane("e", "f", StraightLane([self.de_end, y[i]], [self.ef_end, y[i]], line_types=line_type[i]))
            net.add_lane("f", "g", StraightLane([self.ef_end, y[i]], [self.fg_end, y[i]], line_types=line_type_merge[i]))
            net.add_lane("g", "h", StraightLane([self.fg_end, y[i]], [self.gh_end, y[i]], line_types=line_type[i]))

        # 生成3条匝道
        ramp_bc = StraightLane([self.ab_end, 3* StraightLane.DEFAULT_WIDTH], [self.bc_end, 3*StraightLane.DEFAULT_WIDTH], line_types=[n, c], forbidden=True)
        ramp_de = StraightLane([self.cd_end, 3 * StraightLane.DEFAULT_WIDTH], [self.de_end, 3 * StraightLane.DEFAULT_WIDTH], line_types=[n, c], forbidden=True)
        ramp_fg = StraightLane([self.ef_end, 3 * StraightLane.DEFAULT_WIDTH], [self.fg_end, 3 * StraightLane.DEFAULT_WIDTH], line_types=[n, c], forbidden=True)

        net.add_lane("b", "c", ramp_bc)
        net.add_lane("d", "e", ramp_de)
        net.add_lane("f", "g", ramp_fg)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, ramp_bc.position(self.ramp_length, 0)))  # 以bc段为参考系设置障碍物的位置
        road.objects.append(Obstacle(road, ramp_de.position(self.ramp_length, 0)))
        road.objects.append(Obstacle(road, ramp_fg.position(self.ramp_length, 0)))
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road
        ego_start = 10 # 自车起点
        # 生成自车
        ego_vehicle = self.action_type.vehicle_class(road,
                                                     road.network.get_lane(("a", "b", np.random.randint(0,self.config["lanes_count"]))).position(ego_start, 0),
                                                     speed=25)
        # np.random.randint随机生成整数，范围：[low, high)

        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        # 生成直道车辆
        for i in range(self.config["vehicles_count"]):
            x0 = np.random.uniform(ego_start+35*(i+1), ego_start+35*(i+1)+5)
            y0 = 4 * np.random.randint(0, self.config["lanes_count"])
            road.vehicles.append(other_vehicles_type(road, [x0, y0], 0, speed=np.random.uniform(25, 30)))

            # while  x0 < self.ab_end:
            #     lane = road.network.get_lane(("a", "b", np.random.randint(0, self.config["lanes_count"])))
            #     road.vehicles.append(other_vehicles_type(road, lane.position(x0, 0), lane.heading_at(x0), speed=straight_speed))
            # while x0 >= self.ab_end and x0 < self.bc_end:
            #     lane = road.network.get_lane(("b", "c", np.random.randint(0, self.config["lanes_count"])))
            #     road.vehicles.append(other_vehicles_type(road, lane.position(x0, 0), lane.heading_at(x0), speed=straight_speed))
            # while x0 >= self.bc_end and x0 < self.cd_end:
            #     lane = road.network.get_lane(("c", "d", np.random.randint(0, self.config["lanes_count"])))
            #     road.vehicles.append(other_vehicles_type(road, lane.position(x0, 0), lane.heading_at(x0), speed=straight_speed))
            # while x0 >= self.cd_end and x0 < self.de_end:
            #     lane = road.network.get_lane(("d", "e", np.random.randint(0, self.config["lanes_count"])))
            #     road.vehicles.append(other_vehicles_type(road, lane.position(x0, 0), lane.heading_at(x0), speed=straight_speed))
            # while x0 >= self.de_end and x0 < self.ef_end:
            #     lane = road.network.get_lane(("e", "f", np.random.randint(0, self.config["lanes_count"])))
            #     road.vehicles.append(other_vehicles_type(road, lane.position(x0, 0), lane.heading_at(x0), speed=straight_speed))
            # while x0 >= self.ef_end and x0 < self.fg_end:
            #     lane = road.network.get_lane(("f", "g", np.random.randint(0, self.config["lanes_count"])))
            #     road.vehicles.append(other_vehicles_type(road, lane.position(x0, 0), lane.heading_at(x0), speed=straight_speed))
            # while x0 >= self.fg_end and x0 < self.gh_end:
            #     lane = road.network.get_lane(("g", "h", np.random.randint(0, self.config["lanes_count"])))
            #     road.vehicles.append(other_vehicles_type(road, lane.position(x0, 0), lane.heading_at(x0), speed=straight_speed))

        self.vehicle = ego_vehicle


register(
    id='merge-v0',
    entry_point='highway_env.envs:MergeEnv',
)
