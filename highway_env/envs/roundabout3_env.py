from pickle import FALSE, TRUE
from random import random
import random as rand
from typing import Tuple
from xmlrpc.client import TRANSPORT_ERROR

from gym.envs.registration import register
import numpy as np
from sqlalchemy import true

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, CircularLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.uncertainty.prediction import IntervalVehicle


class RoundaboutEnv(AbstractEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 30,
                "absolute": True,
                "features_range": {"x": [-5, 5], "y": [-5, 5], "vx": [-30, 30], "vy": [-15, 15]},
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                
                "flatten": False,
                "observe_intentions": False
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "incoming_vehicle_destination": None,
            "my_other_vehicles_type":"highway_env.vehicle.uncertainty.estimation.MultipleModelVehicle",
            #"my_other_vehicles_type":"highway_env.vehicle.uncertainty.estimation.IntervalVehicle",
            
            "collision_reward": -5,
            "high_speed_reward": 1,
            "right_lane_reward": 0,
            "lane_change_reward": -0.05,
            "finish_reward":  1,
            "v_is_zero_reward":-1,
            "collision_incoming":-0.1,
            "reward_speed_range" :[7.0,9.0],
            
            "scaling": 17,
            "screen_width": 1800,
            "screen_height": 1400,
            "centering_position": [0.5, 0.6],
            "duration": 30,

            
            "random_destion":False,
            "hard":False,
            "use_uncertain":True,#使用
            "show_trajectories": False ,
            "uncertain":False,#显示
            "ego_attention": False,#显示
            "save_speed":False,#保存速度数据
            "save_speed_picture":False,#保存图像数据
            "is_Dsac":False,
            "num":0
        })
        return config

    def _reward(self, action: int) -> float:
        lane_change = action == 0 or action == 2
        
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        #print(self.vehicle.speed)

        if self.vehicle.lane_index[1] is self.my_destinations:
            self.finish = 1
        else:
            self.finish = 0


        if self.vehicle.speed < 3:
            self.wait_pen = 1
            
        else:
            self.wait_pen = 0
        
        

        reward = self.config["collision_reward"] * self.vehicle.crashed \
                 + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1) \
                 + self.config["lane_change_reward"] * lane_change \


        reward = self.config["finish_reward"] if self.vehicle_is_finish() else reward      

        # reward = self.config["collision_reward"] * self.vehicle.crashed \
        #     + self.config["high_speed_reward"] * \
        #          MDPVehicle.get_speed_index(self.vehicle) / max(MDPVehicle.SPEED_COUNT - 1, 1) \
        #     + self.config["lane_change_reward"] * lane_change \
        #     + self.config["finish_reward"] * self.finish \
        #     + self.config["v_is_zero_reward"] * self.wait_pen \
            #+ self.config["collision_incoming"] * self.pre_collision
        # rewards = utils.lmap(reward,
        #         [self.config["collision_reward"] + self.config["lane_change_reward"]+self.config["v_is_zero_reward"],
        #         self.config["high_speed_reward"]+self.config["finish_reward"]], [0, 1])
        
        return reward
                         

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        
        return self.vehicle.crashed or self.steps >= self.config["duration"]  or self.vehicle_is_finish()###or self.vehicle.speed ==0
##################
    
    def vehicle_is_finish(self) -> bool:
        '''判断车辆是否到达指定位置'''
        
        return (self.vehicle.lane_index[0]  in self.my_destinations  \
               and self.vehicle.lane_index[1] in self.my_destinations \
               and self.vehicle.lane.local_coordinates(self.vehicle.position)[0] >= 25)
    def vehicle_crash(self) -> bool:
        return self.vehicle.crashed    
##########################
    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()
        # try:
        #     self.unwrapped.configure(env_config)
        #     # Reset the environment to ensure configuration is applied
        #     self.reset()
        
    ####################################################
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        #print(action)

        obs, reward, done, info = super().step(action)
        self._clear_vehicles()
        #self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])#这个就是不确定性估计
        #self._make_vehicles()
        return obs, reward, done, info

    def _clear_vehicles(self) -> None:
        is_leaving = lambda vehicle: "il" in vehicle.lane_index[0] and "o" in vehicle.lane_index[1] \
                                     and vehicle.lane.local_coordinates(vehicle.position)[0] \
                                     >= vehicle.lane.length - 4 * vehicle.LENGTH
        self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
                              vehicle in self.controlled_vehicles or not (is_leaving(vehicle) or vehicle.route is None)]
    ##################################

    def _make_road(self) -> None:
        # Circle lanes: (s)outh/(e)ast/(n)orth/(w)est (e)ntry/e(x)it.
        center = [0, 0]  # [m]
        radius = 20  # [m]20
        alpha = 24  # [deg]

        net = RoadNetwork()
        radii = [radius, radius+4]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, s], [n, c]]
        for lane in [0, 1]:
            net.add_lane("se", "ex",
                         CircularLane(center, radii[lane], np.deg2rad(90 - alpha), np.deg2rad(alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("ex", "ee",
                         CircularLane(center, radii[lane], np.deg2rad(alpha), np.deg2rad(-alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("ee", "nx",
                         CircularLane(center, radii[lane], np.deg2rad(-alpha), np.deg2rad(-90 + alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("nx", "ne",
                         CircularLane(center, radii[lane], np.deg2rad(-90 + alpha), np.deg2rad(-90 - alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("ne", "wx",
                         CircularLane(center, radii[lane], np.deg2rad(-90 - alpha), np.deg2rad(-180 + alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("wx", "we",
                         CircularLane(center, radii[lane], np.deg2rad(-180 + alpha), np.deg2rad(-180 - alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("we", "sx",
                         CircularLane(center, radii[lane], np.deg2rad(180 - alpha), np.deg2rad(90 + alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("sx", "se",
                         CircularLane(center, radii[lane], np.deg2rad(90 + alpha), np.deg2rad(90 - alpha),
                                      clockwise=False, line_types=line[lane]))

        # Access lanes: (r)oad/(s)ine
        access = 170  # [m]170
        dev =  85 # [m]85
        a = 5  # [m]
        delta_st = 0.2*dev  # [m]

        delta_en = dev-delta_st
        w = 2*np.pi/dev
        
        net.add_lane("ser", "ses", StraightLane([2, access], [2, dev/2], line_types=(s, c)))
        net.add_lane("ses", "se", SineLane([2+a, dev/2], [2+a, dev/2-delta_st], a, w, -np.pi/2, line_types=(c, c)))
        net.add_lane("sx", "sxs", SineLane([-2-a, -dev/2+delta_en], [-2-a, dev/2], a, w, -np.pi/2+w*delta_en, line_types=(c, c)))
        net.add_lane("sxs", "sxr", StraightLane([-2, dev / 2], [-2, access], line_types=(n, c)))

        net.add_lane("eer", "ees", StraightLane([access, -2], [dev / 2, -2], line_types=(s, c)))
        net.add_lane("ees", "ee", SineLane([dev / 2, -2-a], [dev / 2 - delta_st, -2-a], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("ex", "exs", SineLane([-dev / 2 + delta_en, 2+a], [dev / 2, 2+a], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)))
        net.add_lane("exs", "exr", StraightLane([dev / 2, 2], [access, 2], line_types=(n, c)))

        net.add_lane("ner", "nes", StraightLane([-2, -access], [-2, -dev / 2], line_types=(s, c)))
        net.add_lane("nes", "ne", SineLane([-2 - a, -dev / 2], [-2 - a, -dev / 2 + delta_st], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("nx", "nxs", SineLane([2 + a, dev / 2 - delta_en], [2 + a, -dev / 2], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)))
        net.add_lane("nxs", "nxr", StraightLane([2, -dev / 2], [2, -access], line_types=(n, c)))

        net.add_lane("wer", "wes", StraightLane([-access, 2], [-dev / 2, 2], line_types=(s, c)))
        net.add_lane("wes", "we", SineLane([-dev / 2, 2+a], [-dev / 2 + delta_st, 2+a], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("wx", "wxs", SineLane([dev / 2 - delta_en, -2-a], [-dev / 2, -2-a], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)))
        net.add_lane("wxs", "wxr", StraightLane([-dev / 2, -2], [-access, -2], line_types=(n, c)))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road
        

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """




        




        position_deviation = 2
        speed_deviation = 2

        # Ego-vehicle
        ego_destinations = [["exs","exr"], ["sxs","sxr"], ["nxs","nxr"],["wxs","wxr"]]
        ego_index = np.random.randint(0,4)
        
        ego_lane = self.road.network.get_lane(("ser", "ses", 0))
        #ego_lane = self.road.network.get_lane(("ses", "se", 0))

        ego_vehicle = self.action_type.vehicle_class(self.road,
                                                    #  (0,225/2),
                                                     ego_lane.position(125, 0),
                                                     speed=10,
                                                     heading=ego_lane.heading_at(140))
        
        if self.config["random_destion"]:
            try:
                # ego_vehicle.plan_route_to("nxs")
                ego_vehicle.plan_route_to(ego_destinations[ego_index][1])
            except AttributeError:
                pass

            self.my_destinations = ego_destinations[ego_index]
        else:
            ego_vehicle.plan_route_to("sxr")
            self.my_destinations = ego_destinations[3]
        
        MDPVehicle.SPEED_MIN = 0
        MDPVehicle.SPEED_MAX = 16
        MDPVehicle.SPEED_COUNT = 3
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        # Incoming vehicle
        destinations = ["exr", "sxr", "nxr"]
        # try:

        if self.config["use_uncertain"]:
            other_vehicles_type = utils.class_from_path(self.config["my_other_vehicles_type"])  
        else:
            other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
         

        other_vehicles_type.DISTANCE_WANTED = 7  # Low jam distance
        other_vehicles_type.COMFORT_ACC_MAX = 6
        other_vehicles_type.COMFORT_ACC_MIN = -3
        
        vehicle = other_vehicles_type.make_on_lane(self.road,
                                                ("we", "sx", 1),
                                                longitudinal=5 + self.np_random.randn()*position_deviation,
                                                speed=16 + self.np_random.randn() * speed_deviation)   
         
        
        if self.config["incoming_vehicle_destination"] is not None:
            destination = destinations[self.config["incoming_vehicle_destination"]]
        else:
            destination = self.np_random.choice(destinations)
        vehicle.plan_route_to(destination)
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

        # Other vehicles
        if self.config["hard"]:
            num_oth_vehicle1 = np.random.randint(2,4)
            num_oth_vehicle2 = np.random.randint(-4,0)
        else:
            num_oth_vehicle1 = np.random.randint(2,3)
            num_oth_vehicle2 = np.random.randint(-1,0)




        for i in list(range(1, num_oth_vehicle1)) + list(range(num_oth_vehicle2, 0)):

            vehicle = other_vehicles_type.make_on_lane(self.road,
                                                    ("we", "sx", 0),
                                                    longitudinal=18*i + self.np_random.randn()*position_deviation,
                                                    speed=16 + self.np_random.randn() * speed_deviation)
            vehicle.plan_route_to(self.np_random.choice(destinations))
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

        # Entering vehicle
        # 随机选择n个四个方向的来车
        pre_rote = [("eer", "ees", 0),("ner", "nes", 0),("wer", "wes", 0),("ser", "ses", 0),("fer", "fes", 0)]
        num_entering_vehicle = np.random.randint(3,5)
        #num_entering_vehicle = np.random.randint(4,size=10)
        
        pick_index1 = rand.sample(range(0,len(pre_rote)),num_entering_vehicle) 
        
        pick_index2 = rand.sample(range(0,len(pre_rote)),num_entering_vehicle) 

        # position1 = np.random.randint(45,5)
        # position2 = np.random.randint(65,75)
        for i in pick_index1:
            vehicle = other_vehicles_type.make_on_lane(self.road,
                                                    pre_rote[i],
                                                    longitudinal=50 + self.np_random.randn() * position_deviation,
                                                    speed=16 + self.np_random.randn() * speed_deviation)
            
            
            vehicle.plan_route_to(self.np_random.choice(destinations))
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

            # vehicle1 = other_vehicles_type.make_on_lane(self.road,
            #                                         pre_rote[i],
            #                                         longitudinal=150 + self.np_random.randn() * position_deviation,
            #                                         speed=16 + self.np_random.randn() * speed_deviation)
            
            
            # vehicle1.plan_route_to(self.np_random.choice(destinations))
            # vehicle1.randomize_behavior()
            # self.road.vehicles.append(vehicle1)

        
        # vehicle = other_vehicles_type.make_on_lane(self.road,
        #                                         pre_rote[0],
        #                                         longitudinal=150 + self.np_random.randn() * position_deviation,
        #                                         speed=16 + self.np_random.randn() * speed_deviation)
        
        
        # vehicle.plan_route_to('sxr')
        # vehicle.randomize_behavior()
        # self.road.vehicles.append(vehicle)

        # vehicle = other_vehicles_type.make_on_lane(self.road,
        #                                         pre_rote[1],
        #                                         longitudinal=150 + self.np_random.randn() * position_deviation,
        #                                         speed=16 + self.np_random.randn() * speed_deviation)
        
        
        # vehicle.plan_route_to('sxr')
        # vehicle.randomize_behavior()
        # self.road.vehicles.append(vehicle)
        
        if self.config["hard"]:
            for j in pick_index2:
                vehicle = other_vehicles_type.make_on_lane(self.road,
                                                        pre_rote[j],
                                                        longitudinal=70 + self.np_random.randn() * position_deviation,
                                                        speed=16 + self.np_random.randn() * speed_deviation)
                
                
                vehicle.plan_route_to(self.np_random.choice(destinations))
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
        # print(self.road.vehicles)

register(
    id='roundabout-v3',
    entry_point='highway_env.envs:RoundaboutEnv',
)
