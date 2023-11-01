import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

np.set_printoptions(precision=3)

class Agent:
    def __init__(self):
        pass
    
    def act(self, state):
        # Simple-minded agent that always selects action 1
        return 1

class Car:
    def __init__(self, tyre="Intermediate"):
        self.default_tyre = tyre
        self.possible_tyres = ["Ultrasoft", "Soft", "Intermediate", "Fullwet"]
        self.pitstop_time = 23
        self.fuel_capacity = 100.0  # Maximum fuel capacity
        self.fuel_consumption_rate = 0.01  # Fuel consumption rate per unit distance
        self.reset()

    def reset(self):
        self.change_tyre(self.default_tyre)
        self.fuel_level = self.fuel_capacity
        
    def degrade(self, w, r):
        if self.tyre == "Ultrasoft":
            self.condition *= (1 - 0.0050 * w - (2500 - r) / 90000)
        elif self.tyre == "Soft":
            self.condition *= (1 - 0.0051 * w - (2500 - r) / 93000)
        elif self.tyre == "Intermediate":
            self.condition *= (1 - 0.0052 * abs(0.5 - w) - (2500 - r) / 95000)
        elif self.tyre == "Fullwet":
            self.condition *= (1 - 0.0053 * (1 - w) - (2500 - r) / 97000)
        
    def change_tyre(self, new_tyre):
        # Change car's tires
        assert new_tyre in self.possible_tyres
        self.tyre = new_tyre
        self.condition = 1.0
    
    def get_velocity(self, driving_style):
        # Calculate car velocity based on tire, condition, fuel level, and driving style
        base_velocity = 0
        if self.tyre == "Ultrasoft":
            base_velocity = 80.7
        elif self.tyre == "Soft":
            base_velocity = 80.1
        elif self.tyre == "Intermediate":
            base_velocity = 79.5
        elif self.tyre == "Fullwet":
            base_velocity = 79.0
        
        # Adjust velocity based on fuel level and driving style
        if driving_style == 1:
            velocity = base_velocity * (0.8 * self.condition ** 1.5) * (self.fuel_level / self.fuel_capacity)
        elif driving_style == 2:
            velocity = base_velocity * (0.6 * self.condition ** 1.5) * (self.fuel_level / self.fuel_capacity)
        elif driving_style == 3:
            velocity = base_velocity * (0.4 * self.condition ** 1.5) * (self.fuel_level / self.fuel_capacity)
        return velocity
    

class Track:
    def __init__(self, car=Car(), driving_style=1, agent_risk_level=0.1):
        self.total_laps = 162
        self.car = car
        self.driving_style = driving_style
        self.possible_weather = ["Dry", "20% Wet", "40% Wet", "60% Wet", "80% Wet", "100% Wet"]
        self.wetness = {
            "Dry": 0.00, "20% Wet": 0.20, "40% Wet": 0.40, "60% Wet": 0.60, "80% Wet": 0.80, "100% Wet": 1.00
        }
        self.p_transition = {
            "Dry": {
                "Dry": 0.987, "20% Wet": 0.013, "40% Wet": 0.000, "60% Wet": 0.000, "80% Wet": 0.000, "100% Wet": 0.000
            },
            "20% Wet": {
                "Dry": 0.012, "20% Wet": 0.975, "40% Wet": 0.013, "60% Wet": 0.000, "80% Wet": 0.000, "100% Wet": 0.000
            },
            "40% Wet": {
                "Dry": 0.000, "20% Wet": 0.012, "40% Wet": 0.975, "60% Wet": 0.013, "80% Wet": 0.000, "100% Wet": 0.000
            },
            "60% Wet": {
                "Dry": 0.000, "20% Wet": 0.000, "40% Wet": 0.012, "60% Wet": 0.975, "80% Wet": 0.013, "100% Wet": 0.000
            },
            "80% Wet": {
                "Dry": 0.000, "20% Wet": 0.000, "40% Wet": 0.000, "60% Wet": 0.012, "80% Wet": 0.975, "100% Wet": 0.013
            },
            "100% Wet": {
                "Dry": 0.000, "20% Wet": 0.000, "40% Wet": 0.000, "60% Wet": 0.000, "80% Wet": 0.012, "100% Wet": 0.988
            }
        }
        self.crash_probability = 0
        self.agent_risk_level = agent_risk_level  # Store the agent's risk level as an instance attribute
        self.reset()

    def reset(self):
        self.radius = np.random.randint(600, 1201)
        self.cur_weather = np.random.choice(self.possible_weather)
        self.is_done = False
        self.pitstop = False
        self.laps_cleared = 0
        self.car.reset()
        self.agent_fuel_style = 1  # Adjust this based on the agent's preferred fuel management style
        self.fuel_management_actions = {1: 0.8, 2: 0.6, 3: 0.4}  # Adjust based on desired fuel consumption for each style
        return self._get_state()

    def _get_state(self):
        return [self.car.tyre, self.car.condition, self.car.fuel_level, self.cur_weather, self.radius, self.laps_cleared, self.car.fuel_capacity]

    def transition(self, action=0):
        time_taken = 0

        # Check for pitstops and refueling
        if self.laps_cleared == int(self.laps_cleared):
            if self.pitstop:
                self.car.change_tyre(self.committed_tyre)
                time_taken += self.pitstop_time["ChangeTire"]

                if action % 2 == 1:  # If the action is odd, it's a refuel action
                    time_taken += self.pitstop_time["Refuel"]

                    # Check if the agent chose to refuel and change tires at the same time
                    if action // 2 * 2 != action:
                        reward = -40  # Agent chose both actions simultaneously
                    else:
                        reward = -20  # Agent only refueled
                else:
                    # Agent chose to change tires but not refuel
                    reward = -20
            else:
                # Agent did not choose to change tires or refuel
                reward = -20
        else:
            # Agent is not at the pitstop, continue with normal actions
            reward = 0
            
            
        # Update weather conditions
        self.cur_weather = np.random.choice(
            self.possible_weather, p=list(self.p_transition[self.cur_weather].values())
        )

        # Calculate distance traveled and velocity
        velocity = self.car.get_velocity(self.driving_style)  # Use the predefined driving style
        distance = (2 * np.pi * self.radius / 8)

        # Calculate crash probability based on velocity and wetness
        self.crash_probability = self.calculate_crash_probability(velocity, self.driving_style)

        # Adjust crash probability based on driving style and fuel level
        if self.agent_fuel_style == 2:
            self.crash_probability += 0.02  # Example adjustment for a more aggressive driving style
        elif self.agent_fuel_style == 3:
            self.crash_probability -= 0.02  # Example adjustment for a more cautious driving style

        # Check for crash event
        if np.random.random() < self.crash_probability:
            return -1000, None, True, 0  # Implement a massive negative reward for crashing

        # Add a check to avoid division by zero
        if velocity > 0:
            time_taken += distance / velocity
        else:
            time_taken += 1.0  # You can choose a default time increment when velocity is zero or negative

        # Apply degradation and fuel consumption
        self.car.degrade(w=self.wetness[self.cur_weather], r=self.radius)
        fuel_consumed = distance * self.car.fuel_consumption_rate * self.fuel_management_actions[self.agent_fuel_style]
        self.car.fuel_level -= fuel_consumed

        reward = 0 - time_taken

        # Implement crash-related rewards based on agent's risk level
        reward -= self.agent_risk_level * reward

        # Implement rewards/penalties based on fuel efficiency
        if fuel_consumed > 0:
            reward -= fuel_consumed

        # Check if the race is finished
        if self.laps_cleared == self.total_laps:
            return reward, None, True, 0  # Race is finished

        next_state = self._get_state()
        return reward, next_state, False, velocity

    def calculate_crash_probability(self, velocity, driving_style):
        # Calculate crash probability based on velocity, wetness, and driving style
        crash_probability = 0.01 * (velocity - 80) + 0.05 * self.wetness[self.cur_weather]
    
        # Adjust the crash probability based on driving style
        if driving_style == 2:
            crash_probability += 0.02  #djustment for a more aggressive driving style
        elif driving_style == 3:
            crash_probability -= 0.02  #adjustment for a more cautious driving style
    
        return crash_probability
