import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

np.set_printoptions(precision=3)
        
        
class Car:
    def __init__(self, tyre="Intermediate"):
        self.default_tyre = tyre
        self.possible_tyres = ["Ultrasoft", "Soft", "Intermediate", "Fullwet"]
        self.pitstop_time = 23
        self.fuel_capacity = 100.0  # Maximum fuel capacity  
        self.fuel_consumption_rate = 0.01  # Fuel consumption rate per unit distance
        self.refill = 20
        self.combine = 40
        self.reset()    
    
    def reset(self):
        self.change_tyre(self.default_tyre)
        self.fuel_level = self.fuel_capacity
    
    def degrade(self, w, r):
        if self.tyre == "Ultrasoft":
            self.condition *= (1 - 0.0050*w - (2500-r)/90000)
        elif self.tyre == "Soft":
            self.condition *= (1 - 0.0051*w - (2500-r)/93000)
        elif self.tyre == "Intermediate":
            self.condition *= (1 - 0.0052*abs(0.5-w) - (2500-r)/95000)
        elif self.tyre == "Fullwet":
            self.condition *= (1 - 0.0053*(1-w) - (2500-r)/97000)
        
        
    def change_tyre(self, new_tyre):
        assert new_tyre in self.possible_tyres
        self.tyre = new_tyre
        self.condition = 1.00
    
    
    def get_velocity(self, driving_style):
        if self.tyre == "Ultrasoft":
            vel = 80.7
        elif self.tyre == "Soft":
            vel = 80.1
        elif self.tyre == "Intermediate":
            vel = 79.5
        elif self.tyre == "Fullwet":
            vel = 79.0
            
         # Adjust velocity based on fuel level and driving style
        if driving_style == 1:
            velocity = vel * (0.8 * self.condition ** 1.5) * ((self.fuel_capacity / self.fuel_level)/10)
        elif driving_style == 2:
            velocity = vel * (0.6 * self.condition ** 1.5) * ((self.fuel_capacity / self.fuel_level)/10)
        elif driving_style == 3:
            velocity = vel * (0.4 * self.condition ** 1.5) * ((self.fuel_capacity / self.fuel_level)/10)  
            
        return velocity
    

class Track:
    def __init__(self, car=Car(), driving_style=1):
        # self.radius and self.cur_weather are defined in self.reset()
        self.total_laps = 162
        self.car = car
        self.driving_style = driving_style 
        self.possible_driving_styles = [1, 2, 3]
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
        self.crash_probability = 0.001
        self.reset()
    
    
    def reset(self):
        self.radius = np.random.randint(600,1201)
        self.cur_weather = np.random.choice(self.possible_weather)
        self.is_done = False
        self.pitstop = False
        self.fill = False
        self.combine = False
        self.laps_cleared = 0
        self.car.reset()
        return self._get_state()
    
    
    def _get_state(self):
        return [self.car.tyre, self.car.condition, self.cur_weather, self.radius, self.laps_cleared]
    
    def calculate_crash_probability(self, velocity, driving_style):
        # Calculate crash probability based on velocity, wetness, and driving style
        crash_probability = 0.01 * (velocity - 80) + 0.005 * self.wetness[self.cur_weather] + self.laps_cleared * 0.0000001
    
        # Adjust the crash probability based on driving style
        if driving_style == 2:
            crash_probability *= 1.002  #djustment for a more aggressive driving style
        elif driving_style == 3:
            crash_probability *= 1.002  #adjustment for a more cautious driving style
    
        return crash_probability
        
    
    def transition(self, action=0):
        """
        Args:
            action (int):
                0. Make a pitstop and fit new ‘Ultrasoft’ tyres withot fill
                1. Make a pitstop and fit new ‘Soft’ tyres withot fill
                2. Make a pitstop and fit new ‘Intermediate’ tyres withot fill
                3. Make a pitstop and fit new ‘Fullwet’ tyres withot fill
                4. Continue the next lap without changing tyres and withot fill
                5. Continue the next lap without changing tyres but fill
                6. Make a pitstop and fit new ‘Ultrasoft’ tyres and fill
                7. Make a pitstop and fit new ‘Soft’ tyres and fill
                8. Make a pitstop and fit new ‘Intermediate’ tyres and fill
                9. Make a pitstop and fit new ‘Fullwet’ tyres and fill
        """
        ## Pitstop time will be added on the first eight of the subsequent lap
        time_taken = 0
        if self.laps_cleared == int(self.laps_cleared):
            if self.pitstop:
                self.car.change_tyre(self.committed_tyre)
                time_taken += self.car.pitstop_time
                self.pitstop = False
            if self.fill:
                self.car.fuel_level = self.car.fuel_capacity
                time_taken += self.car.refill
                self.fill = False
            if self.combine:
                self.car.change_tyre(self.committed_tyre)
                self.car.fuel_level = self.car.fuel_capacity
                time_taken += self.car.combine
                self.combine = False
                
                
        ## The environment is coded such that only an action taken at the start of the three-quarters mark of each lap matters
        if self.laps_cleared - int(self.laps_cleared) == 0.75:
            if action < 4:
                self.pitstop = True
                self.fill = False
                self.combine = False
                self.committed_tyre = self.car.possible_tyres[action]
            elif action == 4:
                self.pitstop = False
                self.fill = False
                self.combine = False
            elif action == 5:
                self.pitstop = False
                self.fill = True
                self.combine = False
            else:
                self.pitstop = False
                self.fill = False
                self.combine = True
                self.committed_tyre = self.car.possible_tyres[action-6]              
                
        # Calculate distance traveled and velocity
        velocity = self.car.get_velocity(self.driving_style)  # Use the predefined driving style
        distance = (2 * np.pi * self.radius / 8)
        
        
                 # Check for crash event
        if np.random.random() < self.crash_probability:
            reward = -1000000
            return reward, 'crash' , True, 0  # Implement a massive negative reward for crashing
        
        # Add a check to avoid division by zero
        if velocity > 0:
            time_taken += distance / velocity
        else:
            reward = -1000000  # You can choose a default time increment when velocity is zero or negative
            return reward, 'stop' , True, 0

        # Calculate crash probability based on velocity and wetness
        self.crash_probability = self.calculate_crash_probability(velocity, self.driving_style)

        # Adjust crash probability based on driving style and fuel level
        if self.driving_style == 2:
            self.crash_probability += 0.02  # Example adjustment for a more aggressive driving style
        elif self.driving_style == 3:
            self.crash_probability -= 0.02  # Example adjustment for a more cautious driving style
            
       
        # Apply degradation and fuel consumption
        self.car.degrade(w=self.wetness[self.cur_weather], r=self.radius)
        fuel_consumed = distance * self.car.fuel_consumption_rate 
        self.car.fuel_level -= fuel_consumed        
        
        reward = 0 - time_taken
        self.laps_cleared += 0.125
        
        # Check if the race is finished
        if self.laps_cleared == self.total_laps:
            self.is_done = True

        if np.random.rand() < 0.05:
            self.driving_style = np.random.choice(self.possible_driving_styles)

        
        next_state = self._get_state()
        return reward, next_state, self.is_done, velocity
