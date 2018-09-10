#########################################
# KAIST Summer Session 2018             #
# Deep Q-Network for Self-Driving Car   #
#########################################


import numpy as np
import random
import time

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.config import Config
from kivy.graphics import Color, Line
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

from DQN import DQN
Config.set('input', 'mouse', 'mouse, multitouch_on_demand')




# Instantiate our DQN which contains our neural network that represents our Q-function
# We will choose two types of actions: rotation and velocity of self-driving car
# We add several signals to provide more information on environments

dqn_rotation = DQN(9, 5, 0.95, 0.005)                   # 9 signals, 5 actions, gamma = 0.95, epsilon = 0.001
action2rotation = [0, 15, -15, 40, -40]                 # action for rotation (degree)

dqn_velocity = DQN(9, 3, 0.95, 0.005)                   # 9 signals, 3 actions, gamma = 0.95, epsilon = 0.001
action2velocity = [1, 5, 10]                            # action for velocity



# Initializing the map
first_update = True                     # using this trick to initialize the map only once
def init():
    global sand                         # sand is an array that has as many cells as our graphic interface has pixels. Each cell has a one if there is sand, 0 otherwise.
    global goal_x                       # x-coordinate of the goal (where the car has to go, that is the up-left corner or the bot-right corner)
    global goal_y                       # y-coordinate of the goal (where the car has to go, that is the up-left corner or the bot-right corner)
    global first_update
    global start
    global reached
    global last_reward
    global last_velocity
    global last_onsand
    global last_distance
    global last_signal1
    global last_signal2
    global last_signal3
    global BOUNDARY
    global GOAL
    
    last_signal1 = 0
    last_signal2 = 0
    last_signal3 = 0

    reached = 0
    start = time.time()
    sand = np.zeros((RIGHT, TOP))       # initializing the sand array with only zeros
    goal_x = 0                         # the goal to reach is at the upper left of the map (the x-coordinate is 20 and not 0 because the car gets bad reward if it touches the wall)
    goal_y = 0                  # the goal to reach is at the upper left of the map (y-coordinate)
    first_update = False                # trick to initialize the map only once
    last_reward = 0
    last_distance = 0
    last_velocity = 1
    last_onsand = 0
    GOAL = 'none'
    BOUNDARY = 20



# Creating the car class

class Car(Widget):
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)       # detecting if there is any sand in front of the car
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)       # detecting if there is any sand at the left of the car
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)       # detecting if there is any sand at the right of the car
    signal1 = NumericProperty(0)                                # the signal received by sensor1
    signal2 = NumericProperty(0)                                # the signal received by sensor2
    signal3 = NumericProperty(0)                                # the signal received by sensor3

    def move(self, rotation):
        """
        Allowing the car to go straightly, left of right
        self.pos: x = x + velocity * t
        self.rotation: new rotation
        self.angle: the angle between the x-axis and the axis of the direction of the car
        """
        
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        
        
        # Our car has three sensors        
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle + 30) % 360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle - 30) % 360) + self.pos
        
        
        """
        For each signal, we take
        (1) all the cells from -15 to +15 of the x coordinates of the sensor and
        (2) all the cells from -15 to +15 of the y coordinates of the sensor.
        Therefore we get the square of 30 x 30 pixels surrounding the sensor.
        Inside the square, we sum all the ones, because the cells contain 0 or 1.
        We divide it by 900 to get the density of any obstacles (which are called "sand" in this model) inside the square.
        """     
        
        self.signal1 = int(np.sum(sand[int(self.sensor1_x) - 15: int(self.sensor1_x) + 15, int(self.sensor1_y) - 15: int(self.sensor1_y) + 15]))/ 900.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x) - 15: int(self.sensor2_x) + 15, int(self.sensor2_y) - 15: int(self.sensor2_y) + 15]))/ 900.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x) - 15: int(self.sensor3_x) + 15, int(self.sensor3_y) - 15: int(self.sensor3_y) + 15]))/ 900.
        if self.sensor1_x > RIGHT - BOUNDARY or self.sensor1_x < BOUNDARY or self.sensor1_y > TOP - BOUNDARY or self.sensor1_y < BOUNDARY:
            self.signal1 = 1.                                                                                       # full density of sand, terrible reward
        if self.sensor2_x > RIGHT - BOUNDARY or self.sensor2_x < BOUNDARY or self.sensor2_y > TOP - BOUNDARY or self.sensor2_y < BOUNDARY:
            self.signal2 = 1.
        if self.sensor3_x > RIGHT - BOUNDARY or self.sensor3_x < BOUNDARY or self.sensor3_y > TOP - BOUNDARY or self.sensor3_y < BOUNDARY:
            self.signal3 = 1.

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass



# Creating the game class
class Game(Widget):    
    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):        
        self.car.center = self.center
        self.car.velocity = Vector(1, 0)        

    def update(self, dt):

        global dqn_rotation
        global dqn_velocity
        global last_reward
        global last_velocity
        global last_onsand
        global last_distance
        global last_signal1
        global last_signal2
        global last_signal3
        global goal_x
        global goal_y
        global RIGHT
        global TOP
        global GOAL
        global start
        global reached


        RIGHT = self.width
        TOP = self.height
        
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        if first_update:
            init() 
        
        if GOAL == 'none':
            return


        xx = goal_x - self.car.x
        yy = goal_y - self.car.y        
        orientation = (Vector(*self.car.velocity).angle((xx, yy)) + last_onsand) / 180.
        
        # The car uses 9 signals: 6 from three sensors (balls) + 3 from current car status
        # Three sensors (balls): 3 signals and 3 last signals
        # Current car status: 2 orientations (toward the goal) and 1 velocity       
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, last_signal1, last_signal2, last_signal3, orientation, -orientation, last_velocity/10]
        
        # Determining actions based on reward and signals (states)
        action_rotation = dqn_rotation.update(last_reward, last_signal)
        rotation = action2rotation[action_rotation]
        action_velocity = dqn_velocity.update(last_reward, last_signal)
        velocity = action2velocity[action_velocity]
        
        # Taking actions        
        self.car.move(rotation)
        self.car.velocity = Vector(velocity, 0).rotate(self.car.angle)
                     

        reward = 0
         
        # Reward if the car is approaching the goal
        distance = np.sqrt((self.car.x - goal_x) ** 2 + (self.car.y - goal_y) ** 2)
        reward += (last_distance - distance) / 10
         
        # Penalize if the car rotates (we prefer to keep going unless it is necessary)                
        if action_rotation == 1 or action_rotation == 2:
            reward += -0.1
        if action_rotation == 3 or action_rotation == 4:
            reward += -0.2
            
        # Penalize if the car hits obstacles (sands)
        # On the sands, the car may move quite randomly
        if sand[int(self.car.x), int(self.car.y)] > 0:
            reward += -10
            random_rotation = random.randint(-100, 100)
            random_velocity = random.randint(1, velocity)
            self.car.move(random_rotation)
            self.car.velocity = Vector(random_velocity, 0).rotate(self.car.angle)
            last_onsand = random_rotation
        else:
            last_onsand = 0
            
            
        # Penalize if the car hits obstacles (boundary)
        if self.car.x < BOUNDARY:
            self.car.x = BOUNDARY
            reward += -1
            
        if self.car.x > self.width - BOUNDARY:
            self.car.x = self.width - BOUNDARY
            reward += -1
               
        if self.car.y < BOUNDARY:
            self.car.y = BOUNDARY
            reward += -1
              
        if self.car.y > self.height - BOUNDARY:
            self.car.y = self.height - BOUNDARY
            reward += -1

        # Reaching the destination!
        if distance < 100:
            if reached == 1:
                pass
            elif GOAL == 'airport':
                now = time.time()
                reached = 1
                print('Reach Airport!')
                print('It takes %i seconds.' % (now-start))
                
            elif GOAL == 'downtown':
                now = time.time()
                reached = 1
                print('Reach Downtown!')
                print('It takes %i seconds.'% (now-start))
                
            elif GOAL == 'home':
                now = time.time()
                reached = 1
                print('Reach Home!')
                print('It takes %i seconds.'% (now-start))


        last_distance = distance
        last_reward = reward
        last_velocity = velocity
        last_signal1 = self.car.signal1
        last_signal2 = self.car.signal2
        last_signal3 = self.car.signal3



# Adding the painting tools
RAD = 20
class MyPaintWidget(Widget):
    def on_touch_down(self, touch):
        with self.canvas:
            Color(0.1, 0.5, 0.8)
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = RAD)
            sand[int(touch.x) - RAD: int(touch.x) + RAD, int(touch.y) - RAD: int(touch.y) + RAD] = 1

    def on_touch_move(self, touch):
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            touch.ud['line'].width = RAD
            sand[int(touch.x) - RAD : int(touch.x) + RAD, int(touch.y) - RAD : int(touch.y) + RAD] = 1


class CarApp(App):
    def build(self):     
        parent = Game()
        Clock.schedule_interval(parent.update, 1.5 / 60.0)
        self.painter = MyPaintWidget()
        clearButton = Button(text = 'CLEAR')
        downtownButton = Button(text = 'Go Downtown', pos = (parent.width, 0))
        homeButton = Button(text = 'Go Home', pos = (2*parent.width, 0))
        airportButton = Button(text = 'Go Airport', pos = (3*parent.width, 0))
        clearButton.bind(on_release = self.clear_canvas)
        downtownButton.bind(on_release = self.downtown)
        homeButton.bind(on_release = self.home)
        airportButton.bind(on_release = self.airport)
        parent.add_widget(self.painter)
        parent.add_widget(clearButton)
        parent.add_widget(downtownButton)
        parent.add_widget(homeButton)
        parent.add_widget(airportButton)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((RIGHT, TOP))

    def downtown(self, obj):
        global GOAL
        global goal_x
        global goal_y
        global RIGHT
        global TOP
        global start
        global reached
    
        reached = 0
        GOAL = 'downtown'        
        print('Let\'s go to Downtown.')                
        goal_x = RIGHT - 30
        goal_y = 30
        start = time.time()
        
    def home(self, obj):        
        global GOAL
        global goal_x
        global goal_y
        global RIGHT
        global TOP
        global start        
        global reached
    
        reached = 0  
        GOAL = 'home'
        print('Let\'s go to Home.')                
        goal_x = RIGHT - 30
        goal_y = TOP - 30
        start = time.time()
  
    def airport(self, obj):        
        global GOAL
        global goal_x
        global goal_y
        global RIGHT
        global TOP
        global start
        global reached
    
        reached = 0  
        GOAL = 'airport'
        print('Let\'s go to Airport.')                
        goal_x = 30
        goal_y = TOP - 30
        start = time.time()
            


# Running the self-driving car
if __name__ == '__main__':
    CarApp().run()