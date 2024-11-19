import pygame
import sys
import math
import numpy as np
import tensorflow as tf
import time

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1200, 800
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Racing Game")

# Load images
try:
    finish_image = pygame.image.load("finish.png")
    finish_image = pygame.transform.scale(finish_image, (160, 32))
    track_image = pygame.image.load("track1.png")
    walls_image = pygame.image.load("walls1.png")
    walls_mask = pygame.mask.from_surface(walls_image)
    car_image = pygame.image.load("car.png")
    car_mask = pygame.mask.from_surface(car_image)
except pygame.error as e:
    print(f"Unable to load image: {e}")
    pygame.quit()
    sys.exit()

# Parameters for reinforcement learning
num_actions = 2  # Left, right (forward is handled separately)
state_size = 10  # Number of rays as input

# Reinforcement learning hyperparameters
gamma = 0.99  # Discount factor
target_speed = 2  # Target speed in pixels/second

# Variables to store episode data for training
episode_states = []
episode_actions = []
episode_rewards = []

# Define checkpoints (you need to adjust these to match your track layout)
checkpoints = [(320, 300), (720, 130), (1080, 550), (720, 680), (280, 680), (120, 450)]
num_checkpoints = len(checkpoints)

# Store which checkpoint the car is currently at
current_checkpoint = 0
previous_checkpoint = 0

def is_near_checkpoint(checkpoint_x, checkpoint_y, radius=160):
    # Calculate the distance from the car to the checkpoint
    distance = math.hypot(car_rect.centerx - checkpoint_x, car_rect.centery - checkpoint_y)
    return distance < radius  # The car is considered to have passed the checkpoint if it's within 'radius'

# Define the policy network (AI model)
def create_policy_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(24, input_shape=(state_size,), activation='relu'),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(num_actions, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy')
    return model

# Create the AI policy model
policy_model = create_policy_model()

# Function to get an AI action
def get_ai_action(model, state, epsilon=0.3):
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)  # Explore: Choose a random action
    else:
        state = np.array([state])  # Add batch dimension for model input
        action_probs = model.predict(state, verbose=0)[0]
        return np.random.choice(num_actions, p=action_probs)  # Exploit: Choose based on the policy

# Function to compute discounted rewards
def discount_rewards(rewards, gamma):
    discounted = np.zeros_like(rewards, dtype=np.float32)
    cumulative = 0.0
    for i in reversed(range(len(rewards))):
        cumulative = cumulative * gamma + rewards[i]
        discounted[i] = cumulative
    return discounted

# Car class representing the player vehicle
class Car:
    def __init__(self, image, start_x, start_y, speed):
        self.image = image
        self.x = start_x
        self.y = start_y
        self.speed = speed
        self.angle = 90
        self.width, self.height = image.get_size()

    def get_mask(self):
        return pygame.mask.from_surface(self.get_rotated_image())

    def rotate(self, angle_change):
        self.angle = (self.angle + angle_change) % 360

    def move_forward(self):
        new_x = self.x + self.speed * math.cos(math.radians(self.angle))
        new_y = self.y - self.speed * math.sin(math.radians(self.angle))
        return new_x, new_y

    def get_rotated_image(self):
        return pygame.transform.rotate(self.image, self.angle)

    def get_rect(self):
        rotated_image = self.get_rotated_image()
        rotated_rect = rotated_image.get_rect(center=(self.x, self.y))
        return rotated_rect

    # Draw the rays (sensors) and calculate the distances to the walls
    def draw_rays(self):
        num_rays = 10  # Number of rays
        ray_length = 2000  # Max length of rays
        spread_angle = 50  # Angle spread

        distances = []  # Store distances to the wall
        for i in range(num_rays):
            ray_angle = math.radians(self.angle + (i - num_rays // 2) * spread_angle / (num_rays - 1))
            end_x, end_y = self.cast_ray(ray_angle, ray_length)
            distance = math.hypot(end_x - self.x, end_y - self.y)  # Calculate distance of the ray
            distances.append(distance)

            # Draw the ray for visualization
            pygame.draw.line(screen, RED, (self.x, self.y), (end_x, end_y), 2)
        return distances

    # Cast a ray to detect walls
    def cast_ray(self, ray_angle, max_length):
        for length in range(max_length):
            end_x = self.x + length * math.cos(ray_angle)
            end_y = self.y - length * math.sin(ray_angle)

            if 0 <= end_x < WIDTH and 0 <= end_y < HEIGHT:
                mask_x = int(end_x) - walls_image.get_rect().left
                mask_y = int(end_y) - walls_image.get_rect().top

                if walls_mask.get_at((mask_x, mask_y)) == 1:  # Stop when ray hits a wall
                    return end_x, end_y

        return end_x, end_y
    
    def bounce(self):
        # Reverse the angle by 180 degrees to move in the opposite direction
        reverse_angle = (self.angle + 180) % 360

        # Move the car a few more pixels away to avoid getting stuck
        bounce_distance = 5  # You can adjust this value if needed

        # Move in the reverse direction by the bounce distance
        self.x += bounce_distance * math.cos(math.radians(reverse_angle))
        self.y -= bounce_distance * math.sin(math.radians(reverse_angle))

        # Ensure the car is moved out of the collision zone entirely by moving a bit further if necessary
        while walls_mask.overlap(car_mask, (int(self.x) - car_rect[0], int(self.y) - car_rect[1])):
            # Move slightly more until no collision is detected
            self.x += 1 * math.cos(math.radians(reverse_angle))
            self.y -= 1 * math.sin(math.radians(reverse_angle))

# Function to check collision between car and wall
def check_collision():
    tolerance = 2  # Set a tolerance value (e.g., 2 pixels)

    # Offset the car's rect slightly to check for early collision
    car_rect_with_tolerance = car_rect.inflate(-tolerance, -tolerance)  # Shrink the car's hitbox

    # Create a new mask with the adjusted rect
    car_mask_with_tolerance = pygame.mask.from_surface(car_image)

    # Check for overlap using the smaller hitbox
    overlap = walls_mask.overlap(car_mask_with_tolerance, (car_rect_with_tolerance.x, car_rect_with_tolerance.y))

    if overlap:
        return True  # Collision detected with tolerance
    return False  # No collision

# Function to calculate reward based on car performance
def calculate_reward(is_colliding, speed):
    if is_colliding:
        return -2  # Penalty for hitting a wall
    else:
        # Reward for not hitting a wall and bonus for speed
        speed_reward = speed - target_speed
        return speed_reward + 2 # Reward + speed bonus

# Set up the clock for framerate control
clock = pygame.time.Clock()

# Initialize the car object
car = Car(car_image, 155, 355, 5)

# Finish line position and rectangle
finish_pos = (50, 550)
finish_rect = pygame.Rect(finish_pos, finish_image.get_size())

# Initial game state
state = np.zeros(state_size)

# Initialize time tracking
start_time = time.time()

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Get the car's sensor data (ray distances)
    state = car.draw_rays()

    # AI chooses an action based on the state
    action = get_ai_action(policy_model, state)

    # Store state and action for training later
    episode_states.append(state)
    episode_actions.append(action)

    # Update car based on AI's action
    if action == 0:
        car.rotate(-3)  # Turn left
    elif action == 1:
        car.rotate(3)  # Turn right

    # Move the car forward
    start_x, start_y = car.x, car.y
    new_x, new_y = car.move_forward()
    car_rect = pygame.Rect(new_x - car.width // 2, new_y - car.height // 2, car.width, car.height)

    # Check for collision with walls
    is_colliding = check_collision()
    if not is_colliding:
        car.x, car.y = new_x, new_y

    # Check if the car reached the next checkpoint
    if is_near_checkpoint(checkpoints[current_checkpoint][0], checkpoints[current_checkpoint][1]):
        previous_checkpoint = current_checkpoint
        current_checkpoint += 1
        # If it reaches the last checkpoint, loop back to the start
        if current_checkpoint >= num_checkpoints:
            current_checkpoint = 0
        # Reward for moving forward
        reward = 5
    # Check if car is moving backwards (optional)
    elif current_checkpoint < previous_checkpoint:
        reward = -5  # Larger penalty for moving backwards
    else:
        # Penalize for not making progress (e.g., going off track or moving backwards)
        reward = -0.5

    # Calculate time spent in current state
    current_time = time.time()
    elapsed_time = current_time - start_time
    start_time = current_time

    # Calculate speed
    speed = math.hypot(car.x - start_x, car.y - start_y) / elapsed_time

    # Calculate reward
    reward = calculate_reward(is_colliding, speed)
    episode_rewards.append(reward)

    # Clear the screen and draw the track
    screen.blit(track_image, (0, 0))

    # Draw checkpoints
    for checkpoint in checkpoints:
        pygame.draw.rect(screen, BLUE, (*checkpoint, 20, 20))

    # Draw the car on the screen
    rotated_image = car.get_rotated_image()
    rotated_rect = car.get_rect()
    screen.blit(rotated_image, rotated_rect.topleft)

    screen.blit(finish_image, finish_pos)

    # Draw the car's sensor rays
    car.draw_rays()

    # Update the display
    pygame.display.flip()

    # Limit the frame rate to 60 FPS
    clock.tick(60)

    # For training, you'd need to use the episode_rewards, episode_states, and episode_actions
    # You can train the policy model here or save the data for training later.