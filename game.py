import cv2
import numpy as np
import random
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to draw the goal post
def draw_goal(frame):
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (50, 100), (width-50, height-50), (255, 255, 255), 5)

# Function to draw a soccer ball
def draw_ball(frame, ball_position):
    cv2.circle(frame, ball_position, 20, (0, 0, 255), -1)
    cv2.circle(frame, (ball_position[0], ball_position[1] + 25), 10, (0, 0, 0), -1)  # Shadow effect

# Function to update ball position
def update_ball_position(ball_position, ball_velocity):
    return (ball_position[0] + ball_velocity[0], ball_position[1] + ball_velocity[1])

# Function to detect if the ball is blocked
def is_ball_blocked(ball_position, hand_position, block_radius=50):
    return np.linalg.norm(np.array(ball_position) - np.array(hand_position)) < block_radius

# Function to draw particle effects
def draw_particles(frame, particles):
    for particle in particles:
        cv2.circle(frame, particle['pos'], particle['size'], particle['color'], -1)
        particle['pos'] = (particle['pos'][0] + particle['vel'][0], particle['pos'][1] + particle['vel'][1])
        particle['size'] = max(0, particle['size'] - 1)

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Initialize variables
ball_position = (random.randint(100, 540), 50)
ball_velocity = (random.randint(-3, 3), 5)
blocks = 0
misses = 0
particles = []

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Mirror the frame
    if not ret:
        break

    # Draw the goal post
    draw_goal(frame)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    hand_position = None
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Get the coordinates of the wrist (landmark 0)
            hand_position = (int(hand_landmarks.landmark[0].x * frame.shape[1]),
                             int(hand_landmarks.landmark[0].y * frame.shape[0]))

    # Draw and update ball position
    draw_ball(frame, ball_position)
    ball_position = update_ball_position(ball_position, ball_velocity)

    # Check if the ball is blocked
    if hand_position and is_ball_blocked(ball_position, hand_position):
        blocks += 1
        # Generate particles
        for _ in range(30):
            particles.append({
                'pos': list(ball_position),
                'vel': [random.randint(-5, 5), random.randint(-5, 5)],
                'size': random.randint(5, 10),
                'color': (0, 255, 0)  # Green particles for block
            })
        ball_position = (random.randint(100, 540), 50)
        ball_velocity = (random.randint(-3, 3), 5)
    elif ball_position[1] > frame.shape[0]:
        misses += 1
        # Generate particles
        for _ in range(30):
            particles.append({
                'pos': list(ball_position),
                'vel': [random.randint(-5, 5), random.randint(-5, 5)],
                'size': random.randint(5, 10),
                'color': (0, 0, 255)  # Red particles for miss
            })
        ball_position = (random.randint(100, 540), 50)
        ball_velocity = (random.randint(-3, 3), 5)

    # Draw particles
    draw_particles(frame, particles)
    particles = [p for p in particles if p['size'] > 0]

    # Display the score
    cv2.putText(frame, f'Kills: {blocks}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Misses: {misses}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow('Bug hunt', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
