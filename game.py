import cvzone
import cv2
import numpy as np
import math
import random
from cvzone.HandTrackingModule import HandDetector

# Setup OpenCV capture and window size
capture = cv2.VideoCapture(0)
capture.set(3, 1280)
capture.set(4, 720)

detect = HandDetector(detectionCon=0.8, maxHands=1)


class SnakeGame:
    def __init__(self, foodPath):
        self.points = []  # Points of the snake
        self.length = []  # Distance between points
        self.currentLength = 0  # Total snake length
        self.TotalAllowedLength = 150  # Total allowed length
        self.headPrevious = 0, 0  # Previous head point.

        # Food initialization
        self.foodImg = cv2.imread(foodPath, cv2.IMREAD_UNCHANGED)
        if self.foodImg is None:
            raise ValueError(f"Unable to load food image from path: {foodPath}")
        self.foodHeight, self.foodWidth, _ = self.foodImg.shape
        self.foodLocation = 0, 0
        self.FoodLocationRandom()
        self.score = 0
        self.gameOver = False

    def FoodLocationRandom(self):
        """Randomize food location."""
        self.foodLocation = random.randint(100, 1000), random.randint(100, 600)

    def update(self, mainImg, headCurrent):
        """Update the game state."""
        if self.gameOver:
            cvzone.putTextRect(mainImg, "Game Over", [400, 200], scale=3, thickness=3, colorT=(255, 255, 255),
                               colorR=(0, 0, 255), offset=20)
            cvzone.putTextRect(mainImg, f'Your Score: {self.score}', [400, 300], scale=2, thickness=2,
                               colorT=(255, 255, 255), colorR=(0, 0, 255), offset=20)
            cvzone.putTextRect(mainImg, "Press 'R' to Restart", [400, 400], scale=2, thickness=2,
                               colorT=(255, 255, 255), colorR=(0, 0, 255), offset=20)
            cvzone.putTextRect(mainImg, "Press 'Q' to Quit", [400, 500], scale=2, thickness=2,
                               colorT=(255, 255, 255), colorR=(0, 0, 255), offset=20)
        else:
            # Add the current point
            previousX, previousY = self.headPrevious
            currentX, currentY = headCurrent
            self.points.append([currentX, currentY])
            distance = math.hypot(currentX - previousX, currentY - previousY)
            self.length.append(distance)
            self.currentLength += distance
            self.headPrevious = currentX, currentY

            # Keep the snake length in check
            if self.currentLength > self.TotalAllowedLength:
                for i, length in enumerate(self.length):
                    self.currentLength -= length
                    self.length.pop(i)
                    self.points.pop(i)
                    if self.currentLength < self.TotalAllowedLength:
                        break

            # Draw the snake
            if self.points:
                for i in range(1, len(self.points)):
                    cv2.line(mainImg, tuple(self.points[i - 1]), tuple(self.points[i]), (0, 0, 255), 20)
                cv2.circle(mainImg, tuple(self.points[-1]), 20, (200, 0, 200), cv2.FILLED)

            # Check collision with itself
            if len(self.points) > 2:
                pointArray = np.array(self.points[:-2], np.int32).reshape((-1, 1, 2))
                cv2.polylines(mainImg, [pointArray], False, (0, 200, 0), 3)
                minDist = cv2.pointPolygonTest(pointArray, headCurrent, True)
                if -1 <= minDist <= 1:
                    self.gameOver = True

            # Check collision with food
            foodX, foodY = self.foodLocation
            if foodX - self.foodWidth // 2 < currentX < foodX + self.foodWidth // 2 and \
               foodY - self.foodHeight // 2 < currentY < foodY + self.foodHeight // 2:
                self.FoodLocationRandom()
                self.score += 1
                self.TotalAllowedLength += 50  # Grow the snake

            # Draw the food
            mainImg = cvzone.overlayPNG(mainImg, self.foodImg,
                                        (foodX - self.foodWidth // 2, foodY - self.foodHeight // 2))

            # Display the score
            cvzone.putTextRect(mainImg, f'Score: {self.score}', [50, 80], scale=2, thickness=2, offset=10)

            # Display instructions
            cvzone.putTextRect(mainImg, "Press 'Q' to Quit, 'R' to Restart", [50, 50], scale=1, thickness=1, offset=5)
        return mainImg


game = SnakeGame("Donut.png")

while True:
    success, img = capture.read()
    img = cv2.flip(img, 1)
    hand, img = detect.findHands(img, flipType=False)

    if hand:
        landmarkList = hand[0]['lmList']
        pointIndex = landmarkList[8][0:2]
        img = game.update(img, pointIndex)

    # Display the frame
    cv2.imshow("Snake Game", img)

    # Handle key presses
    key = cv2.waitKey(1)
    if key == ord('r'):  # Restart the game
        game = SnakeGame("Donut.png")
    elif key == ord('q'):  # Quit the game
        break
