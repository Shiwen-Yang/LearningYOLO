import time
import mss
import cv2
import numpy as np
import pyautogui as pag
import torch
import threading
from ultralytics import YOLO
from pynput import keyboard

model = YOLO("yolo_aim.pt").to("cuda")

class Point:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.coordinates = np.array([x, y])

    def __sub__(self, other) -> float:
        """Calculates euclidean distance between two Points."""
        return np.linalg.norm(self.coordinates - other.coordinates)

    def __repr__(self) -> str:
        return f"Point(x={self.x}, y={self.y})"
    
class Target:
    def __init__(self, target: torch.Tensor) -> None:

        self.x1: float = target[0].item()
        self.y1: float = target[1].item()
        self.x2: float = target[2].item()
        self.y2: float = target[3].item()
        
        self.w: float = np.abs(self.x2 - self.x1)
        self.h: float = np.abs(self.y2 - self.y1)
        
        self.confidence: float = target[4].item()
        self.label: int = int(target[5].item())
        self.tensor: torch.Tensor = target
        
        self.x = int((self.x1 + self.x2)/2)
        self.y = int(-self.h/3 + (self.y1 + self.y2)/2)
        self.coordinates = np.array([self.x, self.y])

    def __repr__(self) -> str:
        return f"Target({self.label}: x={self.x}, y={self.y})"
    
def get_targets(results: list) -> list[Target]:
    """Returns a list with results filtered by the classes specified in the labels argument."""
    return [Target(target) for target in results[0].boxes.data]

def sort_targets(targets: list) -> list[Target]:
    """ We shall prioritize targets that are both close, which makes the bbox big, and has high confidence """
    sorted_targets = sorted(targets, key=lambda target: ((target.w * target.h) * target.confidence), reverse=True)
    return sorted_targets

def shoot() -> None:
    pag.mouseDown()
    time.sleep(0.1)
    pag.mouseUp()
    


def aim():
    screen_width, screen_height = pag.size()
    monitor_area = {'top': 0, 'left': 0, 'width': screen_width, 'height': screen_height}
    
    with mss.mss() as sct:
        # Capture the screen
        sct_img = sct.grab(monitor_area)
        # Convert the captured image to a format suitable for OpenCV
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert BGRA to BGR
        # Adjust the gamma to fix brightness if needed
        
        # Perform object detection on the frame
        targets = get_targets(model(frame))
        if len(targets) > 0:
            target_shot = sort_targets(targets)[0]
            print(target_shot)
            pag.moveTo(target_shot.x, target_shot.y)
            shoot()
        time.sleep(0.01)
            
def on_press(key):
    try:
        # Replace 'f1' with the hotkey you want to use
        if key == keyboard.Key.f15:
            aim()
    except AttributeError:
        pass

def main():
    # Start a listener for the hotkey
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

if __name__ == "__main__":
    main()