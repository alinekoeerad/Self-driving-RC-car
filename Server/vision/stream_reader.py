# vision/stream_reader.py

import cv2
import threading
import time
from collections import deque

class VideoStreamReader:
    """
    A robust helper class to read frames from a video stream in a separate thread.
    Includes automatic reconnection logic to handle network interruptions.
    """
    def __init__(self, src=0):
        self.src = src
        self.stream = cv2.VideoCapture(self.src)
        self.queue = deque(maxlen=1)
        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        """Starts the thread to read frames from the video stream."""
        self.thread.start()
        return self

    def update(self):
        """The main loop for the reader thread, with reconnection logic."""
        while not self.stopped:
            # If the stream is not open, it means we are disconnected.
            if not self.stream.isOpened():
                print("⚠️ Stream disconnected. Attempting to reconnect...")
                self.reconnect()
            else:
                # Read the next frame from the stream
                ret, frame = self.stream.read()
                
                # If the frame was read successfully, add it to the queue
                if ret:
                    self.queue.append(frame)
                else:
                    # If read() fails, it signifies a disconnection.
                    print("❌ Failed to read frame. Connection likely lost.")
                    self.stream.release() # Release the broken stream object

    def reconnect(self):
        """Handles the reconnection logic."""
        reconnect_attempts = 0
        while not self.stopped:
            # Try to create a new VideoCapture object
            self.stream = cv2.VideoCapture(self.src)
            
            if self.stream.isOpened():
                print("✅ Reconnected to stream successfully!")
                # Break the reconnection loop and return to the main update loop
                return
            else:
                reconnect_attempts += 1
                print(f"Reconnect attempt {reconnect_attempts} failed. Retrying in 5 seconds...")
                # Release the failed object before the next attempt
                self.stream.release()
                time.sleep(5)

    def read(self):
        """Returns the most recent frame in the queue."""
        try:
            return self.queue[0]
        except IndexError:
            return None # Return None if the queue is empty

    def stop(self):
        """Signals the thread to stop."""
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join()
        if self.stream.isOpened():
            self.stream.release()