# CursorGlide-Smooth-Mouse-Movement-via-Hand-Gestures
The hand gesture controlled mouse project allows users to control the cursor using hand movements captured by a webcam. It is developed using Python, OpenCV, and MediaPipe.

This project offers a touch-free interaction with the computer, enhancing accessibility and hygiene. It uses the webcam to continuously capture video frames.

MediaPipe processes these frames to detect hand landmarks in real time. The position of the index finger is tracked to control the movement of the mouse cursor.

When only the index finger is up, the system enters "movement mode," allowing users to move the cursor by moving their finger.

When both the index and middle fingers are up and brought close together, the system interprets this as a click gesture.

To avoid shaky movement, a smoothing algorithm is used to stabilize the cursor.

The system also considers a frame reduction margin, ensuring movements are restricted to a defined area for precision.

The project is implemented using a custom hand tracking module with functions to detect finger states and calculate distances.

This project can be helpful for people with mobility impairments, or in environments where physical contact with devices must be minimized.

It serves as a foundation for developing gesture-controlled user interfaces in smart systems, robotics, and gaming.

The application is easy to set up and runs on most modern laptops or desktops with a webcam.

It combines real-time computer vision, machine learning, and human-computer interaction in a practical way.

Overall, it provides a unique and interactive way to control the computer without a physical mouse.
