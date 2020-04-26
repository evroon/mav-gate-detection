# Detection of gates in MAV races
This project entails the visual detection of gates found in drone races.
It is part of the course `AE4317 - Autonomous Flight of Micro Air Vehicles` of TU Delft.

## Usage
The Python dependencies can be installed using:

```
pip3 install numpy opencv-python imageio matplotlib pillow
```

A dataset containing images must be put in the root folder in a directory called `WashingtonOBRace`. The images must be called `gate_*.png` and `mask_*.png` for the gates and masks, respectively.
The output images will be stored in the `results` directory.
