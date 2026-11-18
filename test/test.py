import numpy as np
W, H = 60, 30                      # characters are ~2x taller than wide
chars = " .:-=+*#%@"               # dark -> bright
light = np.array([-1, 1, 1]) / np.sqrt(3)

for row in range(H):
    line = ""
    for col in range(W):
        x = (col - W/2) / (W/2)    # map the cell to -1..1
        y = (H/2 - row) / (H/2)
        if x*x + y*y > 1:          # outside the circle -> blank
            line += " "; continue
        z = np.sqrt(1 - x*x - y*y) # height of the sphere at this point
        normal = np.array([x, y, z])          # on a sphere, normal = position
        b = max(0, normal @ light)            # dot product = brightness
        line += chars[int(b * (len(chars) - 1))]
    print(line)
