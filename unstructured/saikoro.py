import numpy as np

# def dice():
#     direction = np.random.choice(["left", "right"])
#     x = np.random.uniform(0, 7)
#     y = np.random.uniform(0, 7)
#     yaw = np.random.randint(-30, 31)

#     return f"{direction}, {x:.1f}, {y:.1f}, {yaw}"


def dice():
    x = np.random.uniform(-7, 7)
    y = np.random.uniform(0, 10)
    yaw = np.random.randint(-30, 31)

    return f"{x:.0f}, {y:.0f}, {yaw}"


print(dice())
