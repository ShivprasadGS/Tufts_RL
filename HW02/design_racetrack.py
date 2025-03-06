import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# starting and ending grid value
STARTING = 0.8
ENDING = 0.4

# build racetrack 'a'
def build_track_a(save_map=False):
    track = np.ones(shape=(32, 17))

    # out of the track values
    track[14:, 0] = 0
    track[22:, 1] = 0
    track[29:, 2] = 0
    track[:4, 0] = 0
    track[:3, 1] = 0
    track[0, 2] = 0
    track[6:, 9:] = 0
    track[6, 9] = 1

    # start line
    track[-1, 3:9] = STARTING

    # end line
    track[:6, -1] = ENDING

    # save the track
    if save_map:
        with open('./racetrack_designs/track_a.npy', 'wb') as f:
            np.save(f, track)

    return track

# build racetrack 'b'
def build_track_b(save_map=False):
    track = np.ones(shape=(30, 32))

    # out of the track values
    for i in range(14):
        track[:(-3 - i), i] = 0
    track[3:7, 11] = 1
    track[2:8, 12] = 1
    track[1:9, 13] = 1
    track[0, 14:16] = 0
    track[-17:, -9:] = 0

    track[12, -8:] = 0
    track[11, -6:] = 0
    track[10, -5:] = 0
    track[9, -2:] = 0

    # start line
    track[-1, :23] = STARTING

    # end line
    track[:9, -1] = ENDING

    # save the track
    if save_map:
        with open('./racetrack_designs/track_b.npy', 'wb') as f:
            np.save(f, track)

    return track

# executes this when this file is called
if __name__ == '__main__':

    track_a = build_track_a(save_map=True)
    with open('./racetrack_designs/track_a.npy', 'rb') as f:
        track_a_data = np.load(f)
    plt.figure(figsize=(10, 10))
    plt.imshow(track_a_data)
    sns.heatmap(track_a_data, linewidths=1)
    plt.savefig(f'./plots/track_a_design.png')

    track_b = build_track_b(save_map=True)
    with open('./racetrack_designs/track_b.npy', 'rb') as f:
        track_b_data = np.load(f)
    plt.figure(figsize=(10, 10))
    plt.imshow(track_b_data)
    sns.heatmap(track_b_data, linewidths=1)
    plt.savefig(f'./plots/track_b_design.png')


    plt.show()
