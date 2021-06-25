import numpy as np
t1 = [240, 245, 270, 360, 1320, 2760, 4260, 8580]
t2 = [240, 245, 270, 360, 1320, 2760, 4260, 5700, 10020]
raw_measurements = [
    [# Person 1
        [
            t1,
            [340, 99, 44, 33, 6.3, 3.45, 2.1, 0.76],
            [2.8, 0.92, 0.17, 0.082, 0.055, 0.018]
        ],
        [
            t1,
            [632, 219, 116, 58, 12.9, 5.2, 3.5, 1.2],
            [5.7, 1.76, 0.36, 0.147, 0.106, 0.072]
        ]
    ],
    [# Person 2
        [
            t1,
            [345, 101, 76, 49, 6.3, 2.7, 1.7, 0.78],
            [3.0, 1.2, 0.15, 0.066, 0.051, 0.020]
        ],
        [
            t1,
            [699, 241, 120, 75, 11.4, 6.7, 4.1, 1.3],
            [8.8, 2.9, 0.36, 0.19, 0.12, 0.036]
        ]
    ],
    [# Person 3
        [
            t1,
            [294, 118, 83, 48, 5.2, 2.55, 1.3, 0.65],
            [3.2, 1.16, 0.115, 0.048, 0.035, 0.015]
        ],
        [
            t2,
            [569, 178, 103, 64, 11, 5.4, 3.0, 2.0, 0.82],
            [6.4, 2.36, 0.260, 0.177, 0.085, 0.085, 0.024]
        ]
    ],
    [# Person 4
        [
            t2,
            [329, 117, 65, 30, 7.2, 2.6, 1.62, 1.08, 0.24],
            [3.1, 1.3, 0.185, 0.068, 0.040, 0.037, 0.0065]
        ],
        [
            t2,
            [646, 249, 126, 101, 11, 5.4, 2.7, 2.1, 0.6],
            [6.0, 2.48, 0.36, 0.165, 0.071, 0.064, 0.018]
        ]
    ],
    [# Person 5
        [
            t2,
            [360, 93, 44, 21, 6.8, 2.4, 1.4, 0.96, 0.38],
            [2.8, 1.12, 0.14, 0.068, 0.047, 0.036, 0.014]
        ],
        [
            t1,
            [686, 108, 98, 65.5, 11.2, 6.2, 3.4, 1.4],
            [6.4, 2.96, 0.35, 0.19, 0.105, 0.05]
        ]
    ],
    [# Person 6
        [
            t2,
            [292, 64, 50, 23, 4.05, 2.5, 2.0, 1.45, 0.9],
            [2.6, 0.96, 0.105, 0.070, 0.051, 0.050, 0.025]
        ],
        [
            t2,
            [628, 193, 100, 56, 9.3, 6.0, 5.1, 3.2, 1.5],
            [6.0, 1.76, 0.245, 0.16, 0.12, 0.098, 0.052]
        ]
    ],
]

no_persons = len(raw_measurements)
no_experiments = 2

weights = []
for person in range(no_persons):
    raw_measurements[person] = raw_measurements[person][:no_experiments]
    weights.append([])
    for experiment in range(no_experiments):
        measurements = raw_measurements[person][experiment]
        te = measurements[0]
        cex = measurements[1]
        cven = measurements[2]
        cven = [cven[0], np.nan, np.nan] + cven[1:]
        cw = np.ones((2, len(te)))
        cw[0, 1:3] = 0
        pm = np.array([np.array(cven), 1e-3*np.array(cex)])
        if len(te) < len(t2):
            te = te + [te[-1]]
            cw = np.concatenate([cw.T, [[0,0]]]).T
            pm = np.concatenate([pm.T, [pm[:, -1]]]).T
        raw_measurements[person][experiment] = np.concatenate([
            [te], pm
        ]).T
        weights[-1].append(cw.T)


raw_measurements = np.array(raw_measurements)
weights = np.array(weights)
