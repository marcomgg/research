from visdom import Visdom
import time
import numpy as np


def update_line(vz: Visdom, win, x, y, env='marco'):
    if np.size(x) == 1:
        x = np.array([x, x])
        y = np.array([y, y])
    else:
        x = np.array(x)
        y = np.array(y)

    if vz.win_exists(win, env):
        vz.line(
            X=np.array([x, x]),
            Y=np.array([y, y]),
            win=win,
            update='append',
            env = env
        )
        return win
    else:
        win = vz.line(
            X=np.array([x, x]),
            Y=np.array([y, y]),
            env=env
            )
        return win

def check_connection(vz: Visdom):
    startup_sec = 1
    while not vz.check_connection() and startup_sec > 0:
        time.sleep(0.1)
        startup_sec -= 0.1
    assert vz.check_connection(), 'No connection could be formed quickly'

