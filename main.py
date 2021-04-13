import time
from itertools import cycle
import pygame
from pygame import gfxdraw, surfarray
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.colors as clr

s = 1


@njit(fastmath=True)
def f(z, c):
    return z * z + c


@njit(fastmath=True)
def mandelbrot(x, y, scale, maxiter):
    pic = np.zeros((1000, 1000, 3), dtype=np.float64)
    for i in range(1000):
        for j in range(1000):
            i0 = (i / scale + x) / 250 - 2
            j0 = (j / scale + y) / 250 - 2

            c = (i0 + j0 * 1j)

            z = 0

            for depth in range(maxiter):
                z = f(z, c)
                if abs(z) > 100:
                    break

            c = depth * (255 / maxiter).real
            pic[i, j] = np.array((c, c, c), dtype=np.float64)

    return pic


@njit(fastmath=True)
def mandelbrot_test(x, y, scale, maxiter):
    pic = np.zeros((1000, 1000))
    for i in range(1000):
        for j in range(1000):
            i0 = (i / scale + x) / 250 - 2
            j0 = (j / scale + y) / 250 - 2
            c = (i0 + j0 * 1j)
            z = 0
            for depth in range(maxiter):
                z = f(z, c)
                if abs(z) > 100:
                    break
            pic[i, j] = depth / maxiter
    return pic


def mandelbrot_draw(x, y, scale):
    global s
    pygame.init()
    win = pygame.display.set_mode((1000, 1000))
    pic = mandelbrot(0, 0, 0.5, 1000)
    surfarray.blit_array(win, pic)
    pygame.display.update()
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.MOUSEBUTTONUP:
                maxiter = (s // 10 + 1) * 1000
                x0, y0 = ev.pos
                s += 1
                x += (x0 - 100) / scale
                y += (y0 - 250) / scale
                scale *= 2
                t = time.time()
                pic = mandelbrot(x, y, scale, maxiter)
                print(f'{time.time() - t}\n{x, y, scale, maxiter}\n')
                surfarray.blit_array(win, pic)
                pygame.display.update()
            if ev.type == pygame.KEYUP:
                draw(x, y, scale, maxiter)


def draw(x, y, scale, maxiter):
    color = [(1 - (1 - q) ** 4, c) for q, c in zip(np.linspace(0, 1, 20), cycle(['#ffff88', '#000000', '#ffaa00']))]
    cmap = clr.LinearSegmentedColormap.from_list('mycmap', color, N=2048)
    plt.figure(figsize=(100, 100))
    plt.xticks([])
    plt.yticks([])
    #
    image = -mandelbrot_test(x, y, scale, maxiter).T
    plt.imshow(image, cmap=cmap, interpolation='none')
    plt.savefig('temp.png')
    print('SAVED')


if __name__ == '__main__':
    mandelbrot_draw(0, 0, 0.5)
