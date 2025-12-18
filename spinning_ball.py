#!/usr/bin/env python3

import pygame
from pygame.locals import *
import sys
import math
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

pygame.init()
clock = pygame.time.Clock()
screen = pygame.display.set_mode((800, 600))
font = pygame.font.Font(None, 36)

ball_radius = 50
ball_x, ball_y = screen.get_width() // 2, screen.get_height() // 2
angle = 0
speed = 1

def draw_ball(x, y, radius):
    for i in range(radius):
        color = colors[i % len(colors)]
        pygame.draw.circle(screen, color, (x, y), radius - i)

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    screen.fill((0, 0, 0))
    draw_ball(ball_x, ball_y, ball_radius)
    angle += speed
    ball_x = int(screen.get_width() // 2 + math.cos(angle) * (screen.get_width() // 4))
    ball_y = int(screen.get_height() // 2 + math.sin(angle) * (screen.get_height() // 4))
    pygame.display.flip()
    clock.tick(60)