import pygame
import sys
import math
def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()
    angle = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        screen.fill((0, 0, 0))
        ball_color = (abs(int(128 + 127 * math.sin(angle))), abs(int(128 + 127 * math.cos(angle))), abs(int(128 + 127 * math.sin(2 * angle))))
        pygame.draw.circle(screen, ball_color, (400, 300), 50)
        angle += 0.05
        pygame.display.flip()
        clock.tick(60)
if __name__ == '__main__':
    main()