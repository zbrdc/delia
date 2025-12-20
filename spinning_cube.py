import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

import math

def draw_cube():
    # Define the vertices of the cube
    vertices = (
        (1, -1, -1),
        (1, 1, -1),
        (-1, 1, -1),
        (-1, -1, -1),
        (1, -1, 1),
        (1, 1, 1),
        (-1, 1, 1),
        (-1, -1, 1)
    )

    # Define the edges of the cube
    edges = (
        (0, 1),
        (0, 3),
        (0, 4),
        (2, 1),
        (2, 3),
        (2, 6),
        (5, 1),
        (5, 4),
        (5, 6),
        (7, 3),
        (7, 4),
        (7, 6)
    )

    # Define the faces of the cube
    faces = (
        (0, 1, 2, 3),
        (3, 2, 6, 7),
        (7, 6, 5, 4),
        (4, 5, 1, 0),
        (1, 5, 6, 2),
        (4, 0, 3, 7)
    )

    # Define colors for each face
    face_colors = (
        (1, 0, 0),  # Red
        (0, 1, 0),  # Green
        (0, 0, 1),  # Blue
        (1, 1, 0),  # Yellow
        (1, 0, 1),  # Magenta
        (0, 1, 1)   # Cyan
    )

    # Draw the faces with colors
    glBegin(GL_QUADS)
    for i, face in enumerate(faces):
        glColor3fv(face_colors[i])
        for vertex in face:
            glVertex3fv(vertices[vertex])
    glEnd()

    # Draw the edges
    glColor3f(0.0, 0.0, 0.0)  # Black edges
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()


def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    # Set up the perspective
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

    # Enable depth testing
    glEnable(GL_DEPTH_TEST)

    # Rotation variables
    rotation_x = 0
    rotation_y = 0

    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Rotate the cube
        rotation_x += 1
        rotation_y += 1
        glRotatef(1, rotation_x, rotation_y, 0)

        # Draw the cube
        draw_cube()

        # Update the display
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()