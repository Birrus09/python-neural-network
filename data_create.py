import pygame
import numpy as np

pygame.init()
screen = pygame.display.set_mode((280, 280))
pygame.display.set_caption("data creation for AI training")

mouse_x, mouse_y = 0, 0

def simplify_image(screen):
    image = pygame.surfarray.array3d(screen)
    image = image[:, :, 0]  # Convert to grayscale
    image = image / 255.0  # Normalize pixel values to [0, 1]

    simplified_image = np.zeros((28, 28))
    for i in range(28):
        for j in range(28):
            x1, x2 = i * 10, (i + 1) * 10
            y1, y2 = j * 10, (j + 1) * 10
            square = image[x1:x2, y1:y2]
            simplified_image[i, j] = np.mean(square)

    return simplified_image


run = True

image_data = []
classes_data = []

while run:
    pygame.time.delay(10)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        elif event.type == pygame.MOUSEMOTION:
            mouse_x, mouse_y = event.pos

    
    key = pygame.key.get_pressed()
    if key[pygame.K_SPACE]:
        pygame.draw.circle(screen, (255, 255, 255), (mouse_x, mouse_y), 15)

    if key[pygame.K_1]:

        image_data.append(simplify_image(screen))
        
        classes_data.append(int(input("insert class (number drawn + 1): ")))

        screen.fill((0, 0, 0))
        

    if key[pygame.K_2]:
        file_name = str(input("save data as (write the name without extensions): "))
        np.save(file_name + ".npy", image_data)
        print(f"data saved as {file_name}")
        Class = str(input("insert the name of the class file: "))
        np.save = (Class + ".npy", classes_data)
        print(f"data saved as {Class}")

    
    pygame.display.update()

