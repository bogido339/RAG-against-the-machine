import pygame

pygame.init()

screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("My Character")

character = pygame.image.load("character.png").convert_alpha()
character = pygame.transform.scale(character, (150, 150))

x = 300
y = 200
speed = 5

clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT]:
        x -= speed

    if keys[pygame.K_RIGHT]:
        x += speed

    if keys[pygame.K_UP]:
        y -= speed

    if keys[pygame.K_DOWN]:
        y += speed

    screen.fill("white")
    screen.blit(character, (x, y))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()