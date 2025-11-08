import pygame
import pygame.freetype
import numpy as np
import testing

#setup

def rendering_text(x,y,posx,posy):
    font = pygame.freetype.SysFont('Arial', y)
    text, rect = font.render(x,(255, 255, 255))
    screen.blit(text, (posx, posy))
    
pygame.init()
screenx, screeny = 1280, 720
screen = pygame.display.set_mode((screenx, screeny))
bg_color = "#1D2547"
screen.fill(bg_color)
ai_canva = pygame.Rect(340, 150, 420, 420)
canva = pygame.Rect(810, 150, 420, 420)
pygame.draw.rect(screen, "white", canva)
pygame.draw.rect(screen, "white", ai_canva)


rendering_text('My guess',36,50,250)
rendering_text('Canva',26,990,580)
rendering_text("AI's view",26,500,580)

mouse_last_pos = (0.0, 0.0)
mouse_down = False

Erase = pygame.image.load('./assets/erase.png')
Erase_size = (124,52)
Erase_hitbox = pygame.Rect(108, 150, Erase_size[0], Erase_size[1])

#loop
running = True

#closing
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
    #setup in loop
    mouse_pos = pygame.mouse.get_pos()
    mouse_buttons = pygame.mouse.get_pressed()
    keys = pygame.key.get_pressed()
    
    #collect data
    save_image = screen.subsurface(canva)
    save_image = pygame.transform.smoothscale(save_image, (28, 28))
    drawing_data = pygame.surfarray.array3d(save_image)
    grayscale_array = np.round( # chatgpt
        0.299 * drawing_data[:, :, 0] +
        0.587 * drawing_data[:, :, 1] +
        0.114 * drawing_data[:, :, 2]
    ).astype(np.uint8)
    
    #clone drawing
    pygame.Surface.blit(screen.subsurface(ai_canva), pygame.transform.scale(save_image, (420, 420)), (0, 0))
    
    if mouse_down:
        results = testing.get_answer(grayscale_array)
        # Clear the text area
        pygame.draw.rect(screen, bg_color, (50, 300, 200, 50 * len(results)))

        # Draw new results
        for i, result in enumerate(results):
            rendering_text(result,36,50,300 + 50 * i)

    #draw UI
    screen.blit(Erase, (108, 150))
    
    #store mouse pos(s)
    mouse_pos_x = mouse_pos[0] - 810
    mouse_pos_y = mouse_pos[1] - 150
    mouse_last_pos_x = mouse_last_pos[0] - 810
    mouse_last_pos_y = mouse_last_pos[1] - 150
    
    #draw
    if (mouse_buttons[0]):
        
        if mouse_down == False:
            pygame.draw.circle(screen.subsurface(canva), "black", (mouse_pos_x, mouse_pos_y), 10)
        
        pygame.draw.line(screen.subsurface(canva), "black", (mouse_last_pos_x, mouse_last_pos_y), (mouse_pos_x, mouse_pos_y), 20)
        pygame.draw.circle(screen.subsurface(canva), "black", (mouse_pos_x, mouse_pos_y), 8)
        mouse_down = True
    else:
        mouse_down = False
    
    #erase
    if Erase_hitbox.collidepoint(mouse_pos):
        Erase = pygame.image.load('./assets/erase2.png')
        if mouse_buttons[0]:
            pygame.draw.rect(screen.subsurface(canva), "white", (0, 0, screenx, screeny))
    else:
        Erase = pygame.image.load('./assets/erase.png')
    
    #update screen
    pygame.display.flip() 
    mouse_last_pos = pygame.mouse.get_pos()
    
pygame.quit()