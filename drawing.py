import pygame
import numpy as np
import testing

#setup
pygame.init()
screenx, screeny = 960, 720
screen = pygame.display.set_mode((screenx, screeny), )
mouse_last_pos = (0.0, 0.0)
screen.fill("white")
font = pygame.font.SysFont('Arial', 36)
old_text = font.render('', True, "black")
Erase = pygame.image.load('./assets/erase.png')
Erase_size = (124,52)
Erase_hitbox = pygame.Rect(50, 150, Erase_size[0], Erase_size[1])
Mouse_down = False
UI_text = font.render('My guess', True, "black")
screen.blit(UI_text, (50, 250))

#AI's guess
Guess = ''

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
    save_image = screen.subsurface(pygame.Rect(240, 0, screenx - 240, screeny))
    save_image = pygame.transform.smoothscale(save_image, (28, 28))
    drawing_data = pygame.surfarray.array3d(save_image)
    grayscale_array = np.round( # chatgpt
        0.299 * drawing_data[:, :, 0] +
        0.587 * drawing_data[:, :, 1] +
        0.114 * drawing_data[:, :, 2]
    ).astype(np.uint8)
    results = testing.get_answer(grayscale_array)
    pygame.image.save(save_image, "./test.png")
    
    # Clear the text area
    pygame.draw.rect(screen, "white", (50, 300, 200, 50 * len(results)))

    # Draw new results
    for i, result in enumerate(results):
        text = font.render(result, False, "black")
        screen.blit(text, (50, 300 + 50 * i))
    
    #draw UI
    pygame.draw.line(screen, "black", (229, 0), (229, screeny), 21)
    screen.blit(Erase, (50, 150))
    
    #store mouse pos(s)
    mouse_pos_x = mouse_pos[0]
    mouse_pos_y = mouse_pos[1]
    mouse_last_pos_x = mouse_last_pos[0]
    mouse_last_pos_y = mouse_last_pos[1]
    
    #define an boundary
    if mouse_pos_x < 229:
        mouse_pos_x = 229
    if mouse_last_pos_x < 229:
        mouse_last_pos_x = 229
        

    
    #draw
    if (mouse_buttons[0]):
        
        if Mouse_down == False:
            pygame.draw.circle(screen, "black", (mouse_pos_x, mouse_pos_y), 10)
        
        pygame.draw.line(screen, "black", (mouse_last_pos_x, mouse_last_pos_y), (mouse_pos_x, mouse_pos_y), 20)
        pygame.draw.circle(screen, "black", (mouse_pos_x, mouse_pos_y), 8)
        Mouse_down = True
    else:
        Mouse_down = False
    
        
    #erase
    if Erase_hitbox.collidepoint(mouse_pos):
        Erase = pygame.image.load('./assets/erase2.png')
        if mouse_buttons[0]:
            pygame.draw.rect(screen, "white", (240, 0, screenx - 240, screeny))
    else:
        Erase = pygame.image.load('./assets/erase.png')
    
    #update screen
    pygame.display.flip() 
    
    mouse_last_pos = pygame.mouse.get_pos()
    old_text = font.render(str(Guess), False, "white")
       
pygame.quit()
