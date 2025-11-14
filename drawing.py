import pygame
import pygame.freetype
import numpy as np
import testing
import main
import time
import os
import preprocess_data as ppdt
import torch
from torchvision import transforms
import tkinter as tk
from tkinter import messagebox

#setup

def rendering_text(x,y,posx,posy):
    font = pygame.freetype.SysFont('Arial', y)
    text, rect = font.render(x,(255, 255, 255))
    screen.blit(text, (posx, posy))
    
def button(size, pos, file, file2):
    if pygame.Rect(pos[0], pos[1], size[0], size[1]).collidepoint(mouse_pos):
        screen.blit(file2, pos)
        if mouse_buttons[0]:
            return True
    else:
        screen.blit(file, pos)
    return False

Popup = messagebox.askyesno("Import New Data", "Do you want to import new data?")
if Popup:
    os.startfile("raw_training_data")
    time.sleep(1)
    messagebox.showinfo("", "Finished?")
    for file in os.listdir("processed_training_data"):
        file_path = os.path.join("processed_training_data", file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    ppdt.preprocess_raw_data()
    main.Main()

pygame.init()
screenx, screeny = 1280, 720
screen = pygame.display.set_mode((screenx, screeny))
bg_color = "#1D2547"
screen.fill(bg_color)
ai_canva = pygame.Rect(340, 150, 420, 420)
canva = pygame.Rect(810, 150, 420, 420)
pygame.draw.rect(screen, "white", canva)
pygame.draw.rect(screen, "white", ai_canva)

rendering_text('MY GUESS',36,95,250)
rendering_text('Canva',26,990,580)
rendering_text("AI's view",26,500,580)

mouse_last_pos = (0.0, 0.0)
mouse_down = False

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
    
    #store mouse pos(s)
    mouse_pos_x = mouse_pos[0] - 810
    mouse_pos_y = mouse_pos[1] - 150
    mouse_last_pos_x = mouse_last_pos[0] - 810
    mouse_last_pos_y = mouse_last_pos[1] - 150
    
    #clone drawing
    pygame.Surface.blit(screen.subsurface(ai_canva), pygame.transform.scale(save_image, (420, 420)), (0, 0))
    
    
    #draw results
    arranged_result = []
    if (mouse_down) & (canva.collidepoint(mouse_pos)):
        results = testing.get_answer(grayscale_array)
        for result in results:
            breakdown_result = result.split(': ')
            arranged_result.append([-1*float(breakdown_result[1][:-1:]),breakdown_result[0]])
        arranged_result = sorted(arranged_result)
        
        # Clear the text area
        pygame.draw.rect(screen, bg_color, (50, 300, 290, 50 * len(results)))

        # Draw new results
        for i, result in enumerate(arranged_result):
            rendering_text(result[1]+': '+str(-1*result[0])+'%',36 - round(0.5 * len(result[1])),50,300 + 50 * i)
            if i > 3 :break
    
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
    if button((124,52), (108, 160), pygame.image.load('./assets/erase.png'), pygame.image.load('./assets/erase2.png')):
        pygame.draw.rect(screen.subsurface(canva), "white", (0, 0, screenx, screeny))
        pygame.draw.rect(screen, bg_color, (50, 300, 290, 50 * len(results)))
        
    #update screen
    pygame.display.flip() 
    mouse_last_pos = pygame.mouse.get_pos()
    
pygame.quit()