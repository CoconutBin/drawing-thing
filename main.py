import pygame
import pygame.freetype
import numpy as np
import testing
import time
import os
import sys
import subprocess

import preprocess_data as ppdt
import tkinter as tk
from tkinter import messagebox
import training

#setup
preview_dataset = False
batch_size = 256
learning_rate = 1e-4
momentum = 0.9
num_epochs = 6
model_file_name = 'my_model.pth'

def open_file(path): # Chatgpt
    if sys.platform.startswith("win"):
        # Windows
        os.startfile(path)

    elif sys.platform.startswith("darwin"):
        # macOS
        subprocess.Popen(["open", path])

    elif sys.platform.startswith("linux"):
        # Linux (uses xdg-open)
        subprocess.Popen(["xdg-open", path])

    else:
        raise OSError("Unsupported operating system.")

def rendering_text(x,y,posx,posy):
    font = pygame.freetype.SysFont('Arial', y)
    text, rect = font.render(x,(255, 255, 255))
    screen.blit(text, (posx, posy))
    
def button(size, pos, file1, file2):
    if pygame.Rect(pos[0], pos[1], size[0], size[1]).collidepoint(mouse_pos):
        screen.blit(pygame.image.load(file2), pos)
        if mouse_buttons[0]:
            return True
    else:
        screen.blit(pygame.image.load(file1), pos)
    return False

def training_model():
    train_dataset, val_dataset, train_dataloader, val_dataloader, labels_dict = training.prepare_dataset_and_labels(batch_size=batch_size)

    model = training.make_new_model(num_categories=len(labels_dict))

    training.random_preview_model(model, val_dataset, labels_dict, title="Before Train")

    # Training
    training.train_and_save_model(model, num_epochs, learning_rate, momentum, batch_size, model_file_name, train_dataloader, val_dataloader, show_graph=True)

    training.random_preview_model(model, val_dataset, labels_dict, title="After Train")

# Popup windows
basefile = tk.Tk()
basefile.withdraw()
basefile.wm_attributes("-topmost", True) # make the pop up(s) stay on the top
if (os.path.exists(model_file_name)) & (os.listdir("processed_training_data") != []):
    Popup = messagebox.askyesno("Option", "Do you want to Import new data / Retrain the model?")
    basefile.withdraw()
    if Popup:
        open_file("raw_training_data")
        time.sleep(1)
        messagebox.showinfo("Importing Data", "Finished?")
        for file in os.listdir("processed_training_data"):
            file_path = os.path.join("processed_training_data", file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        ppdt.preprocess_raw_data()
        training_model()
else:
    open_file("raw_training_data")
    time.sleep(1)
    basefile.withdraw()
    messagebox.showinfo("Importing Data", "Finished?")
    for file in os.listdir("processed_training_data"):
        file_path = os.path.join("processed_training_data", file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    ppdt.preprocess_raw_data()
    training_model()

testing.preparing_stuff()

## Main window ##

# Setup
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
rendering_text('Canvas',26,990,580)
rendering_text("Model's view",26,500,580)

mouse_last_pos = (0.0, 0.0)
mouse_down = False

Erase_size = (124,52)
Erase_hitbox = (108, 160)

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
        result = testing.get_answer(grayscale_array)
        for r in result:
            breakdown_result = r.split(': ')
            arranged_result.append([-1*float(breakdown_result[1][:-1:]),breakdown_result[0]])
        arranged_result = sorted(arranged_result)
        
        # Clear the text area
        pygame.draw.rect(screen, bg_color, (50, 300, 290, 50 * len(result)))

        # Draw new results
        for i, r in enumerate(arranged_result):
            rendering_text(r[1]+': '+str(-1*r[0])+'%',36 - round(0.5 * len(r[1])),50,300 + 50 * i)
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
    if button(Erase_size, Erase_hitbox, './assets/erase.png', './assets/erase2.png'):
        pygame.draw.rect(screen, bg_color, (50, 300, 290, 50 * len(result)))
        pygame.draw.rect(screen.subsurface(canva), "white", (0, 0, screenx, screeny))
    
    #update screen
    pygame.display.flip() 
    mouse_last_pos = pygame.mouse.get_pos()
    
pygame.quit()
sys.exit()