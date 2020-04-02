# -*- coding: utf-8 -*-
import pygame


class showResult:
    def __init__(self, pygame, screen, 
                 type_forText = {   -1: "Nothing",
                                     0: "Left Flick ",
                                     1: "Left Push  ",
                                     2: "Right Flick",
                                     3: "Right Push ",
                                     4: "Rub        "   }):
        # First need to connect pygame and screen where you want to show in
        # Also need index and text of all gestures --> just use -1 for nothing
    
    
        self.pygame = pygame
        self.screen = screen
        
        self.pygame_w, self.pygame_h = self.pygame.display.get_surface().get_size()
        
        self.type_forText = type_forText
        self.num_type = len(self.type_forText.keys())
        self.string_result = "Nothing"
        
        self.result_txt_position = (self.pygame_w/6, self.pygame_h/2) #Position of BIG result 
        self.list_txt_position = [self.pygame_w/3, (self.pygame_h/(self.num_type+2))]
        self.bar_padding = self.pygame_w/3
        self.bar_size = [self.pygame_w/4, self.pygame_h/20]
        
        
        
        self.font_base = int(10*(self.pygame_h/1080))
        # initialize font; must be called after 'pygame.init()' to avoid 'Font not Initialized' error
        self.type_font = self.pygame.font.SysFont("arial", self.font_base*10, bold=True)
        self.lr_font = self.pygame.font.SysFont("monospace", self.font_base*8)
        
        
        self.resultBig_font = self.pygame.font.SysFont("arial", self.font_base*10, bold=True)
        self.list_font = self.pygame.font.SysFont("monospace", self.font_base*5)
        

    def showResultTextwProp(self, res, res_prop):
        """
            res: index of the selected gesture
            res_prop: scores of each gesture must be same size with 'type_forText'
        
        """
        
        # set the result highlighted as blue
        if (res in self.type_forText) and res != -1:
            self.string_result = self.type_forText[res]
            color_lr = (0,100,255)
        else:
            color_lr = (100,100,100)
    
    
        
        res_label = self.resultBig_font.render(self.string_result, 1, color_lr)
        res_pos = res_label.get_rect(center=self.result_txt_position)
        self.screen.blit(res_label, res_pos)
        
        
        for i,i_type in enumerate(list(self.type_forText.keys())):
            string_row = '{:15}: {:3.0f}%'.format(self.type_forText[i_type], (res_prop[i]*100))
            list_label = self.list_font.render(string_row, 1, (30,30,30))
            list_rect = (self.list_txt_position[0], self.list_txt_position[1]*(i+1) )
            self.screen.blit(list_label, list_rect)
            
            full_bar = list(list_rect)
            full_bar[0] += self.bar_padding
            full_bar += self.bar_size
            self.pygame.draw.rect(self.screen, (220,220,220), tuple(full_bar), 0)
            
            if res == i_type:
                bar_color = (0,100,255)
            else:
                bar_color = (100,100,100)
                
            bar_rect = list(list_rect)
            bar_rect[0] += self.bar_padding
            bar_rect += [int(res_prop[i]*self.bar_size[0]), self.bar_size[1]]
            self.pygame.draw.rect(self.screen, bar_color, tuple(bar_rect), 0)
        
        
    
    
def divideScreen(w,h,m,n):
    w_unit = (w/m)/2
    h_unit = (h/n)/2
    
    mid_points = []
    for j in range(n):
        for i in range(m):
            mid_points.append([int(w_unit*(2*i+1)),int(h_unit*(2*j+1))])
    
    max_radius = w_unit<h_unit and w_unit or h_unit
    
    return mid_points, int(max_radius)
