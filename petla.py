import pygame
from pygame.locals import *
import keyboard
import numpy as np 
from math import *
import sys

width = 1000
height = 500
screen=pygame.display.set_mode((width,height))
screen.fill((0,0,0))
camera = 1000
zoom = 1
step = 20
angle = 0.02

#wczytanine pliku z danymi
f = open("bryly.txt", "r")  
tekst = f.read()

i = 0
tmp = None
V = []
E = []
e = False
while (i < len(tekst)):
    if(tekst[i] == '\n'):
        e = not(e)
    if(tekst[i].isdigit()):
        if(tmp == None):
            tmp = 0
        tmp = tmp *10 +  int(tekst[i])
    else:
        x = tmp
        tmp = None
        if(x != None):
            if(e):
                E.append(x)
            else:
                V.append(x)
    i+=1
verticies = []
point = []
edges = []

for i in range(len(V)):
    point.append(V[i])
    if(i%3 == 2):
        verticies.append(point)
        point = []

for i in range(len(E)):
    point.append(E[i])
    if(i%2 == 1):
        edges.append(point)
        point = []
# print(verticies)
# print(edges)
#tuuuu
number_of_blocks = int(len(edges)/12)

print(number_of_blocks)

def get_straight(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    A = y1-y2
    B = x2-x1
    C = (y2 - y1)*x1 - (x2 - x1)*y1

    return (A,B,C)

def is_in_straight(A,B,C,x,y):
    if(abs(A*x + B*y + C) < 1 ):  
        return True
    return False

def get_x(y, A, B, C):
    if(A == 0):
        return False
    else:
        x = - (B * y + C) /A
    return x

def get_y(x, A, B, C):
    if(B == 0):
        return False
    else:
        y = - (A * x + C) /B
    return y

def get_z (x1,y1,z1,x2,y2,z2,x,y):
    if (z1 == z2): 
        return z1
    if (x1!=x2):
        z = z1 + (x-x1)/(x2-x1)*(z2-z1)
    else:
        z = z1 + (y-y1)/(y2-y1)*(z2-z1)
    return z

def draw_linie(screen,p1,p2,index):
    white = (255, 255, 255)
    pygame.draw.line(screen, white, [p1[0],p1[1]], [p1[0],p1[1]])
    pygame.draw.line(screen, white, [p2[0],p2[1]], [p2[0],p2[1]])

    if (index == 0):
        color = (255,0,0)
    elif (index == 1):
        color = (50,255,50)
    elif (index == 2):
        color = (0,100,200)
    elif (index == 3):
        color = (255,0,127)
    if(p1[0] > p2[0]):
        pygame.draw.line(screen, color, [p1[0]-2,p1[1]], [p2[0]+2,p2[1]])
    else: 
        pygame.draw.line(screen, color, [p1[0]+2,p1[1]], [p2[0]-2,p2[1]])


#translacje
def translateUp(verticies):
    for vertex in verticies:
        vertex[1] += step
    return verticies

def translateDown(verticies):
    for vertex in verticies:
        vertex[1] -= step
    return verticies

def translateLeft(verticies):  
    for vertex in verticies:
        vertex[0] +=step
    return verticies

def translateRight(verticies): 
    for vertex in verticies:
        vertex[0] -= step
    return verticies

def translateForward(verticies):
    for vertex in verticies:
        vertex[2] -= step
    return verticies

def translateBack(verticies):
    for vertex in verticies:
        vertex[2] += step
    return verticies

#obroty
def rotateX(verticies):
    M =([1,0,0],
        [0,cos(angle),-sin(angle)],
        [0,sin(angle),cos(angle)])
    for i in range (len(verticies)):
        # przesuni??cie z powodu kamery w z = -1000
        # (x,y,z) jest w takim samym pomo??eniu wzgl??dem ??rodka rzutni jak (x,y,z+1000) wzgl??dem ??rodka uk??adu wsp????rz??dnych 
        verticies[i][2] += camera
        verticies[i] = np.dot(M,verticies[i])
        verticies[i][2] -= camera
    return verticies

def rotateBackX(verticies):
    M =([1,0,0],
        [0,cos(-angle),-sin(-angle)],
        [0,sin(-angle),cos(-angle)])
    for i in range (len(verticies)):
        verticies[i][2] += camera
        verticies[i] = np.dot(M,verticies[i])
        verticies[i][2] -= camera
    return verticies

def rotateY(verticies):
    M =([cos(angle),0,sin(angle)],
        [0,1,0],
        [-sin(angle),0,cos(angle)])
    for i in range (len(verticies)):
        verticies[i][2] += camera
        verticies[i] = np.dot(M,verticies[i])
        verticies[i][2] -= camera
    return verticies

def rotateYBack(verticies):
    M =([cos(-angle),0,sin(-angle)],
        [0,1,0],
        [-sin(-angle),0,cos(-angle)])
    for i in range (len(verticies)):
        verticies[i][2] += camera
        verticies[i] = np.dot(M,verticies[i])
        verticies[i][2] -= camera
    return verticies

def rotateZ(verticies):
    M =([cos(angle),-sin(angle),0],
        [sin(angle),cos(angle),0],
        [0,0,1])
    for i in range (len(verticies)):
        verticies[i][2] += camera
        verticies[i] = np.dot(M,verticies[i])
        verticies[i][2] -= camera
    return verticies

def rotateZBack(verticies):
    M =([cos(-angle),-sin(-angle),0],
        [sin(-angle),cos(-angle),0],
        [0,0,1])
    for i in range (len(verticies)):
        verticies[i][2] += camera
        verticies[i] = np.dot(M,verticies[i])
        verticies[i][2] -= camera
    return verticies

#zoom
def enlarging(zoom):
    if(zoom < 10):
        zoom +=0.1
    return zoom

def reducing(zoom ):
    if(zoom > 1):
        zoom -=0.1
    return zoom

#rzutowanie
def projection(verticies, zoom):
    two_dimensional = []
    for i in range (len(verticies)):
        
        scale = camera/(camera + verticies[i][2] )
        x = verticies[i][0] * scale * zoom
        y = verticies[i][1] * scale * zoom

        # w pygame y jest odwr??cony a (0,0) jest w lewym g??rnym rogu
        x = x + width/2
        y = y*(-1) + height/2
        two_dimensional.append((x,y))
    return two_dimensional

while True:
    #rysowanie
    screen.fill((0,0,0))

    two_dimensional = projection(verticies, zoom)

    #stare rysowanie
    # for i in range(len(edges)):
    #     vertex_1 = edges[i][0]
    #     vertex_2 = edges[i][1]
    #     if(verticies[vertex_1][2] > -camera and verticies[vertex_2][2] > -camera ):
    #         x1 = two_dimensional[vertex_1][0]
    #         y1 = two_dimensional[vertex_1][1]

    #         x2 = two_dimensional[vertex_2][0]
    #         y2 = two_dimensional[vertex_2][1]
        
    #         pygame.draw.line(screen,(255,255,255),(x1,y1),(x2,y2))
    #nowe rysowanie

    for y in range(0,height):
        points = []
        distance = []

        for i in range (number_of_blocks):
            points.append(None)
            distance.append(None)
            points[i] = []
            distance[i] = []

        min = []
        max = []
        min_distance = []
        max_distance = []

        for i in range(len(edges)):
            vertex_1 = edges[i][0]
            vertex_2 = edges[i][1]
            if(verticies[vertex_1][2] > -camera and verticies[vertex_2][2] > -camera ):
                x1 = two_dimensional[vertex_1][0]
                y1 = two_dimensional[vertex_1][1]

                x2 = two_dimensional[vertex_2][0]
                y2 = two_dimensional[vertex_2][1]
                if not((y > y1 and y > y2) or (y < y1 and y < y2)):
                    A,B,C = get_straight(two_dimensional[vertex_1], two_dimensional[vertex_2])
                    x =  get_x(y, A, B, C)
                    if(x != False):
                        plus = True
                        minus = True
                        x_tmp = x + 1
                        while(plus):
                            if (abs(get_y(float(x_tmp),A,B,C) - y) <0.5 ):
                                block_indeks = (int(i/12))
                                points[block_indeks].append([x_tmp,y])
                                distance[block_indeks].append((verticies[vertex_1][2]+verticies[vertex_2][2])/2)
                                x_tmp = x_tmp + 1
                                if(x_tmp > x1 and x_tmp > x2):
                                    plus = False
                            else:
                                plus = False
                        x_tmp = x - 1

                        while(minus):
                            if (abs(get_y(float(x_tmp),A,B,C) - y) <=0.5 ):
                                block_indeks = (int(i/12))
                                points[block_indeks].append([x_tmp,y])
                                distance[block_indeks].append((verticies[vertex_1][2]+verticies[vertex_2][2])/2)
                                x_tmp = x_tmp - 1
                                if(x_tmp < x1 and x_tmp < x2):
                                    minus = False
                            else:
                                minus = False
                           
                        block_indeks = (int(i/12))
                        points[block_indeks].append([x,y])
                        # z = get_z(x1,y1,verticies[vertex_1][2],x2,y2,verticies[vertex_2][2],x,y)
                        # distance[block_indeks].append(z)
                        distance[block_indeks].append((verticies[vertex_1][2]+verticies[vertex_2][2])/2)
                        
        for i in range (number_of_blocks):
            max.append(-1)
            min.append(1000000)
            min_distance.append(None)
            max_distance.append(None)
            for p in range(len(points[i])):
                if (points[i][p][0] > max[i]):
                    max[i] = points[i][p][0]
                    max_distance[i] = distance[i][p]
                if (points[i][p][0] < min[i]):
                    min[i] = points[i][p][0]
                    min_distance[i] = distance[i][p]

        
        for i in range (2):
            ok = True
            for j in range (2):
                if (i != j):
                    if(not(max[i] < min[j] or min[i] > max[j])):
                        ok = False
            #bry??a oddzielnie
            if(ok):
                if(max[i] != -1):
                    draw_linie(screen,[min[i],y],[max[i],y],i)

                    for p in range(len(points[i])):
                        if((max[i]-min[i]) != 0):
                            tmp = (points[i][p][0] - min[i]) / (max[i]-min[i])
                        tmp = 0
                        if(distance[i][p] <= tmp * max_distance[i] + (1-tmp) * min_distance[i]):
                            pygame.draw.line(screen,(255,255,255),points[i][p],points[i][p])

            #jedna wewn??trz drugiej
            else:
                for j in range (2):
                    begin = 100000000
                    end = -1000000000
                    #mog?? by?? wyci??te 2+ oddzielne kawa??ki
                    if((min_distance[i] + max_distance[i] < min_distance[j] + max_distance[j]) and not(max[i] < min[j] or min[i] > max[j]) ):
                        if(min[j] < begin):
                            begin = min[j]
                        if(max[j] > end):
                            end = max[j]
                if(begin != 100000000):
                    if (min[i] < begin):
                        draw_linie(screen,[min[i],y],[begin,y],i)
                    if (max[i] > end):
                        draw_linie(screen,[end,y],[max[i],y],i)

                        for p in range(len(points[i])):
                            if(points[i][p][0] < begin or points[i][p][0] > end):   
                                tmp = (points[i][p][0] - min[i]) / (max[i]-min[i])
                                if(distance[i][p] <= tmp * max_distance[i] + (1-tmp) * min_distance[i]):
                                    pygame.draw.line(screen,(255,255,255),points[i][p],points[i][p])
                else:
                    if(max[i] != -1):
                        draw_linie(screen,[min[i],y],[max[i],y],i)

                        for p in range(len(points[i])):
                            if((max[i]-min[i]) != 0):
                                tmp = (points[i][p][0] - min[i]) / (max[i]-min[i])
                            tmp = 0
                            if(distance[i][p] <= tmp * max_distance[i] + (1-tmp) * min_distance[i]):
                                pygame.draw.line(screen,(255,255,255),points[i][p],points[i][p])

    for events in pygame.event.get():
        if events.type == QUIT:
            sys.exit(0)

    #sprawdzanie u??ycia klawiatury 
    if keyboard.is_pressed("up arrow"):
        translateUp(verticies)  

    if keyboard.is_pressed("down arrow"):
        translateDown(verticies)

    if keyboard.is_pressed("left arrow"):
        translateLeft(verticies)

    if keyboard.is_pressed("right arrow"):
        translateRight(verticies)

    if keyboard.is_pressed('r'):
        translateForward(verticies) 

    if keyboard.is_pressed('f'):
        translateBack(verticies)

    if keyboard.is_pressed('w'):
        rotateX(verticies) 

    if keyboard.is_pressed('s'):
        rotateBackX(verticies)

    if keyboard.is_pressed('a'):
        rotateY(verticies) 

    if keyboard.is_pressed('d'):
        rotateYBack(verticies)

    if keyboard.is_pressed('q'):
        rotateZ(verticies) 

    if keyboard.is_pressed('e'):
        rotateZBack(verticies)

    if keyboard.is_pressed('p'):
        zoom = enlarging(zoom) 

    if keyboard.is_pressed('l'):
        zoom = reducing(zoom)

    pygame.display.flip()
    pygame.time.wait(1)
