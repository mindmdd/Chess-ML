import numpy as np
import math
import pyrr
import colorsys

def rgb2hsv(rgb):
    #rgb normal: range (0-255, 0-255, 0.255)
    red=rgb[0]
    green=rgb[1]
    blue=rgb[2]

    #get rgb percentage: range (0-1, 0-1, 0-1 )
    red_percentage= red / float(255)
    green_percentage= green/ float(255)
    blue_percentage=blue / float(255)

    
    #get hsv percentage: range (0-1, 0-1, 0-1)
    color_hsv_percentage=colorsys.rgb_to_hsv(red_percentage, green_percentage, blue_percentage) 

    #get normal hsv: range (0-360, 0-255, 0-255)
    color_h=round(360*color_hsv_percentage[0])
    color_s=round(255*color_hsv_percentage[1])
    color_v=round(255*color_hsv_percentage[2])
    color_hsv=(color_h, color_s, color_v)

    return color_hsv

def vector(start,end):
    x1,y1,z1 = start
    x2,y2,z2 = end
    return np.array([x2-x1, y2-y1, z2-z1])

def distance(start,end):
    x1,y1,z1 = start
    x2,y2,z2 = end
    return math.sqrt((x2-x1)**2 + (y2-y1)**2  + (z2-z1)**2 )

def add_vector(v, w):
    vx, vy,vz = v
    wx, wy, wz = w
    return np.array([vx+wx, vy+wy, vz+wz])

def point_on_line_from_distance(start, end, dist):
    u = pyrr.vector.normalise(vector(start, end))
    point = add_vector(start, dist*u)
    return point

def get_x_y_co(circles, radius, deg):
    xc = circles[0] #x-co of circle (center)
    yc = circles[1] #y-co of circle (center)

    y = yc + radius*math.sin(math.radians(deg))
    x = xc+ radius*math.cos(math.radians(deg))
    x=int(x)
    y=int(y)

    return x, y

def centroid(vertexes):
     _x_list = [vertex [0] for vertex in vertexes]
     _y_list = [vertex [1] for vertex in vertexes]
     _len = len(vertexes)
     _x = sum(_x_list) / _len
     _y = sum(_y_list) / _len
     return(_x, _y)

def best_fit_slope_and_intercept(xs,ys):
    m,b = np.polyfit(xs, ys, 1)
   
    return m, b

def rotate_square(edge, deg):
    # (0, 0), (1, 0), (1, 1), (0, 1)
    cx, cy = centroid(edge)
    radius = distance([cx, cy, 0], [edge[0][0], edge[0][1], 0])

    new_edge = []
    for a in edge:
        theta = math.degrees(math.atan2(a[1] - cy, a[0] - cx))
        if theta < 0:
            theta = 360+theta
        new_x, new_y = get_x_y_co([cx, cy], radius, 20+theta)
        new_edge.append([new_x, new_y])

    return new_edge
