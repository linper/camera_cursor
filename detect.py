
import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backend_bases import MouseButton

from proc import proc_image, get_cut

series=7


# Sample image: we'll use a random image for this example
image = np.random.rand(300, 400, 3)
exit_flag = False
g_image = None
g_cut_idx=0

GREEN = (0, 1, 0, 1)
RED = (1, 0, 0, 1)

# List of rectangles as (x, y, width, height)
# rectangles = [(50, 50, 100, 50), (200, 100, 120, 60), (100, 200, 80, 40)]
patches = []
rectangles = []


# Function to find the closest rectangle to the mouse click
def closest_rectangle(event):
    if event.inaxes is None:  # Click is outside the image
        return None
    mouse_x, mouse_y = event.xdata, event.ydata
    min_distance = float('inf')
    closest_idx = None
    
    # Calculate the distance to the center of each rectangle
    for i, (x, y, w, h) in enumerate(rectangles):
        rect_center_x = x + w / 2
        rect_center_y = y + h / 2
        distance = np.sqrt((mouse_x - rect_center_x) ** 2 + (mouse_y - rect_center_y) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_idx = i
    
    return closest_idx

# Event handler for mouse clicks
def on_click(event):
    if event.button is MouseButton.LEFT:  # Left mouse button
        closest_idx = closest_rectangle(event)
        if closest_idx is not None:
            # Change the color of the closest rectangle to green
            col = patches[closest_idx].get_edgecolor()
            if col == (1, 0, 0, 1):
                patches[closest_idx].set_edgecolor(GREEN)
            elif col == (0, 1, 0, 1):
                patches[closest_idx].set_edgecolor(RED)
            else:
                print(f"Unrecognized edge color: {col}")

            # print(f"edgecolor of {i}={col}")
            fig.canvas.draw()  # Redraw the figure to update the colors


def save_images():
    global g_cut_idx
    
    # r_pat = [p for p in patches if p.get_edgecolor() == RED]
    # g_pat = [p for p in patches if p.get_edgecolor() == GREEN]


    for r, p in zip(rectangles, patches):
        # min_edge = 30
        # x, y, w, h = r
        # # print(f"{i} x:{x} y:{y} w:{w} h:{h}")
        # shp = g_image.shape

        # if w < min_edge:
            # x = x + w // 2 - min_edge // 2
            # x = min(max(x, 0), shp[1]-min_edge-1)
            # w = min_edge

        # if h < min_edge:
            # y = y + h // 2 - min_edge // 2
            # y = min(max(y, 0), shp[0]-min_edge-1)
            # h = min_edge


        # bb_img = g_image[y:y+h, x:x+w]
        # bb_img_scl = cv2.resize(bb_img, (14, 14), interpolation=cv2.INTER_AREA)

        bb_img_scl = get_cut(g_image, r)


        cv2.imwrite(f"img/detect/norm/{'y' if p.get_edgecolor() == GREEN else 'n'}-{series}-{g_cut_idx}.png", bb_img_scl)
        g_cut_idx += 1
    

def on_key(event):
    # global exit_flag
    print(f"event.key={event.key}")
    if event.key == 'q':
        save_images()
        # ax.clear()
        # exit_flag = True
        # fig.canvas.draw()  # Redraw the figure to update the colors
    # elif event.key == 'c':
        r_pat = [p for p in patches if p.get_edgecolor() == RED]
        g_pat = [p for p in patches if p.get_edgecolor() == GREEN]
        print(f"r:{len(r_pat)} g:{len(g_pat)} total:{len(patches)}")
        print(f"r_pat:{r_pat}")
        print(f"g_pat:{g_pat}")
        # save_images()
        # exit_flag = True
        
        

# Create a figure and axis


def detect_boxes(file, progress):
    global patches, rectangles, g_image

    image, rects = proc_image(file, show=False)
    g_image = image
    
    rectangles = rects
    
    ax.clear()
    ax.imshow(image)

    lst = []
    colors = [RED] * len(rects)  # Initial colors for rectangles (red)
    # colors = [GREEN] * len(rects)  # Initial colors for rectangles (red)
    for i, (x, y, w, h) in enumerate(rects):
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=colors[i], facecolor='none')
        ax.add_patch(rect)
        lst.append(rect)

    patches = lst
    plt.title(f"{progress[0]}/{progress[1]}")

    

    plt.show()





if __name__ == '__main__':
    files = glob.glob(os.path.join(f"./img/test/image_{series}_*.png"))

    for i, file in enumerate(files):
        fig, ax = plt.subplots()

        fig.canvas.mpl_connect('key_press_event', on_key)
        fig.canvas.mpl_connect('button_press_event', on_click)
        ax.clear()
        detect_boxes(file, (i+1, len(files)))


