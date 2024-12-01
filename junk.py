


    # diff_images = []
    # diff_b_images = []

    # for pair in palettes:
        # for i, palette in enumerate(pair):
            # for p in palette:
                # # print(f"p min = {min(p)}")
                # # print(f"p max = {max(p)}")
                # # diff_r = cv2.absdiff(r, np.full(r.shape, p[0], dtype=np.uint8))
                # # diff_g = cv2.absdiff(g, np.full(g.shape, p[1], dtype=np.uint8))
                # # diff_b = cv2.absdiff(b, np.full(b.shape, p[2], dtype=np.uint8))

                

                # difference_image = cv2.merge([diff_r, diff_g, diff_b])
                # grey_diff_image = np.linalg.norm(difference_image, axis=2)

                # grey_diff_image = diff_r

                # # grey_diff_image = diff_r // 3 + diff_g // 3 + diff_b // 3

                
                # if i == 0:
                    # diff_images.append(grey_diff_image)
                # else:
                    # diff_b_images.append(grey_diff_image)

                # # equalized_image = cv2.equalizeHist(grey_diff_image)

                # break





    # aggr_diff_image = np.minimum.reduce(diff_images, dtype=np.uint8)
    aggr_diff_image = np.minimum.reduce(diff_images)
    aggr_diff_image = np.clip(aggr_diff_image, 0, 255).astype(np.uint8)
    if show:
        cax = plt.imshow(aggr_diff_image)
        plt.colorbar(cax)
        plt.title(f"Aggr min diff clipped: {file}")
        plt.show()
    
    aggr_min_b_image = np.minimum.reduce(diff_b_images)
    aggr_min_b_image = np.clip(aggr_min_b_image, 0, 255).astype(np.uint8)
    # if show:
        # cax = plt.imshow(aggr_min_b_image)
        # plt.colorbar(cax)
        # plt.title(f"Aggr min diff: {file}")
        # plt.show()

    aggr_min_image = cv2.GaussianBlur(aggr_diff_image,(3,3),cv2.BORDER_DEFAULT)
    if show:
        cax = plt.imshow(aggr_min_image)
        plt.colorbar(cax)
        plt.title(f"Aggr min diff gauss 3: {file}")
        plt.show()


    # _, thr2_image = cv2.threshold(aggr_min_image, 25, 255, cv2.THRESH_BINARY_INV)
    # cax = plt.imshow(thr2_image)
    # plt.colorbar(cax)
    # plt.title(f"Aggr thr 25 min diff: {file}")
    # plt.show()

    _, thr_b_image = cv2.threshold(aggr_min_b_image, 20, 255, cv2.THRESH_BINARY)
    # cax = plt.imshow(thr_b_image)
    # plt.colorbar(cax)
    # plt.title(f"Aggr thr 20 b min diff: {file}")
    # plt.show()

    # kernel = np.ones((5, 5), np.uint8)  # A 5x5 square kernel
    kernel = np.ones((3, 3), np.uint8)  # A 5x5 square kernel
    opened_image = cv2.morphologyEx(thr_b_image, cv2.MORPH_OPEN, kernel)
    # opened_image = cv2.erode(thr_b_image, kernel, iterations=1)
    if show:
        cax = plt.imshow(opened_image)
        plt.colorbar(cax)
        plt.title(f"opened b min: {file}")
        plt.show()
    
    
    b_contours, _ = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(f"counters;{b_contours}")
    # out_image = cv2.cvtColor(aggr_min_image, cv2.COLOR_GRAY2BGR)
    out_image = opened_image
    
    b_bbs = []
    for contour in b_contours:
        x, y, w, h = cv2.boundingRect(contour)
        b_bbs.append((x, y, w, h))
        # cv2.rectangle(out_image, (x, y), (x + w, y + h), (128), 1)

    



    _, thr1_image = cv2.threshold(aggr_min_image, 10, 255, cv2.THRESH_BINARY_INV)
    if show:
        cax = plt.imshow(thr1_image)
        plt.title(f"Aggr thr 10 min diff: {file}")
        plt.show()
    
    kernel = np.ones((7, 7), np.uint8)  # A 5x5 square kernel
    dilated_image = cv2.dilate(thr1_image, kernel, iterations=1)
    # cax = plt.imshow(dilated_image)
    # plt.colorbar(cax)
    # plt.title(f"dilated thr1 min: {file}")
    # plt.show()

    # and_image = cv2.bitwise_and(dilated_image, opened_image)
    # cax = plt.imshow(and_image)
    # plt.title(f"and dilated diff: {file}")
    # plt.show()

    # kernel = np.ones((9, 9), np.uint8)  # A 5x5 square kernel
    # closed_image = cv2.morphologyEx(thr1_image, cv2.MORPH_CLOSE, kernel)
    # closeed_image = cv2.closee(thr_b_image, kernel, iterations=1)
    # cax = plt.imshow(closed_image)
    # plt.colorbar(cax)
    # plt.title(f"closed b min: {file}")
    # plt.show()


    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(f"counters;{contours}")
    out_image = cv2.cvtColor(aggr_min_image, cv2.COLOR_GRAY2BGR)
    out_image = aggr_min_image.copy()

    f_bbs = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        f_bbs.append((x, y, w, h))
        # cv2.rectangle(out_image, (x, y), (x + w, y + h), (0), 1)

    # plt.imshow(out_image)
    # plt.title(f"f_bbs: {file}")
    # plt.show()


    out_image = aggr_min_image.copy()
    cntr=[]
    i_bbs = get_itc_bbs(b_bbs, f_bbs, MIN_AREA, MAX_AREA)
    for (x, y, w, h) in f_bbs:
        if show:
            cv2.rectangle(out_image, (x, y), (x + w, y + h), (60), 1)
    for (x, y, w, h) in b_bbs:
        if show:
            cv2.rectangle(out_image, (x, y), (x + w, y + h), (45), 1)
    for (x, y, w, h) in i_bbs:
        cntr.append((x + w/2, y + h/2))
        if show:
            cv2.rectangle(out_image, (x, y), (x + w, y + h), (0), 1)

    if show:
        plt.imshow(out_image)
        plt.title(f"result: {file}")
        plt.show()


    return i_bbs

    # plt.title(f"detect: {file}")
    # plt.show()

    # and1_image = cv2.bitwise_and(thr1_image, and_image)
    # cax = plt.imshow(and1_image)
    # plt.colorbar(cax)
    # plt.title(f"and trr1 trh2: {file}")
    # plt.show()
    # aggr_diff_image = np.minimum.reduce(diff_images, dtype=np.uint8)
    aggr_diff_image = np.minimum.reduce(diff_images)
    aggr_diff_image = np.clip(aggr_diff_image, 0, 255).astype(np.uint8)
    if show:
        cax = plt.imshow(aggr_diff_image)
        plt.colorbar(cax)
        plt.title(f"Aggr min diff clipped: {file}")
        plt.show()
    
    aggr_min_b_image = np.minimum.reduce(diff_b_images)
    aggr_min_b_image = np.clip(aggr_min_b_image, 0, 255).astype(np.uint8)
    # if show:
        # cax = plt.imshow(aggr_min_b_image)
        # plt.colorbar(cax)
        # plt.title(f"Aggr min diff: {file}")
        # plt.show()

    aggr_min_image = cv2.GaussianBlur(aggr_diff_image,(3,3),cv2.BORDER_DEFAULT)
    if show:
        cax = plt.imshow(aggr_min_image)
        plt.colorbar(cax)
        plt.title(f"Aggr min diff gauss 3: {file}")
        plt.show()


    # _, thr2_image = cv2.threshold(aggr_min_image, 25, 255, cv2.THRESH_BINARY_INV)
    # cax = plt.imshow(thr2_image)
    # plt.colorbar(cax)
    # plt.title(f"Aggr thr 25 min diff: {file}")
    # plt.show()

    _, thr_b_image = cv2.threshold(aggr_min_b_image, 20, 255, cv2.THRESH_BINARY)
    # cax = plt.imshow(thr_b_image)
    # plt.colorbar(cax)
    # plt.title(f"Aggr thr 20 b min diff: {file}")
    # plt.show()

    # kernel = np.ones((5, 5), np.uint8)  # A 5x5 square kernel
    kernel = np.ones((3, 3), np.uint8)  # A 5x5 square kernel
    opened_image = cv2.morphologyEx(thr_b_image, cv2.MORPH_OPEN, kernel)
    # opened_image = cv2.erode(thr_b_image, kernel, iterations=1)
    if show:
        cax = plt.imshow(opened_image)
        plt.colorbar(cax)
        plt.title(f"opened b min: {file}")
        plt.show()
    
    
    b_contours, _ = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(f"counters;{b_contours}")
    # out_image = cv2.cvtColor(aggr_min_image, cv2.COLOR_GRAY2BGR)
    out_image = opened_image
    
    b_bbs = []
    for contour in b_contours:
        x, y, w, h = cv2.boundingRect(contour)
        b_bbs.append((x, y, w, h))
        # cv2.rectangle(out_image, (x, y), (x + w, y + h), (128), 1)

    



    _, thr1_image = cv2.threshold(aggr_min_image, 10, 255, cv2.THRESH_BINARY_INV)
    if show:
        cax = plt.imshow(thr1_image)
        plt.title(f"Aggr thr 10 min diff: {file}")
        plt.show()
    
    kernel = np.ones((7, 7), np.uint8)  # A 5x5 square kernel
    dilated_image = cv2.dilate(thr1_image, kernel, iterations=1)
    # cax = plt.imshow(dilated_image)
    # plt.colorbar(cax)
    # plt.title(f"dilated thr1 min: {file}")
    # plt.show()

    # and_image = cv2.bitwise_and(dilated_image, opened_image)
    # cax = plt.imshow(and_image)
    # plt.title(f"and dilated diff: {file}")
    # plt.show()

    # kernel = np.ones((9, 9), np.uint8)  # A 5x5 square kernel
    # closed_image = cv2.morphologyEx(thr1_image, cv2.MORPH_CLOSE, kernel)
    # closeed_image = cv2.closee(thr_b_image, kernel, iterations=1)
    # cax = plt.imshow(closed_image)
    # plt.colorbar(cax)
    # plt.title(f"closed b min: {file}")
    # plt.show()


    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(f"counters;{contours}")
    out_image = cv2.cvtColor(aggr_min_image, cv2.COLOR_GRAY2BGR)
    out_image = aggr_min_image.copy()

    f_bbs = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        f_bbs.append((x, y, w, h))
        # cv2.rectangle(out_image, (x, y), (x + w, y + h), (0), 1)

    # plt.imshow(out_image)
    # plt.title(f"f_bbs: {file}")
    # plt.show()


    out_image = aggr_min_image.copy()
    cntr=[]
    i_bbs = get_itc_bbs(b_bbs, f_bbs, MIN_AREA, MAX_AREA)
    for (x, y, w, h) in f_bbs:
        if show:
            cv2.rectangle(out_image, (x, y), (x + w, y + h), (60), 1)
    for (x, y, w, h) in b_bbs:
        if show:
            cv2.rectangle(out_image, (x, y), (x + w, y + h), (45), 1)
    for (x, y, w, h) in i_bbs:
        cntr.append((x + w/2, y + h/2))
        if show:
            cv2.rectangle(out_image, (x, y), (x + w, y + h), (0), 1)

    if show:
        plt.imshow(out_image)
        plt.title(f"result: {file}")
        plt.show()


    return i_bbs

    # plt.title(f"detect: {file}")
    # plt.show()

    # and1_image = cv2.bitwise_and(thr1_image, and_image)
    # cax = plt.imshow(and1_image)
    # plt.colorbar(cax)
    # plt.title(f"and trr1 trh2: {file}")
    # plt.show()
    
    # sharpening_kernel = np.array([[ 0, -1,  0],
                                  # [-1,  5, -1],
                                  # [ 0, -1,  0]])

    # sharpening_kernel = np.array([
                    # [ 0, 0, -1,  0, 0],
                    # [0, -1,  -2, -1, 0],
                    # [ -1, -2,  17, -2, -1],
                    # [0, -1,  -2, -1, 0],
                    # [ 0, 0, -1,  0, 0]
                    # ])

    # Apply the sharpening kernel using cv2.filter2D
    # image = cv2.filter2D(image, -1, sharpening_kernel)

    # cax = plt.imshow(image)
    # if show:
        # cax = plt.imshow(image)
        # plt.title(f"Sharpened")
        # plt.colorbar(cax)
        # plt.show()


    # image = cv2.GaussianBlur(image,(3,3),cv2.BORDER_DEFAULT)
    # cax = plt.imshow(image)
    # if show:
        # cax = plt.imshow(gauss_image)
        # plt.title(f"Gauss")
        # plt.colorbar(cax)
        # plt.show()
    


def get_itc_bbs(dt, flt, min_area, max_area):
    arr = []
    for d in dt:
        S = d[2]*d[3]
        if S > max_area or S < min_area:
            continue

        found = False
        for f in flt:
            # Check if one rectangle is to the left of the other
            if (d[0]+d[2]) < f[0] or (f[0]+f[2]) < d[0]:
                continue
            # Check if one rectangle is above the other
            if (d[1]+d[3]) < f[1] or (f[1]+f[3]) < d[1]:
                continue

            # If neither condition is met, the rectangles intersect
            found = True
            break

        if found:
            arr.append(d)

    return arr



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backend_bases import MouseButton

# Sample image: we'll use a random image for this example
image = np.random.rand(300, 400, 3)

# List of rectangles as (x, y, width, height)
rectangles = [(50, 50, 100, 50), (200, 100, 120, 60), (100, 200, 80, 40)]

# Create a figure and axis
fig, ax = plt.subplots()
ax.imshow(image)

# Store the patches (rectangles)
patches = []
colors = ['red'] * len(rectangles)  # Initial colors for rectangles (red)
for i, (x, y, w, h) in enumerate(rectangles):
    rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=colors[i], facecolor='none')
    ax.add_patch(rect)
    patches.append(rect)

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
            patches[closest_idx].set_edgecolor('green')
            fig.canvas.draw()  # Redraw the figure to update the colors

# Connect the click event to the handler
fig.canvas.mpl_connect('button_press_event', on_click)

# Display the plot
plt.show()






def filter_bbs_by_color(img, bbs:list, dif:int, ok:float):
    lst = []
    min_edge = 30
    for i, bb in enumerate(bbs):
        # print(f"bb[{i}]:{bb}")
        x, y, w, h = bb
        # print(f"{i} x:{x} y:{y} w:{w} h:{h}")
        shp = img.shape

        if w < min_edge:
            x = x + w // 2 - min_edge // 2
            x = min(max(x, 0), shp[1]-min_edge-1)
            w = min_edge

        if h < min_edge:
            y = y + h // 2 - min_edge // 2
            y = min(max(y, 0), shp[0]-min_edge-1)
            h = min_edge

        # print(f"{i} x:{x} y:{y} w:{w} h:{h}")

        bb_img = img[y:y+h, x:x+w]
        bb_img_scl = cv2.resize(bb_img, (14, 14), interpolation=cv2.INTER_AREA)
        bb_img_scl = cv2.normalize(bb_img_scl, None, 0, 255, cv2.NORM_MINMAX)
        
        # plt.imshow(bb_img)
        # plt.title(f"{i}")
        # plt.show()
        # plt.imshow(bb_img_scl)
        # plt.title(f"{i}")
        # plt.show()
        # mask = np.zeros(bb_img.shape, dtype=bool)
        dr_lst = []
        for r in mix_r_palette:
            # lo = max(r-dif, 0)
            # hi = min(r+dif, 255)
            # m = ((bb_img > lo) & (bb_img < hi))
            dr = cv2.absdiff(bb_img, np.full(bb_img.shape, r, dtype=np.uint8))
            dr_lst.append(dr)
            
            # plt.imshow(dr)
            # plt.title(f"{i}:{r}")
            # plt.show()
            score = np.mean(dr)
            # print(f"r:{r} s:{score}")

            # print(m)
            # mask = mask | m
            # print()

        dr_aggr = np.minimum.reduce(dr_lst)
        # plt.imshow(dr_aggr)
        # plt.title(f"{i}:dr_aggr")
        # plt.show()

        
        score = np.mean(dr_aggr)
        # score = np.mean(mask)
        # print(score)
        if score <= ok:
            lst.append(bb)
    return lst

