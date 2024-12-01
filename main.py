
import numpy as np
import cv2
import time
import pyautogui
from Xlib import X, display
import Xlib
from Xlib.ext.xtest import fake_input
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from proc import proc_image, detect_fingers
from classify import load_model

exit_flag = False
series_id = 7
N_FINGERS = 3
CLAMP_VAL = 150
MAX_MOVE_DIST= 100
# CLAMP_VAL = 100

DIST_READY = 70
DIST_CLICK = 40

last_top = (0, 0, 0, 0, 0, 0.0)
last_pos = {"top": 6*[0], "ready":True}

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def clamp_discard(val, min_val, max_val):
    if min_val <= val <= max_val:
        return val

    return 0
    # return max(min_val, min(val, max_val))

def on_key(event):
    global exit_flag
    if event.key == 'q':
        exit_flag = True

def open_camera():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open camera.")
        exit()

    return camera

def close_camera(camera):
    camera.release()

def click_button(root, disp, x, y, button=1):
    # Button 1 is left-click, 2 is middle-click, 3 is right-click
    
    # Simulate a mouse press
    root.warp_pointer(10, 10)  # Move pointer to a location (optional)
    root.change_attributes(event_mask=X.ButtonPressMask)
    
    # ButtonPress event (mouse down)
    event = Xlib.protocol.event.ButtonPress(
        time=int(time.time()),
        root=root,
        window=root,
        same_screen=1,
        child=X.NONE,
        root_x=0,
        root_y=0,
        event_x=5,
        event_y=5,
        state=0,
        detail=button,
    )
    root.send_event(event, propagate=True)
    
    # ButtonRelease event (mouse up)
    event = Xlib.protocol.event.ButtonRelease(
        time=int(time.time()),
        root=root,
        window=root,
        same_screen=1,
        child=X.NONE,
        root_x=0,
        root_y=0,
        event_x=5,
        event_y=5,
        state=0,
        detail=button,
    )
    root.send_event(event, propagate=True)
    
    disp.flush()  # Send the events to the X server


def control_cursor(image, rects, clss, vals):
    arr = [(*r, c, v) for r, c, v in zip(rects, clss, vals)]
    # arr = [i for i, c in enumerate(clss) if c == 1]
    gr_rects = [r for r in arr if r[4] == 1]
    if len(gr_rects) == 0:
        return 

    # print(gr_rects)

    gr_rects.sort(key=lambda t: t[5], reverse=True)
    gr_capped_best = gr_rects[:N_FINGERS]
    gr_capped_best.sort(key=lambda t: t[1])

    r = gr_capped_best[0]


    plt.text(r[0]+5, r[1]-15, "top", fontsize=9, color="black")

    disp = display.Display()
    screen = disp.screen()
    root = screen.root

    scr_w = screen.width_in_pixels
    scr_h = screen.height_in_pixels
    

    pointer = root.query_pointer()
    ptr = (pointer.root_x, pointer.root_y)
    
    last_top = last_pos["top"] 
    # last_pointer = last_pos["pointer"] 
    

    scl_x = 3.0 * scr_w / image.shape[1]
    scl_y = -3.0 * scr_h / image.shape[0]


    move_dist = np.linalg.norm(np.array(r[:2]) - np.array(last_top[:2]))
    # if move_dist > MAX_MOVE_DIST:
        # print(f"move dist {move_dist} > {MAX_MOVE_DIST}")
        # return
    
    # p_x = r[0] + r[2] // 2
    # p_y = r[1] + r[3] // 2
    
    # d_x = last_top[0] - p_x
    # d_y = last_top[1] - p_y

    d_x = last_top[0] - r[0]
    d_y = last_top[1] - r[1]

    # acc_d_x = np.sign(d_x) * 0.2 * d_x**2
    # acc_d_y = np.sign(d_y) * 0.2 * d_y**2


    acc_d_x = d_x - (d_x / (1 + 0.03*d_x**2))
    acc_d_y = d_y - (d_y / (1 + 0.03*d_y**2))
    
    acc_d_x = clamp_discard(acc_d_x, -CLAMP_VAL, CLAMP_VAL)
    acc_d_y = clamp_discard(acc_d_y, -CLAMP_VAL, CLAMP_VAL)
    

    # acc_d_x = clamp(acc_d_x, -CLAMP_VAL, CLAMP_VAL)
    # acc_d_y = clamp(acc_d_y, -CLAMP_VAL, CLAMP_VAL)
    
    final_x = int(ptr[0] + scl_x * acc_d_x)
    final_y = int(ptr[1] + scl_y * acc_d_y)

    

    # print(f"scl: {(scl_x, scl_y)}")
    # print(f"last: {last_top}")
    # print(f"top: {r}")
    # print(f"ptr: {ptr} => {(final_x, final_y)} diff:{(int(acc_d_x), int(acc_d_y))}")
    

    # rel_x = 1 - (p_x / image.shape[1])
    # rel_y = p_y / image.shape[0]

    # print(f"rel: {rel_x}:{rel_x}")
    # print(f"screen: {height}:{width}")

    # final_x = int(rel_x * scr_w)
    # final_y = int(rel_y * scr_h)

    # root.warp_pointer(p_y, p_x)
    root.warp_pointer(final_x, final_y)
    disp.sync()

    if len(gr_capped_best) >= N_FINGERS:
        r1 = gr_capped_best[1]
        r2 = gr_capped_best[2]
        click_dist = np.linalg.norm(np.array(r1[:2]) - np.array(r2[:2]))
        
        # print(f"{click_dist}")
        
        ready = last_pos["ready"] 
        if ready and click_dist <= DIST_CLICK:
            last_pos["ready"] = False
            
            fake_input(disp, X.ButtonPress, 1)
            disp.sync()
            fake_input(disp, X.ButtonRelease, 1)
            disp.sync()
            # click_button(root, disp, final_x, final_y, button=1)
            print("CLICKED")
            return 
        elif not ready and click_dist >= DIST_READY:
            last_pos["ready"] = True
            print("READY")

    # last_pos["pointer"] = ptr

    last_pos["top"] = r
    


def main_loop(camera):

    num = 0
    downscale_factor = 2

    # m = load_model("models/model_1")
    m = load_model("models/model_2")
    # Set up the plot
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', on_key)
    try: 
        while not exit_flag:
            ret, frame = camera.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # original_height, original_width = frame_rgb.shape[:2]
            # frame_rgb = cv2.resize(frame_rgb, (original_width // downscale_factor, original_height // downscale_factor), interpolation=cv2.INTER_AREA)
            # frame = cv2.resize(frame, (original_width // downscale_factor, original_height // downscale_factor), interpolation=cv2.INTER_AREA)
            ax.clear()
            # b_bs, b_cts = proc_image(frame, False)
            p_image, rects, clss, vals = detect_fingers(frame_rgb, model=m, is_file=False, show=False)

            # cv2.drawContours(frame_rgb, b_cts, -1, (128), thickness=-1)
            ax.imshow(p_image, cmap='gray' )

            for (x, y, w, h), c, v in zip(rects, clss, vals):
                # Create a red rectangle (edgecolor='r' for red, linewidth=2 for thickness)
                col = "g" if c == 1 else "r"
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=col, facecolor='none')
                ax.add_patch(rect)
                plt.text(x+5, y-5, f"{round(v, 2)}", fontsize=9, color="black")

            control_cursor(p_image, rects, clss, vals)

                # cv2.rectangle(p_image, (x, y), (x + w, y + h), (255, 0, 0), 1)
            # ax.axis('off')
            # plt.draw()

            plt.pause(0.01)

            # cv2.imwrite(f'img/test/image_{series_id}_{num}.png', frame)
            # num += 1


    except KeyboardInterrupt:
        print("Stopped by user")

if __name__ == '__main__':
    c = open_camera()
    main_loop(c)
    close_camera(c)

