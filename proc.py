import os
import subprocess
import shutil
import glob
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from classify import load_model, image_predict, build_model

MIN_AREA = 36
MAX_AREA = 512

# rgb_palette = np.array([[73, 54, 64], [55, 37, 45], [90, 78, 93], [33, 19, 25]])
# rgb_palette = np.array([[255, 0, 0]])
# rgb_b_palette = np.array([[0, 5, 3]])

# mix_rgb_palette = np.array([[76, 57, 66], [ 11, 8, 8], [ 63, 17, 22], [167, 58, 48]])
mix_rgb_palette = np.array(
    [
        [84, 65, 76],
        [7, 7, 7],
        [96, 18, 18],
        [200, 100, 93],
        [58, 39, 48],
        [38, 11, 16],
        [163, 51, 41],
    ]
)
mix_r_palette = mix_rgb_palette[:, 0].flatten()

# rgb_palette = np.array([[127, 35, 28], [73, 14, 15], [167, 60, 50], [199, 103, 98]])
# rgb_palette = np.array([ [141, 62, 55], [ 66, 22, 17], [177, 100, 94], [103, 46, 38]])
# rgb_b_palette = np.array([[8, 9, 6], [18, 12, 14]])

# palettes = [(rgb_palette, rgb_b_palette)]


def get_cut(image, cut):
    min_edge = 30
    x, y, w, h = cut
    # print(f"{i} x:{x} y:{y} w:{w} h:{h}")
    shp = image.shape

    if w < min_edge:
        x = x + w // 2 - min_edge // 2
        x = min(max(x, 0), shp[1] - min_edge - 1)
        w = min_edge

    if h < min_edge:
        y = y + h // 2 - min_edge // 2
        y = min(max(y, 0), shp[0] - min_edge - 1)
        h = min_edge

    # print(f"{i} x:{x} y:{y} w:{w} h:{h}")

    bb_img = image[y : y + h, x : x + w]
    bb_img_scl = cv2.resize(bb_img, (14, 14), interpolation=cv2.INTER_AREA)
    bb_img_scl = cv2.normalize(bb_img_scl, None, 0, 255, cv2.NORM_MINMAX)

    return bb_img_scl


def save_cut_images(images):
    # subprocess.run(["rm", "-r", "/tmp/img/*"])
    directory = "/tmp/img/"
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove directory and its contents
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    for i, im in enumerate(images):
        cv2.imwrite(f"/tmp/img/n-0-{i}.png", im)


def detect_blobs(image):
    params = cv2.SimpleBlobDetector_Params()

    # Filter by Area (blobs must be within the specified area range)
    params.filterByArea = True
    params.minArea = 20  # Minimum blob area
    params.maxArea = 1000  # Maximum blob area

    # Filter by Circularity (blobs must be near-circular)
    params.filterByCircularity = True
    params.minCircularity = 0.65  # Minimum circularity (1.0 is a perfect circle)
    # params.minCircularity = 0.7  # Minimum circularity (1.0 is a perfect circle)
    # params.minCircularity = 0.85

    # Filter by Convexity (blobs must be mostly convex)
    params.filterByConvexity = True
    # params.minConvexity = 0.9
    # params.minConvexity = 0.8
    # params.minConvexity = 0.75
    params.minConvexity = 0.6

    # Filter by Inertia (to detect elongated blobs)
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.8

    # Create a blob detector with the defined parameters
    detector = cv2.SimpleBlobDetector_create(params)
    kpts = detector.detect(image)

    # orb = cv2.ORB_create()
    # kpts, descriptors = orb.detectAndCompute(image, None)

    # fast = cv2.FastFeatureDetector_create()
    # kpts = fast.detect(image, None)

    return kpts


def proc_image(file, model=None, is_file=True, show=False):
    # im = cv2.imread(file)

    if is_file:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = file

    r, g, b = cv2.split(img)

    image = r
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = 255 - image

    kpts = detect_blobs(image)

    kpt_bbs = []
    # print(kpts)
    for i, kpt in enumerate(kpts):
        # print(f"Keypoint {i+1}:")
        # print(f" - Location (x, y): ({kpt.pt[0]:.2f}, {kpt.pt[1]:.2f})")
        # print(f" - Size (diameter): {kpt.size:.2f}")
        # print(f" - Angle: {kpt.angle}")
        # print(f" - Response: {kpt.response}")
        # print(f" - Octave (pyramid layer where the kpt was found): {kpt.octave}")
        # print(f" - Class ID: {kpt.class_id}")

        kpt_bb = (
            int(kpt.pt[0] - kpt.size * 0.7),
            int(kpt.pt[1] - kpt.size * 0.7),
            int(1.4 * kpt.size),
            int(1.4 * kpt.size),
        )
        # print(f" - Rectangle: {kpt_bb}")
        kpt_bbs.append(kpt_bb)

    # col_bbs = filter_bbs_by_color(image, kpt_bbs, 20, 15.0)
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of the blob
    # draw_type = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    # draw_type = cv2.DRAW_MATCHES_FLAGS_DEFAULT
    # im_with_kpts = cv2.drawKeypoints(image, kpts, np.array([]), (0, 0, 255), draw_type)

    # for i, p in enumerate(kpt_bbs):
    # cv2.putText(im_with_kpts, f"{i}", (p[0]+5, p[1]-5), None , 0.4, (0,0,0), 2)
    # cv2.drawContours(image, kpt_bbs, -1, (0, 0, 255), thickness=-1)

    # for rect in kpt_bbs:
    # x, y, w, h = rect  # Unpack the rectangle data
    # r = cv2.rectangle(im_with_kpts, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # x, y, w, h = bb
    # for rect in col_bbs:
    # x, y, w, h = rect  # Unpack the rectangle data
    # cv2.rectangle(im_with_kpts, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # if show:
    # cax = plt.imshow(im_with_kpts)
    # plt.show()

    return image, kpt_bbs


def detect_fingers(file, model=None, is_file=True, show=False):
    image, rects = proc_image(file, model, is_file, show)

    cut_images = []

    for r in rects:
        cut = get_cut(image, r)
        cut_images.append(cut)

    # save_cut_images(cut_images)
    pred = []
    vals = []
    if len(cut_images) != 0 and model is not None:
        pred, vals = image_predict(model, cut_images)

    # for i, (p, t) in enumerate(zip(rects, pred)):
    # x, y, w, h = p # Unpack the rectangle data
    # cv2.putText(image, f"{i}", (x+5, y-5), None , 0.4, (255,0,0), 2)
    # col = (255, 0, 0) if t == 1 else (128, 0, 0)
    # r = cv2.rectangle(image, (x, y), (x + w, y + h), col, 2)

    # cax = plt.imshow(image)
    # plt.show()

    return image, rects, pred, vals


def img_hists(image):
    # Step 2: Split the image into Blue, Green, and Red channels
    blue_channel, green_channel, red_channel = cv2.split(image)
    b_range = [10, 128]
    g_range = [10, 128]
    r_range = [10, 128]

    max_val = 128

    # Step 3: Calculate the histogram for each channel using OpenCV's calcHist function
    # The arguments are: image, channels, mask, histSize, ranges
    # hist_blue = cv2.calcHist([blue_channel], [0], None, [b_range[1] - b_range[0]], b_range)
    # hist_green = cv2.calcHist([green_channel], [0], None, [g_range[1] - g_range[0]], g_range)
    # hist_red = cv2.calcHist([red_channel], [0], None, [r_range[1] - r_range[0]], r_range)
    hist_blue = cv2.calcHist([blue_channel], [0], None, [24], b_range)
    hist_green = cv2.calcHist([green_channel], [0], None, [24], g_range)
    hist_red = cv2.calcHist([red_channel], [0], None, [24], r_range)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.title("Pixel Intensity Histograms for Each Channel")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    plt.plot(hist_blue, color="blue", label="Blue Channel")
    plt.plot(hist_green, color="green", label="Green Channel")
    plt.plot(hist_red, color="red", label="Red Channel")
    plt.grid("both")

    plt.legend()
    plt.show()


def downscale_image(file):
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Step 2: Get the original dimensions of the image
    original_height, original_width = image.shape[:2]

    for i in [1, 2, 4, 8, 16]:
        # Step 3: Downscale the image by factors of 2, 4, and 8
        downscale_factor = cv2.resize(
            image,
            (original_width // i, original_height // i),
            interpolation=cv2.INTER_AREA,
        )

        # Step 4: Display the downscaled images
        plt.imshow(downscale_factor)
        raw_name = os.path.splitext(os.path.basename(file))[
            0
        ]  # Returns 'example_image'
        plt.tight_layout()
        cv2.imwrite(
            f"img/test_down/{raw_name}_{i}.png",
            cv2.cvtColor(downscale_factor, cv2.COLOR_RGB2BGR),
        )
        # plt.savefig(f"img/test_down/{raw_name}_{i}.png")
        plt.title(f"{raw_name}_{i}.png")
        # plt.show()


if __name__ == "__main__":
    # files = glob.glob(os.path.join("./img/test_down/image2.0_2.png"))
    # files = glob.glob(os.path.join("./img/test/", '*.png'))
    files = glob.glob(os.path.join("./img/series/7/", "*.png"))
    # files = glob.glob(os.path.join("./img/", '*.jpg'))
    # files = glob.glob(os.path.join("./img/test_down/", '*1.png'))
    # files = glob.glob(os.path.join("./img/test_down/", 'image_1_*2.png'))
    # files = glob.glob(os.path.join("./img/test_down/", 'image_0_*1.png'))
    # files = glob.glob(os.path.join("./img/test_down/", 'image_*1.png'))
    # files = glob.glob(os.path.join("./img/test_down/", 'image_*2.png'))
    # files = glob.glob(os.path.join("./img/test_down/", '*4.png'))
    # files = glob.glob(os.path.join("./img/test_down/", '*8.png'))
    # files = glob.glob(os.path.join("./img/test_down/", '*16.png'))

    m = load_model("models/model_1")
    # m = build_model()

    for file in files:
        # print(f"file={file}")
        # downscale_image(file)
        # image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        # image = cv2.imread(file)
        # proc_image(file, show=True)
        # proc_image(file, model=m, show=False)
        detect_fingers(file, model=m, show=False)
        # img_hists(image)
