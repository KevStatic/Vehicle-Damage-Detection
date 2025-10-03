import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def visualize_prediction(img_path, boxes):
    img = Image.open(img_path)
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for box in boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()
