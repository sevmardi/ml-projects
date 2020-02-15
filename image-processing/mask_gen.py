import numpy as np
from PIL import Image, ImageDraw

# IMAGE = 'hacks/t5XM4.png'
# IMAGE_MAP = 'hacks/Gz8b8.png'

IMAGE = 'hacks/color.png'
IMAGE_MAP = 'hacks/green.png'

IMAGE_OUTPUT = 'result.png'

GREEN = (0, 255, 0)
# RED = (229, 0, 0)
# LIGHT_RED = (220, 20, 60)

OFFSET = 12

image_map = Image.open(IMAGE_MAP)
image = Image.open(IMAGE)

pixels = image_map.load()
size_sm = image_map.size
# print(size_sm)
size = image.size
ratio = (size_sm[0] / size[0], size_sm[1] / size[1])

# print(ratio)
x_list = []
y_list = []

for x in range(size_sm[0]):
    for y in range(size_sm[1]):
        if pixels[x, y] == GREEN:
            x_list.append(x)
            y_list.append(y)
            print(x_list)
            print(y_list)

draw = ImageDraw.Draw(image)
draw.rectangle(((min(x_list)/ratio[0]-OFFSET, min(y_list)/ratio[1]-OFFSET),
                (max(x_list)/ratio[0]+OFFSET,max(y_list)/ratio[1]+OFFSET)),
               width=3, outline=LIGHT_RED)

# x1 = min(x_list) / ratio[0] - OFFSET
# y1 = min(y_list) / ratio[1] - OFFSET
# x2 = max(x_list) / ratio[0] + OFFSET
# y2 = max(y_list) / ratio[1] + OFFSET

# draw = ImageDraw.Draw(image)
# draw.rectangle(((x1, y1), (x2, y2)),
#                width=3, outline=GREEN)

# sub_img = img[y1:y2, x1:x2]
# image.save(IMAGE_OUTPUT, 'PNG')




# paper and noise on bounding box
sub_img = img[y1:y2,x1:x2]
sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
shape = sub_img.shape
for row in range(shape[0]):
    for col in range(shape[1]):
        rnd = random.randint(0,1)
        sub_img[row,col] = rnd*255

# img[y1:y2,x1:x2,0]=sub_img
# img[y1:y2,x1:x2,1]=sub_img
# img[y1:y2,x1:x2,2]=sub_img
