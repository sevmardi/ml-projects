
def center_extent(image, size):
    (eW, eH) = size
   	# checks to see if the width is greater than the height of the image. If this is the case, the image is resized
  	# based on its width.
    if image.shape[1] > image.shape[0]:
        image = imutils.resize(image, width=eW)
    else:
        image = imutils.resize(image, height=eH)
    extent = np.zeros((eH, eW), dtype='unit8')
    # These offsets indicate the starting (x, y) coordinates (iny, x order) of
    # where the image will be placed in the extent.
    offsetX = (eW - image.shape[1]) // 2
    offsetY = (eH - image.shape[0]) // 2
    extent[offsetY:offsetY + image.shape[0],
           offsetX:offsetX + image.shape[1]] = image
    # computes the weighted mean of the white pixels in the image
    CM = mahotas.center_of_mass(extent)
    (cY, cX) = np.round(CM).astype("int32")
    (dX, dY) = ((size[0] // 2) - cX, (size[1] // 2) - cY)
    M = np.float32([[1, 0, dX], [0, 1, dY]])
    extent = cv2.warpAffine(extent, M, size)

    return extent
