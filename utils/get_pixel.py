from PIL import Image

def getpixel_pil(image):
    arr = []
    for w in range(image.size[0]):
        for h in range(image.size[1]):
            p = image.getpixel((w, h))
            if p not in arr:
                arr.append(p)
    return arr