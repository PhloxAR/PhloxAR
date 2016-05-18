from PhloxAR import Camera, Image, Display

cam = Camera()

display = Display()

img = cam.get_image()

img.save(display)
