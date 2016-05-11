from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pygame, pygame.image
from pygame.locals import *
from scipy import linalg
import numpy as npy
import pickle
from PCV.geometry import camera, homography
from PCV.localdescriptors import sift

width,height = 1000,747


def my_calibration(sz):
    """
    Calibration function for the camera (iPhone4) used in this example.
    """
    row,col = sz
    fx = 2555*col/2592
    fy = 2586*row/1936
    K = npy.diag([fx,fy,1])
    K[0,2] = 0.5*col
    K[1,2] = 0.5*row
    return K



# compute features
sift.process_image('../data/book_frontal.JPG','im0.sift')
l0,d0 = sift.read_features_from_file('im0.sift')

sift.process_image('../data/book_perspective.JPG','im1.sift')
l1,d1 = sift.read_features_from_file('im1.sift')


# match features and estimate homography
matches = sift.match_twosided(d0,d1)
ndx = matches.nonzero()[0]
fp = homography.make_homog(l0[ndx,:2].T)
ndx2 = [int(matches[i]) for i in ndx]
tp = homography.make_homog(l1[ndx2,:2].T)

model = homography.RansacModel()
H, inliers = homography.H_from_ransac(fp,tp,model)

# camera calibration
K = my_calibration((747,1000))


def set_projection_from_camera(K):
    """ Set view from a camera calibration matrix. """
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    fx = K[0,0]
    fy = K[1,1]
    fovy = 2*npy.arctan(0.5*height/fy)*180/npy.pi
    aspect = (width*fy)/(height*fx)
    # define the near and far clipping planes
    near = 0.1
    far = 100.0
    # set perspective
    gluPerspective(fovy,aspect,near,far)
    glViewport(0,0,width,height)


def set_modelview_from_camera(Rt):
    """ Set the model view matrix from camera pose. """
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    # rotate teapot 90 deg around x-axis so that z-axis is up
    Rx = npy.array([[1,0,0],[0,0,-1],[0,1,0]])
    # set rotation to best approximation
    R = Rt[:,:3]
    U,S,V = linalg.svd(R)
    R = npy.dot(U,V)
    R[0,:] = -R[0,:] # change sign of x-axis
    # set translation
    t = Rt[:,3]
    # setup 4*4 model view matrix
    M = npy.eye(4)
    M[:3,:3] = npy.dot(R,Rx)
    M[:3,3] = t
    # transpose and flatten to get column order
    M = M.T
    m = M.flatten()
    # replace model view with the new matrix
    glLoadMatrixf(m)


def draw_background(imname):
    """ Draw background image using a quad. """
    # load background image (should be .bmp) to OpenGL texture
    bg_image = pygame.image.load(imname).convert()
    bg_data = pygame.image.tostring(bg_image,"RGBX",1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    # bind the texture
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D,glGenTextures(1))
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,width,height,0,GL_RGBA,GL_UNSIGNED_BYTE,bg_data)
    glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST)
    # create quad to fill the whole window
    glBegin(GL_QUADS)
    glTexCoord2f(0.0,0.0); glVertex3f(-1.0,-1.0,-1.0)
    glTexCoord2f(1.0,0.0); glVertex3f( 1.0,-1.0,-1.0)
    glTexCoord2f(1.0,1.0); glVertex3f( 1.0, 1.0,-1.0)
    glTexCoord2f(0.0,1.0); glVertex3f(-1.0, 1.0,-1.0)
    glEnd()
    # clear the texture
    glDeleteTextures(1)


def draw_teapot(size):
    """ Draw a red teapot at the origin. """
    glutInit()
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_DEPTH_TEST)
    glClear(GL_DEPTH_BUFFER_BIT)
    # draw red teapot
    glMaterialfv(GL_FRONT, GL_AMBIENT, [0, 0, 0, 0])
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.5, 0.0, 0.0, 0.0])
    glMaterialfv(GL_FRONT, GL_SPECULAR, [0.7, 0.6, 0.6, 0.0])
    glMaterialf(GL_FRONT, GL_SHININESS, 0.25 * 128.0)
    glutSolidTeapot(size)


def setup():
    """ Setup window and pygame environment. """
    pygame.init()
    pygame.display.set_mode((width,height),OPENGL | DOUBLEBUF)
    pygame.display.set_caption('OpenGL AR demo')


cam1 = camera.Camera(npy.hstack((K, npy.dot(K, npy.array([[0],[0],[-1]])))))

Rt = npy.hstack((K, npy.dot(K, npy.array([[0],[0],[-1]]))))

# load camera data
#with open('ar_camera.pkl', 'r') as f:
#    K = pickle.load(f)
#    Rt = pickle.load(f)

setup()
draw_background('../data/book_perspective.bmp')
set_projection_from_camera(K)
set_modelview_from_camera(Rt)
draw_teapot(0.02)

while True:
    event = pygame.event.poll()
    if event.type in (QUIT,KEYDOWN):
        break
    pygame.display.flip()