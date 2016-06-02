# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals
from __future__ import division, print_function

from PhloxAR.base import *
from PhloxAR.core.image import Image
from PhloxAR.features.feature_extractor_base import FeatureExtractorBase

__all__ = [
    'BOFFeatureExtractor'
]


class BOFFeatureExtractor(FeatureExtractorBase):
    """
    For a discussion of bag of features please see:
    http://en.wikipedia.org/wiki/Bag_of_words_model_in_computer_vision
    Initialize the bag of features extractor. This assumes you don't have
    the features codebook pre-computed.
    patchsz = the dimensions of each codebook patch
    numcodes = the number of different patches in the codebook.
    imglayout = the shape of the resulting image in terms of patches
    padding = the pixel padding of each patch in the resulting image.
    """
    _patch_size = (11, 11)
    _num_codes = 128
    _padding = 0
    _layout = (8, 16)
    _codebook_img = None
    _codebook = None

    def __init__(self, patchsz=(11, 11), numcodes=128, imglayout=(8, 16),
                 padding=0):

        self._padding = padding
        self._layout = imglayout
        self._patch_size = patchsz
        self._num_codes = numcodes

    def generate(self, imgdirs, numcodes=128, sz=(11, 11), imgs_per_dir=50,
                 img_layout=(8, 16), padding=0, verbose=True):
        """
        This method builds the bag of features codebook from a list of directories
        with images in them. Each directory should be broken down by image class.
        * imgdirs: This list of directories.
        * patchsz: the dimensions of each codebook patch
        * numcodes: the number of different patches in the codebook.
        * imglayout: the shape of the resulting image in terms of patches - this must
          match the size of numcodes. I.e. numcodes == img_layout[0]*img_layout[1]
        * padding:the pixel padding of each patch in the resulting image.
        * imgs_per_dir: this method can use a specified number of images per directory
        * verbose: print output
        Once the method has completed it will save the results to a local file
        using the file name codebook.png
        WARNING:
            THIS METHOD WILL TAKE FOREVER
        """
        if numcodes != img_layout[0] * img_layout[1]:
            warnings.warn("Numcodes must match the size of image layout.")
            return None

        self._padding = padding
        self._layout = img_layout
        self._num_codes = numcodes
        self._patch_size = sz
        rawFeatures = npy.zeros(
            sz[0] * sz[1])  # fakeout numpy so we can use vstack
        for path in imgdirs:
            fcount = 0
            files = []
            for ext in IMAGE_FORMATS:
                files.extend(glob.glob(os.path.join(path, ext)))
            nimgs = min(len(files), imgs_per_dir)
            for i in range(nimgs):
                infile = files[i]
                if verbose:
                    print(path + " " + str(i) + " of " + str(imgs_per_dir))
                    print("Opening file: " + infile)
                img = Image(infile)
                newFeat = self._get_patches(img, sz)
                if verbose:
                    print("     Got " + str(len(newFeat)) + " features.")
                rawFeatures = npy.vstack((rawFeatures, newFeat))
                del img
        rawFeatures = rawFeatures[1:, :]  # pop the fake value we put on the top
        if verbose:
            print("==================================")
            print("Got " + str(len(rawFeatures)) + " features ")
            print("Doing K-Means .... this will take a long time")
        self._codebook = self._make_codebook(rawFeatures, self._num_codes)
        self._codebook_img = self._codebook2img(self._codebook,
                                                self._patch_size,
                                                self._num_codes, self._layout,
                                                self._padding)
        self._codebook_img.save('codebook.png')

    def extract_patches(self, img, sz=(11, 11)):
        """
        Get patches from a single images. This is an external access method. The
        user will need to maintain the list of features. See the generate method
        as a guide to doing this by hand. Sz is the image patch size.
        """
        return self._get_patches(img, sz)

    def make_codebook(self, featureStack, ncodes=128):
        """
        This method will return the centroids of the k-means analysis of a large
        number of images. Ncodes is the number of centroids to find.
        """
        return self._make_codebook(featureStack, ncodes)

    def _make_codebook(self, data, ncodes=128):
        """
        Do the k-means ... this is slow as as shit
        """
        [centroids, membership] = scv.kmeans2(data, ncodes, minit='points')
        return centroids

    def _img2codebook(self, img, patchsize, count, patch_arrangement, spacersz):
        """
        img = the image
        patchsize = the patch size (ususally 11x11)
        count = total codes
        patch_arrangement = how are the patches grided in the image (eg 128 = (8x16) 256=(16x16) )
        spacersz = the number of pixels between patches
        """
        img = img.toHLS()
        lmat = cv.CreateImage((img.width, img.height), cv.IPL_DEPTH_8U, 1)
        patch = cv.CreateImage(patchsize, cv.IPL_DEPTH_8U, 1)
        cv.Split(img.getBitmap(), None, lmat, None, None)
        w = patchsize[0]
        h = patchsize[1]
        length = w * h
        ret = npy.zeros(length)
        for widx in range(patch_arrangement[0]):
            for hidx in range(patch_arrangement[1]):
                x = (widx * patchsize[0]) + ((widx + 1) * spacersz)
                y = (hidx * patchsize[1]) + ((hidx + 1) * spacersz)
                cv.SetImageROI(lmat, (x, y, w, h))
                cv.Copy(lmat, patch)
                cv.ResetImageROI(lmat)
                ret = npy.vstack((ret, npy.array(patch[:, :]).reshape(length)))
        ret = ret[1:, :]
        return ret

    def _codebook2img(self, cb, patchsize, count, patch_arrangement, spacersz):
        """
        cb = the codebook
        patchsize = the patch size (ususally 11x11)
        count = total codes
        patch_arrangement = how are the patches grided in the image (eg 128 = (8x16) 256=(16x16) )
        spacersz = the number of pixels between patches
        """
        w = (patchsize[0] * patch_arrangement[0]) + (
            (patch_arrangement[0] + 1) * spacersz)
        h = (patchsize[1] * patch_arrangement[1]) + (
            (patch_arrangement[1] + 1) * spacersz)
        bm = cv.CreateImage((w, h), cv.IPL_DEPTH_8U, 1)
        cv.Zero(bm)
        img = Image(bm)
        count = 0
        for widx in range(patch_arrangement[0]):
            for hidx in range(patch_arrangement[1]):
                x = (widx * patchsize[0]) + ((widx + 1) * spacersz)
                y = (hidx * patchsize[1]) + ((hidx + 1) * spacersz)
                temp = Image(cb[count, :].reshape(patchsize[0], patchsize[1]))
                img.blit(temp, pos=(x, y))
                count += 1
        return img

    def _get_patches(self, img, sz=None):
        if sz is None:
            sz = self._patch_size
        img2 = img.toHLS()
        lmat = cv.CreateImage((img.width, img.height), cv.IPL_DEPTH_8U, 1)
        patch = cv.CreateImage(self._patch_size, cv.IPL_DEPTH_8U, 1)
        cv.Split(img2.getBitmap(), None, lmat, None, None)
        wsteps = img2.width / sz[0]
        hsteps = img2.height / sz[1]
        w = sz[0]
        h = sz[1]
        length = w * h
        ret = npy.zeros(length)
        for widx in range(wsteps):
            for hidx in range(hsteps):
                x = (widx * sz[0])
                y = (hidx * sz[1])
                cv.SetImageROI(lmat, (x, y, w, h))
                cv.EqualizeHist(lmat, patch)
                # cv.Copy(lmat,patch)
                cv.ResetImageROI(lmat)

                ret = npy.vstack((ret, npy.array(patch[:, :]).reshape(length)))
                # ret.append()
        ret = ret[1:, :]  # pop the fake value we put on top of the stack
        return ret

    def load(self, datafile):
        """
        Load a codebook from file using the datafile. The datafile
        should point to a local image for the source patch image.
        """
        myFile = open(datafile, 'r')
        temp = myFile.readline()
        # print(temp)
        self._num_codes = int(myFile.readline())
        # print(self._num_codes)
        w = int(myFile.readline())
        h = int(myFile.readline())
        self._patch_size = (w, h)
        # print(self._patch_size)
        self._padding = int(myFile.readline())
        # print(self._padding)
        w = int(myFile.readline())
        h = int(myFile.readline())
        self._layout = (w, h)
        # print(self._layout)
        imgfname = myFile.readline().strip()
        # print(imgfname)
        self._codebook_img = Image(imgfname)
        self._codebook = self._img2codebook(self._codebook_img,
                                            self._patch_size,
                                            self._num_codes,
                                            self._layout,
                                            self._padding)
        # print(self._codebook)
        return

    def save(self, imgfname, datafname):
        """
        Save the bag of features codebook and data set to a local file.
        """
        myFile = open(datafname, 'w')
        myFile.write("BOF Codebook Data\n")
        myFile.write(str(self._num_codes) + "\n")
        myFile.write(str(self._patch_size[0]) + "\n")
        myFile.write(str(self._patch_size[1]) + "\n")
        myFile.write(str(self._padding) + "\n")
        myFile.write(str(self._layout[0]) + "\n")
        myFile.write(str(self._layout[1]) + "\n")
        myFile.write(imgfname + "\n")
        myFile.close()
        if self._codebook_img is None:
            self._codebook2img(self._codebook, self._patch_size,
                               self._num_codes, self._layout, self._padding)
        self._codebook_img.save(imgfname)
        return

    def __getstate__(self):
        if self._codebook_img is None:
            self._codebook2img(self._codebook, self._patch_size,
                               self._num_codes, self._layout, self._padding)
        attr = self.__dict__.copy()
        del attr['_codebook']
        return attr

    def __setstate__(self, state):
        self.__dict__ = state
        self._codebook = self._img2codebook(self._codebook_img,
                                            self._patch_size,
                                            self._num_codes,
                                            self._layout,
                                            self._padding)

    def extract(self, img):
        """
        This method extracts a bag of features histogram for the input image using
        the provided codebook. The result are the bin counts for each codebook code.
        """
        data = self._get_patches(img)
        p = spsd.cdist(data, self._codebook)
        codes = npy.argmin(p, axis=1)
        [ret, foo] = npy.histogram(codes, self._num_codes, normed=True,
                                      range=(0, self._num_codes - 1))
        return ret

    def reconstruct(self, img):
        """
        This is a "just for fun" method as a sanity check for the BOF codeook.
        The method takes in an image, extracts each codebook code, and replaces
        the image at the position with the code.
        """
        ret = cv.CreateImage((img.width, img.height), cv.IPL_DEPTH_8U, 1)
        data = self._get_patches(img)
        p = spsd.cdist(data, self._codebook)
        foo = p.shape[0]
        codes = npy.argmin(p, axis=1)
        count = 0
        wsteps = img.width / self._patch_size[0]
        hsteps = img.height / self._patch_size[1]
        w = self._patch_size[0]
        h = self._patch_size[1]
        length = w * h
        ret = Image(ret)
        for widx in range(wsteps):
            for hidx in range(hsteps):
                x = (widx * self._patch_size[0])
                y = (hidx * self._patch_size[1])
                p = codes[count]
                temp = Image(self._codebook[p, :].reshape(self._patch_size[0],
                                                          self._patch_size[1]))
                ret = ret.blit(temp, pos=(x, y))
                count += 1
        return ret

    def get_field_names(self):
        """
        This method gives the names of each field in the features vector in the
        order in which they are returned. For example, 'xpos' or 'width'
        """
        ret = []
        for widx in range(self._layout[0]):
            for hidx in range(self._layout[1]):
                temp = "CB_R" + str(widx) + "_C" + str(hidx)
                ret.append(temp)
        return ret

    def get_num_fields(self):
        """
        This method returns the total number of fields in the features vector.
        """
        return self._num_codes
