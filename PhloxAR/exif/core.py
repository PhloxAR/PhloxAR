# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from .utils import s2n_motorola, s2n_intel, Ratio, get_logger, make_string
from tags import *
import makernote

import sys
import getopt
import logging
import timeit

import struct
import re


__version__ = '0.2.0'

logger = get_logger()

try:
    basestring
except NameError:
    basestring = str


class IfdTag(object):
    """
    Eases dealing with tags.
    """
    def __init__(self, printable, tag, field_type, values, field_offset,
                 field_length):
        self._printable = printable
        self._tag = tag
        self._field_type = field_type
        self._field_length = field_length
        self._field_offset = field_offset
        self._values = values

    def __str__(self):
        return self._printable

    def __repr__(self):
        try:
            s = '(0x%04X) %s=%s @ %d' % (self._tag,
                                         FIELD_TYPES[self._field_type][2],
                                         self._printable,
                                         self._field_offset)
        except:
            s = '(%s) %s=%s @ %s' % (str(self._tag),
                                     FIELD_TYPES[self._field_type][2],
                                     self._printable,
                                     str(self._field_offset))
        return s


class ExifHeader(object):
    """
    Handle an EXIF header.
    """
    def __init__(self, filename, endian, offset, fake_exif, strict,
                 debug=False, detailed=True):
        self._file = filename
        self._endian = endian
        self._offset = offset
        self._fake_exif = fake_exif
        self._debug = debug
        self._detailed = detailed
        self._tags = {}

    def s2n(self, offset, length, signed=0):
        """
        Convert slice to integer, base on sign and endian flags.

        Usually this offset is assumed to be relative to the beginning of
        the start of the EXIF information.
        For some cameras that use relative tags, this offset may be
        relative to some other starting point.
        """
        self._file.seek(self._offset + offset)
        sliced = self._file.read(length)

        if self._endian == 'I':
            val = s2n_intel(sliced)
        else:
            val = s2n_motorola(sliced)

        if signed:
            msb = 1 << (8 * length - 1)
            if val & msb:
                val -= (msb << 1)

        return val

    def n2s(self, offset, length):
        """
        Convert offset to string.
        """
        s = ''
        for i in range(length):
            if self._endian == 'I':
                s += chr(offset & 0xFF)
            else:
                s = chr(offset & 0xFF) + s

            offset >>= 8

        return s

    def _first_ifd(self):
        """
        Return first IFD.
        """
        return self.s2n(4, 4)

    def _next_ifd(self, ifd):
        """
        Returns the pointer to next IFD.
        """
        entries = self.s2n(ifd, 2)
        next_ifd = self.s2n(ifd + 2 + 12 * entries, 4)
        if next_ifd == ifd:
            return 0
        else:
            return next_ifd

    def list_ifd(self):
        """
        Return the list of IFDs in the header.
        """
        i = self._first_ifd()
        ifds = []
        while i:
            ifds.append(i)
            i = self._next_ifd(i)
        return ifds

    def dump_ifd(self, ifd, ifd_name, tag_dict=EXIF_TAGS, relative=0,
                 stop_tag=DEFAULT_STOP_TAG):
        """
        Return a list of entries in the given IFD.
        """
        # make sure we can process the entries
        try:
            entries = self.s2n(ifd, 2)
        except TypeError:
            logger.warning("Possibly corrupted IFD: %s" % ifd)
            return

        for i in range(entries):
            # entry is index of start of this IFD in the file
            entry = ifd + 2 + 12 * i
            tag = self.s2n(entry, 2)

            # get tag name early to avoid errors, help debug
            tag_entry = tag_dict.get(tag)
            if tag_entry:
                tag_name = tag_entry[0]
            else:
                tag_name = 'Tag 0x%04X' % tag

            # ignore certain tags for faster processing
            if not (not self._detailed and tag in IGNORE_TAGS):
                field_type = self.s2n(entry + 2, 2)

                # unknown field type
                if not 0 < field_type < len(FIELD_TYPES):
                    if not self.strict:
                        continue
                    else:
                        raise ValueError('Unknown type %d in tag 0x%04X' %
                                         (field_type, tag))

                type_length = FIELD_TYPES[field_type][0]
                count = self.s2n(entry + 4, 4)
                # Adjust for tag id/type/count (2+2+4 bytes)
                # Now we point at either the data or the 2nd level offset
                offset = entry + 8

                # If the value fits in 4 bytes, it is inlined, else we
                # need to jump ahead again.
                if count * type_length > 4:
                    # offset is not the value; it's a pointer to the value
                    # if relative we set things up so s2n will seek to the
                    # right place when it adds self._offset.  Note that this
                    # 'relative' is for the Nikon type 3 makernote.  Other
                    # cameras may use other relative offsets, which would
                    # have to be computed here slightly differently.
                    if relative:
                        tmp_offset = self.s2n(offset, 4)
                        offset = tmp_offset + ifd - 8
                        if self._fake_exif:
                            offset += 18
                    else:
                        offset = self.s2n(offset, 4)

                field_offset = offset
                values = None
                if field_type == 2:
                    # special case: null-terminated ASCII string
                    # XXX investigate
                    # sometimes gets too big to fit in int value
                    # 2E31 is hardware dependant. --gd
                    if count != 0:  # and count < (2**31):
                        file_position = self._offset + offset
                        try:
                            self._file.seek(file_position)
                            values = self._file.read(count)
                            # Drop any garbage after a null.
                            values = values.split(b'\x00', 1)[0]
                            if isinstance(values, bytes):
                                try:
                                    values = values.decode("utf-8")
                                except UnicodeDecodeError:
                                    logger.warning("Possibly corrupted field %s in %s IFD",
                                                   tag_name, ifd_name)
                        except OverflowError:
                            logger.warn('OverflowError at position: %s, length: %s',
                                        file_position, count)
                            values = ''
                        except MemoryError:
                            logger.warn('MemoryError at position: %s, length: %s',
                                        file_position, count)
                            values = ''
                else:
                    values = []
                    signed = (field_type in [6, 8, 9, 10])

                    # XXX investigate
                    # some entries get too big to handle could be malformed
                    # file or problem with self.s2n
                    if count < 1000:
                        for dummy in range(count):
                            if field_type in (5, 10):
                                # a ratio
                                value = Ratio(self.s2n(offset, 4, signed),
                                              self.s2n(offset + 4, 4, signed))
                            else:
                                value = self.s2n(offset, type_length, signed)
                            values.append(value)
                            offset = offset + type_length
                    # The test above causes problems with tags that are
                    # supposed to have long values! Fix up one important case.
                    elif tag_name in ('MakerNote',
                                      makernote.canon.CAMERA_INFO_TAG_NAME):
                        for dummy in range(count):
                            value = self.s2n(offset, type_length, signed)
                            values.append(value)
                            offset = offset + type_length

                # now 'values' is either a string or an array
                if count == 1 and field_type != 2:
                    printable = str(values[0])
                elif count > 50 and len(values) > 20 and not isinstance(values, basestring):
                    printable = str(values[0:20])[0:-1] + ", ... ]"
                else:
                    try:
                        printable = str(values)
                    except UnicodeEncodeError:
                        printable = unicode(values)
                # compute printable version of values
                if tag_entry:
                    # optional 2nd tag element is present
                    if len(tag_entry) != 1:
                        if callable(tag_entry[1]):
                            # call mapping function
                            printable = tag_entry[1](values)
                        elif type(tag_entry[1]) is tuple:
                            ifd_info = tag_entry[1]
                            try:
                                logger.debug('%s SubIFD at offset %d:',
                                             ifd_info[0], values[0])
                                self.dump_ifd(values[0], ifd_info[0],
                                              tag_dict=ifd_info[1],
                                              stop_tag=stop_tag)
                            except IndexError:
                                logger.warn('No values found for %s SubIFD',
                                            ifd_info[0])
                        else:
                            printable = ''
                            for i in values:
                                # use lookup table for this tag
                                printable += tag_entry[1].get(i, repr(i))

                self._tags[ifd_name + ' ' + tag_name] = IfdTag(printable, tag,
                                                              field_type,
                                                              values,
                                                              field_offset,
                                                              count * type_length)
                try:
                    tag_value = repr(self._tags[ifd_name + ' ' + tag_name])
                # fix for python2's handling of unicode values
                except UnicodeEncodeError:
                    tag_value = unicode(self._tags[ifd_name + ' ' + tag_name])
                logger.debug(' %s: %s', tag_name, tag_value)

            if tag_name == stop_tag:
                break

    def extract_tiff_thumbnail(self, thumb_ifd):
        """
        Extract uncompressed TIFF thumbnail.
        Take advantage of the pre-existing layout in the thumbnail IFD as
        much as possible
        """
        thumb = self._tags.get('Thumbnail Compression')
        if not thumb or thumb._printable != 'Uncompressed TIFF':
            return

        entries = self.s2n(thumb_ifd, 2)
        # this is header plus offset to IFD ...
        if self._endian == 'M':
            tiff = 'MM\x00*\x00\x00\x00\x08'
        else:
            tiff = 'II*\x00\x08\x00\x00\x00'
            # ... plus thumbnail IFD data plus a null "next IFD" pointer
        self._file.seek(self._offset + thumb_ifd)
        tiff += self._file.read(entries * 12 + 2) + '\x00\x00\x00\x00'

        # fix up large value offset pointers into data area
        for i in range(entries):
            entry = thumb_ifd + 2 + 12 * i
            tag = self.s2n(entry, 2)
            field_type = self.s2n(entry + 2, 2)
            type_length = FIELD_TYPES[field_type][0]
            count = self.s2n(entry + 4, 4)
            old_offset = self.s2n(entry + 8, 4)
            # start of the 4-byte pointer area in entry
            ptr = i * 12 + 18
            # remember strip offsets location
            if tag == 0x0111:
                strip_off = ptr
                strip_len = count * type_length
                # is it in the data area?
            if count * type_length > 4:
                # update offset pointer (nasty "strings are immutable" crap)
                # should be able to say "tiff[ptr:ptr+4]=newoff"
                newoff = len(tiff)
                tiff = tiff[:ptr] + self.n2s(newoff, 4) + tiff[ptr + 4:]
                # remember strip offsets location
                if tag == 0x0111:
                    strip_off = newoff
                    strip_len = 4
                # get original data and store it
                self._file.seek(self._offset + old_offset)
                tiff += self._file.read(count * type_length)

        # add pixel strips and update strip offset info
        old_offsets = self._tags['Thumbnail StripOffsets'].values
        old_counts = self._tags['Thumbnail StripByteCounts'].values
        for i in range(len(old_offsets)):
            # update offset pointer (more nasty "strings are immutable" crap)
            offset = self.n2s(len(tiff), strip_len)
            tiff = tiff[:strip_off] + offset + tiff[strip_off + strip_len:]
            strip_off += strip_len
            # add pixel strip to end
            self._file.seek(self._offset + old_offsets[i])
            tiff += self._file.read(old_counts[i])

        self._tags['TIFFThumbnail'] = tiff

    def extract_jpeg_thumbnail(self):
        """
        Extract JPEG thumbnail.
        (Thankfully the JPEG data is stored as a unit.)
        """
        thumb_offset = self._tags.get('Thumbnail JPEGInterchangeFormat')
        if thumb_offset:
            self._file.seek(self._offset + thumb_offset._values[0])
            size = self._tags['Thumbnail JPEGInterchangeFormatLength']._values[0]
            self._tags['JPEGThumbnail'] = self._file.read(size)

        # Sometimes in a TIFF file, a JPEG thumbnail is hidden in the MakerNote
        # since it's not allowed in a uncompressed TIFF IFD
        if 'JPEGThumbnail' not in self._tags:
            thumb_offset = self._tags.get('MakerNote JPEGThumbnail')
            if thumb_offset:
                self._file.seek(self._offset + thumb_offset._values[0])
                self._tags['JPEGThumbnail'] = self._file.read(thumb_offset._field_length)

    def decode_maker_note(self):
        """
        Decode all the camera-specific MakerNote formats
        Note is the data that comprises this MakerNote.
        The MakerNote will likely have pointers in it that point to other
        parts of the file. We'll use self._offset as the starting point for
        most of those pointers, since they are relative to the beginning
        of the file.
        If the MakerNote is in a newer format, it may use relative addressing
        within the MakerNote. In that case we'll use relative addresses for
        the pointers.
        As an aside: it's not just to be annoying that the manufacturers use
        relative offsets.  It's so that if the makernote has to be moved by the
        picture software all of the offsets don't have to be adjusted.
        Overall, this is probably the right strategy for makernotes, though the
        spec is ambiguous.
        The spec does not appear to imagine that makernotes would
        follow EXIF format internally.  Once they did, it's ambiguous whether
        the offsets should be from the header at the start of all the EXIF
        info, or from the header at the start of the makernote.
        """
        note = self._tags['EXIF MakerNote']

        # Some apps use MakerNote tags but do not use a format for which we
        # have a description, so just do a raw dump for these.
        make = self._tags['Image Make']._printable

        # Nikon
        # The maker note usually starts with the word Nikon, followed by the
        # type of the makernote (1 or 2, as a short).  If the word Nikon is
        # not at the start of the makernote, it's probably type 2, since some
        # cameras work that way.
        if 'NIKON' in make:
            if note._values[0:7] == [78, 105, 107, 111, 110, 0, 1]:
                logger.debug("Looks like a type 1 Nikon MakerNote.")
                self.dump_ifd(note._field_offset + 8, 'MakerNote',
                              tag_dict=makernote.nikon.TAGS_OLD)
            elif note._values[0:7] == [78, 105, 107, 111, 110, 0, 2]:
                logger.debug("Looks like a labeled type 2 Nikon MakerNote")
                if note._values[12:14] != [0, 42] and note._values[12:14] != [42, 0]:
                    raise ValueError("Missing marker tag '42' in MakerNote.")
                    # skip the Makernote label and the TIFF header
                self.dump_ifd(note._field_offset + 10 + 8, 'MakerNote',
                              tag_dict=makernote.nikon.TAGS_NEW, relative=1)
            else:
                # E99x or D1
                logger.debug("Looks like an unlabeled type 2 Nikon MakerNote")
                self.dump_ifd(note._field_offset, 'MakerNote',
                              tag_dict=makernote.nikon.TAGS_NEW)
            return

        # Olympus
        if make.startswith('OLYMPUS'):
            self.dump_ifd(note._field_offset + 8, 'MakerNote',
                          tag_dict=makernote.olympus.TAGS)
            # TODO
            # for i in (('MakerNote Tag 0x2020', makernote.OLYMPUS_TAG_0x2020)):
            #    self.decode_olympus_tag(self._tags[i[0]].values, i[1])
            # return

        # Casio
        if 'CASIO' in make or 'Casio' in make:
            self.dump_ifd(note._field_offset, 'MakerNote',
                          tag_dict=makernote.casio.TAGS)
            return

        # Fujifilm
        if make == 'FUJIFILM':
            # bug: everything else is "Motorola" endian, but the MakerNote
            # is "Intel" endian
            endian = self._endian
            self._endian = 'I'
            # bug: IFD offsets are from beginning of MakerNote, not
            # beginning of file header
            offset = self._offset
            self._offset += note._field_offset
            # process note with bogus values (note is actually at offset 12)
            self.dump_ifd(12, 'MakerNote', tag_dict=makernote.fujifilm.TAGS)
            # reset to correct values
            self._endian = endian
            self._offset = offset
            return

        # Apple
        if (make == 'Apple' and note.values[0:10] == [
            65, 112, 112, 108, 101, 32, 105, 79, 83, 0
        ]):
            t = self._offset
            self._offset += note._field_offset+14
            self.dump_ifd(0, 'MakerNote',
                          tag_dict=makernote.apple.TAGS)
            self._offset = t
            return

        # Canon
        if make == 'Canon':
            self.dump_ifd(note._field_offset, 'MakerNote',
                          tag_dict=makernote.canon.TAGS)

            for i in (('MakerNote Tag 0x0001', makernote.canon.CAMERA_SETTINGS),
                      ('MakerNote Tag 0x0002', makernote.canon.FOCAL_LENGTH),
                      ('MakerNote Tag 0x0004', makernote.canon.SHOT_INFO),
                      ('MakerNote Tag 0x0026', makernote.canon.AF_INFO_2),
                      ('MakerNote Tag 0x0093', makernote.canon.FILE_INFO)):
                if i[0] in self._tags:
                    logger.debug('Canon ' + i[0])
                    self._canon_decode_tag(self._tags[i[0]].values, i[1])
                    del self._tags[i[0]]
            if makernote.canon.CAMERA_INFO_TAG_NAME in self._tags:
                tag = self._tags[makernote.canon.CAMERA_INFO_TAG_NAME]
                logger.debug('Canon CameraInfo')
                self._canon_decode_camera_info(tag)
                del self._tags[makernote.canon.CAMERA_INFO_TAG_NAME]
            return

    def _olympus_decode_tag(self, value, mn_tags):
        """ TODO Decode Olympus MakerNote tag based on offset within tag."""
        pass

    def _canon_decode_tag(self, value, mn_tags):
        """
        Decode Canon MakerNote tag based on offset within tag.
        See http://www.burren.cx/david/canon.html by David Burren
        """
        for i in range(1, len(value)):
            tag = mn_tags.get(i, ('Unknown', ))
            name = tag[0]
            if len(tag) > 1:
                val = tag[1].get(value[i], 'Unknown')
            else:
                val = value[i]
            try:
                logger.debug(" %s %s %s", i, name, hex(value[i]))
            except TypeError:
                logger.debug(" %s %s %s", i, name, value[i])

            # it's not a real IFD Tag but we fake one to make everybody
            # happy. this will have a "proprietary" type
            self._tags['MakerNote ' + name] = IfdTag(str(val), None, 0, None,
                                                    None, None)

    def _canon_decode_camera_info(self, camera_info_tag):
        """
        Decode the variable length encoded camera info section.
        """
        model = self._tags.get('Image Model', None)
        if not model:
            return
        model = str(model.values)

        camera_info_tags = None
        for (model_name_re, tag_desc) in makernote.canon.CAMERA_INFO_MODEL_MAP.items():
            if re.search(model_name_re, model):
                camera_info_tags = tag_desc
                break
        else:
            return

        # We are assuming here that these are all unsigned bytes (Byte or
        # Unknown)
        if camera_info_tag.field_type not in (1, 7):
            return
        camera_info = struct.pack('<%dB' % len(camera_info_tag.values),
                                  *camera_info_tag.values)

        # Look for each data value and decode it appropriately.
        for offset, tag in camera_info_tags.items():
            tag_format = tag[1]
            tag_size = struct.calcsize(tag_format)
            if len(camera_info) < offset + tag_size:
                continue
            packed_tag_value = camera_info[offset:offset + tag_size]
            tag_value = struct.unpack(tag_format, packed_tag_value)[0]

            tag_name = tag[0]
            if len(tag) > 2:
                if callable(tag[2]):
                    tag_value = tag[2](tag_value)
                else:
                    tag_value = tag[2].get(tag_value, tag_value)
            logger.debug(" %s %s", tag_name, tag_value)

            self._tags['MakerNote ' + tag_name] = IfdTag(str(tag_value), None,
                                                        0, None, None, None)

    def parse_xmp(self, xmp_string):
        import xml.dom.minidom

        logger.debug('XMP cleaning data')

        xml = xml.dom.minidom.parseString(xmp_string)
        pretty = xml.toprettyxml()
        cleaned = []
        for line in pretty.splitlines():
            if line.strip():
                cleaned.append(line)
        self._tags['Image ApplicationNotes'] = IfdTag('\n'.join(cleaned), None,
                                                     1, None, None, None)


def increment_base(data, base):
    return ord(data[base + 2]) * 256 + ord(data[base + 3]) + 2


def process_file(f, stop_tag=DEFAULT_STOP_TAG, details=True, strict=False, debug=False):
    """
    Process an image file (expects an open file object).
    This is the function that has to deal with all the arbitrary nasty bits
    of the EXIF standard.
    """

    # by default do not fake an EXIF beginning
    fake_exif = 0

    # determine whether it's a JPEG or TIFF
    data = f.read(12)
    if data[0:4] in [b'II*\x00', b'MM\x00*']:
        # it's a TIFF file
        logger.debug("TIFF format recognized in data[0:4]")
        f.seek(0)
        endian = f.read(1)
        f.read(1)
        offset = 0
    elif data[0:2] == b'\xFF\xD8':
        # it's a JPEG file
        logger.debug("JPEG format recognized data[0:2]=0x%X%X", ord(data[0]),
                     ord(data[1]))
        base = 2
        logger.debug("data[2]=0x%X data[3]=0x%X data[6:10]=%s",
                     ord(data[2]), ord(data[3]), data[6:10])
        while ord(data[2]) == 0xFF and data[6:10] in (b'JFIF', b'JFXX',
                                                      b'OLYM', b'Phot'):
            length = ord(data[4]) * 256 + ord(data[5])
            logger.debug(" Length offset is %s", length)
            f.read(length - 8)
            # fake an EXIF beginning of file
            # I don't think this is used. --gd
            data = b'\xFF\x00' + f.read(10)
            fake_exif = 1
            if base > 2:
                logger.debug(" Added to base")
                base = base + length + 4 - 2
            else:
                logger.debug(" Added to zero")
                base = length + 4
            logger.debug(" Set segment base to 0x%X", base)

        # Big ugly patch to deal with APP2 (or other) data coming before APP1
        f.seek(0)
        # in theory, this could be insufficient since 64K is the maximum size--gd
        data = f.read(base + 4000)
        # base = 2
        while 1:
            logger.debug(" Segment base 0x%X", base)
            if data[base:base + 2] == b'\xFF\xE1':
                # APP1
                logger.debug("  APP1 at base 0x%X", base)
                logger.debug("  Length: 0x%X 0x%X", ord(data[base + 2]),
                             ord(data[base + 3]))
                logger.debug("  Code: %s", data[base + 4:base + 8])
                if data[base + 4:base + 8] == b"Exif":
                    logger.debug("  Decrement base by 2 to get to pre-segment header (for compatibility with later code)")
                    base -= 2
                    break
                increment = increment_base(data, base)
                logger.debug(" Increment base by %s", increment)
                base += increment
            elif data[base:base + 2] == b'\xFF\xE0':
                # APP0
                logger.debug("  APP0 at base 0x%X", base)
                logger.debug("  Length: 0x%X 0x%X", ord(data[base + 2]),
                             ord(data[base + 3]))
                logger.debug("  Code: %s", data[base + 4:base + 8])
                increment = increment_base(data, base)
                logger.debug(" Increment base by %s", increment)
                base += increment
            elif data[base:base + 2] == b'\xFF\xE2':
                # APP2
                logger.debug("  APP2 at base 0x%X", base)
                logger.debug("  Length: 0x%X 0x%X", ord(data[base + 2]),
                             ord(data[base + 3]))
                logger.debug(" Code: %s", data[base + 4:base + 8])
                increment = increment_base(data, base)
                logger.debug(" Increment base by %s", increment)
                base += increment
            elif data[base:base + 2] == b'\xFF\xEE':
                # APP14
                logger.debug("  APP14 Adobe segment at base 0x%X", base)
                logger.debug("  Length: 0x%X 0x%X", ord(data[base + 2]),
                             ord(data[base + 3]))
                logger.debug("  Code: %s", data[base + 4:base + 8])
                increment = increment_base(data, base)
                logger.debug(" Increment base by %s", increment)
                base += increment
                logger.debug("  There is useful EXIF-like data here, but we "
                             "have no parser for it.")
            elif data[base:base + 2] == b'\xFF\xDB':
                logger.debug("  JPEG image data at base 0x%X No more segments "
                             "are expected.",
                             base)
                break
            elif data[base:base + 2] == b'\xFF\xD8':
                # APP12
                logger.debug("  FFD8 segment at base 0x%X", base)
                logger.debug("  Got 0x%X 0x%X and %s instead",
                             ord(data[base]),
                             ord(data[base + 1]),
                             data[4 + base:10 + base])
                logger.debug("  Length: 0x%X 0x%X", ord(data[base + 2]),
                             ord(data[base + 3]))
                logger.debug("  Code: %s", data[base + 4:base + 8])
                increment = increment_base(data, base)
                logger.debug("  Increment base by %s", increment)
                base += increment
            elif data[base:base + 2] == b'\xFF\xEC':
                # APP12
                logger.debug("  APP12 XMP (Ducky) or Pictureinfo segment at "
                             "base 0x%X",
                             base)
                logger.debug("  Got 0x%X and 0x%X instead", ord(data[base]),
                             ord(data[base + 1]))
                logger.debug("  Length: 0x%X 0x%X",
                             ord(data[base + 2]),
                             ord(data[base + 3]))
                logger.debug("Code: %s", data[base + 4:base + 8])
                increment = increment_base(data, base)
                logger.debug("  Increment base by %s", increment)
                base += increment
                logger.debug(
                    "  There is useful EXIF-like data here (quality, comment, "
                    "copyright), but we have no parser for it.")
            else:
                try:
                    increment = increment_base(data, base)
                    logger.debug("  Got 0x%X and 0x%X instead",
                                 ord(data[base]),
                                 ord(data[base + 1]))
                except IndexError:
                    logger.debug("  Unexpected/unhandled segment type or file"
                                 " content.")
                    return {}
                else:
                    logger.debug("  Increment base by %s", increment)
                    base += increment
        f.seek(base + 12)
        if ord(data[2 + base]) == 0xFF and data[6 + base:10 + base] == b'Exif':
            # detected EXIF header
            offset = f.tell()
            endian = f.read(1)
            # HACK TEST:  endian = 'M'
        elif ord(data[2 + base]) == 0xFF and data[6 + base:10 + base + 1] == b'Ducky':
            # detected Ducky header.
            logger.debug("EXIF-like header (normally 0xFF and code): 0x%X and"
                         " %s", ord(data[2 + base]), data[6 + base:10 + base + 1])
            offset = f.tell()
            endian = f.read(1)
        elif ord(data[2 + base]) == 0xFF and data[6 + base:10 + base + 1] == b'Adobe':
            # detected APP14 (Adobe)
            logger.debug("EXIF-like header (normally 0xFF and code): 0x%X and "
                         "%s", ord(data[2 + base]), data[6 + base:10 + base + 1])
            offset = f.tell()
            endian = f.read(1)
        else:
            # no EXIF information
            logger.debug("No EXIF header expected data[2+base]==0xFF and "
                         "data[6+base:10+base]===Exif (or Duck)")
            logger.debug("Did get 0x%X and %s",
                         ord(data[2 + base]), data[6 + base:10 + base + 1])
            return {}
    else:
        # file format not recognized
        logger.debug("File format not recognized.")
        return {}

    endian = chr(ord(endian[0]))
    # deal with the EXIF info we found
    logger.debug("Endian format is %s (%s)", endian, {
        'I': 'Intel',
        'M': 'Motorola',
        '\x01': 'Adobe Ducky',
        'd': 'XMP/Adobe unknown'
    }[endian])

    hdr = ExifHeader(f, endian, offset, fake_exif, strict, debug, details)
    ifd_list = hdr.list_ifd()
    thumb_ifd = False
    ctr = 0
    for ifd in ifd_list:
        if ctr == 0:
            ifd_name = 'Image'
        elif ctr == 1:
            ifd_name = 'Thumbnail'
            thumb_ifd = ifd
        else:
            ifd_name = 'IFD %d' % ctr
        logger.debug('IFD %d (%s) at offset %s:', ctr, ifd_name, ifd)
        hdr.dump_ifd(ifd, ifd_name, stop_tag=stop_tag)
        ctr += 1
    # EXIF IFD
    exif_off = hdr._tags.get('Image ExifOffset')
    if exif_off:
        logger.debug('Exif SubIFD at offset %s:', exif_off._values[0])
        hdr.dump_ifd(exif_off._values[0], 'EXIF', stop_tag=stop_tag)

    # deal with MakerNote contained in EXIF IFD
    # (Some apps use MakerNote tags but do not use a format for which we
    # have a description, do not process these).
    if details and 'EXIF MakerNote' in hdr._tags and 'Image Make' in hdr._tags:
        hdr.decode_maker_note()

    # extract thumbnails
    if details and thumb_ifd:
        hdr.extract_tiff_thumbnail(thumb_ifd)
        hdr.extract_jpeg_thumbnail()

    # parse XMP tags (experimental)
    if debug and details:
        xmp_string = b''
        # Easy we already have them
        if 'Image ApplicationNotes' in hdr.tags:
            logger.debug('XMP present in Exif')
            xmp_string = make_string(hdr.tags['Image ApplicationNotes'].values)
        # We need to look in the entire file for the XML
        else:
            logger.debug('XMP not in Exif, searching file for XMP info...')
            xml_started = False
            xml_finished = False
            for line in f:
                open_tag = line.find(b'<x:xmpmeta')
                close_tag = line.find(b'</x:xmpmeta>')

                if open_tag != -1:
                    xml_started = True
                    line = line[open_tag:]
                    logger.debug('XMP found opening tag at line '
                                 'position %s' % open_tag)

                if close_tag != -1:
                    logger.debug('XMP found closing tag at line '
                                 'position %s' % close_tag)
                    line_offset = 0
                    if open_tag != -1:
                        line_offset = open_tag
                    line = line[:(close_tag - line_offset) + 12]
                    xml_finished = True

                if xml_started:
                    xmp_string += line

                if xml_finished:
                    break

            logger.debug('XMP Finished searching for info')
        if xmp_string:
            hdr.parse_xmp(xmp_string)

    return hdr._tags
