# -*- coding: utf-8 -*-

from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals
# TODO: more detailed
from PhloxAR.base import SocketServer
from PhloxAR.base import SimpleHTTPRequestHandler
from PhloxAR.base import time
from PhloxAR.base import socket
from PhloxAR.base import re
from PhloxAR.base import warnings
from PhloxAR.base import threading


_jpeg_streamers = {}
class JpegStreamHandler(SimpleHTTPRequestHandler):
    """
    Handles requests to the threaded HTTP server.
    Once initialized, any request to this port will receive
    a multipart/replace jpeg.
    """
    def get(self):
        global _jpeg_streamers

        if self.path == '/' or not self.path:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write("""
            <html>
                <head>
                    <style type=text/css>
                       body {
                           background-image: url(/stream);
                           background-repeat: no-repeat;
                           background-position: center top;
                           background-attachment: fixed;
                           height: 100%;
                       }
                    </style>
                </head>
                <body>
                &nbsp
                </body>
            </html>
            """)
            return
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Connection', 'close')
            self.send_header('Max-Age', '0')
            self.send_header('Expires', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=--BOUNDARYSTRING')
            self.end_headers()
            host, port = self.server.socket.getsockname()[:2]

            count = 0
            timeout = 0.75
            last_time_served = 0

            while True:
                if (_jpeg_streamers[port].refreshtime > last_time_served
                    or time.time() - timeout > last_time_served):
                    try:
                        self.wfile.write('--BOUNDARYSTRING\r\n')
                        self.send_header('Content-type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(
                            _jpeg_streamers[port].jpgdata.getvalue()
                        )))
                        self.end_headers()
                        self.wfile.write(_jpeg_streamers[port].jpgdata.getvalue() + '\r\n')
                        last_time_served = time.time()
                    except socket.error as e:
                        return
                    except IOError as e:
                        return
                    count += 1
                time.sleep(_jpeg_streamers[port].sleeptime)


class JpegTCPServer(SocketServer.ThreadingMixIn, SocketServer.TCPServer):
    allow_reuse_address = True
    daemon_threads =  True

# factory class for jpeg tcp server.
class JpegStreamer(object):
    """
    Allow user to stream a jpeg encoded file to a HTTP port. Any
    updates to the jpeg file will automatically be pushed to the
    browser via multipart/replace content type.
    initialization:
    js = JpegStreamer()

    update:
    img.save(js)

    open a browser and display:
    import webbrowser
    webbrowser.open(js.url)

    Note 3 optional parameters on the constructor:
    - port (default 8080) which sets the TCP port you need to connect to
    - sleep time (default 0.1) how often to update.  Above 1 second seems
      to cause dropped connections in Google chrome Once initialized,
      the buffer and sleeptime can be modified and will function
      properly -- port will not.
    """
    server = ''
    host = ''
    port = ''
    sleep_time = ''
    frame_buffer = ''
    counter = 0
    refresh_time = 0

    def __init__(self, host_port=8080, sleeptime=0.1):
        global _jpeg_streamers

        if isinstance(host_port, int):
            self.port = host_port
            self.host = 'localhost'
        elif isinstance(host_port, str) and re.search(':', host_port):
            self.host, self.port = host_port.split(':')
            self.port = int(self.port)
        elif isinstance(host_port, tuple):
            self.host, self.port = host_port
        else:
            self.port = 8080
            self.host = 'localhost'

        self.sleep_time = sleeptime
        self.server = JpegTCPServer((self.host, self.host), JpegStreamHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        _jpeg_streamers[self.port] = self
        self.server_thread.daemon = True
        self.server_thread.start()
        self.frame_buffer = self

    def url(self):
        """
        Returns the JpegStreams Webbrowser-appropriate URL, if not provided
        in the constructor, it defaults to "http://localhost:8080"
        :return: url
        """
        return 'http://' + self.host + ':' + str(self.port) + '/'

    def stream_url(self):
        """
        Returns the URL of the MJPEG stream. If host and port are not set in
         the constructor, defaults to "http://localhost:8080/stream/"
        :return: url
        """
        return self.url() + 'stream'
