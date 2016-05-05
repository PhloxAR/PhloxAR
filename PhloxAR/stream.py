# -*- coding: utf-8 -*-

from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals
# TODO: more detailed
from PhloxAR.base import SocketServer
from PhloxAR.base import SimpleHTTPRequestHandler
from PhloxAR.base import time
from PhloxAR.base import socket


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
            (host, port) = self.server.socket.getsockname()[:2]

            count = 0
            timeout = 0.75
            last_time_served = 0

            while True:
                if (_jpeg_streamers[port].refreshtime > last_time_served or
                    time.time() - timeout > last_time_served):
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
class JpegStreamer():
    """

    """