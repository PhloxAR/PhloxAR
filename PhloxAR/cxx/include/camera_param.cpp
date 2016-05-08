/**
 * Copyright 2016(c) Matthias Y. Chen
 * <matthiasychen@gmail.com/matthias_cy@outlook.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "camera_param.hpp"
#include <fstream>
#include <iostream>
#include <opencv/cv.h>

namespace phloxar {
  CameraParam::CameraParam()
  {
    camera_matrix = cv::Mat();
    distortion = cv::Mat();
    cam_size = cv::Size(-1, -1);
  }

  CameraParam::CameraParam(cv::Mat camMat, cv::Mat distCoeff, cv::Size size)
  {
    setParams(camMat, distCoeff, size);
  }

  CameraParam::CameraParam(const CameraParam &cam)
  {
    cam.camera_matrix.copyTo(camera_matrix);
    cam.distortion.copyTo(distortion);
    cam_size = cam.cam_size;
  }

  void CameraParam::setParams(cv::Mat camMat, cv::Mat distCoeff,
                              cv::Size size)
  {
    if (camMat.rows != 3 || camMat.cols != 3)
      throw cv::Exception(9000, "invalid input camera matrix",
                          "CameraParam::setParams", __FILE__, __LINE__);
    camMat.covertTo(camera_matrix, CV_32FC1);

    if (cam)
  }
}

