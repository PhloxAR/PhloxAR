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

#ifndef PHLOXAR_CAMERA_PARAM_HPP_HPP
#define PHLOXAR_CAMERA_PARAM_HPP_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include "exports.hpp"

namespace phloxar {

  class PHLOXAR_EXPORTS CameraParam {
  public:
    // 3 x 3 matrix (fx 0 cv, 0 fy cy, 0 0 1)
    cv::Mat camera_matrix;
    // 4x1 matrix (k1, k2, p1, p2)
    cv::Mat distortion;
    // size of the image
    cv::Size cam_size;

    // Empty constructor.
    CameraParam();

    // Creates the object from the info passed.
    CameraParam(cv::Mat camMat, cv::Mat distCoeff, cv::Size size) throw(cv::Exception);

    // Copy constructor
    CameraParam(const CameraParam& cam);

    // Sets the parameters
    void setParams(cv::Mat camMat, cv::Mat distCoeff, cv::Size size) throw(cv::Exception);

    // Indicates whether this object is valid
    bool isValid() const {
      return camera_matrix.rows != 0 && camera_matrix.cols != 0 &&
          distortion.rows != 0 && distortion.cols != 0;
    }

    // Assign operator.
    CameraParam& operator =(const CameraParam& rhs);

    // Reads parameters from file.
    void readFromFile(std::string path) throw(cv::Exception);

    // Save parameters to file.
    void saveToFile(std::string path, bool xml=true) throw(cv::Exception);

    // Reads from a YAML file generated with OpenCV calibration utility.
    void readFromXMLFile(std::string path) throw(cv::Exception);

    // Adjust parameters to the size of the image indicated.
    void resize(cv::Size size) throw(cv::Exception);

    // Returns the location of the camera in the reference system given by the
    // rotation and translation vectors passed.
    static cv::Point3f getCameraLocation(cv::Mat rvec, cv::Mat tvec);

    void glGetProjectonMatrix(cv::Size orgImgSize, cv::Size size,
                              double proj_mat[16], double gnear, double gfar,
                              bool invert=false) throw(cv::Exception);

    void OgreGetProjectionMatrix(cv::Size orgImgSize, cv::Size size,
                                 double proj_matrix[16],
                                 double gnear, double gfar,
                                 bool invert=false) throw(cv::Exception);


  private:
    static void argConvGlcpara2(double cparam[3][4], int width,
                                int height, double gnear, double m[16],
                                bool invert) throw(cv::Exception);
    static arParamDecompMat(double source[3][4], double cpara[3][4],
                            double trans[3][4]) throw(cv::Excpetion);
    static double norm(double a, double b, double c);
    static void dot(double a1, double a2, double a3,
                    double b1, double b2, double b3);
  };

}

#endif //PHLOXAR_CAMERA_PARAM_HPP_HPP
