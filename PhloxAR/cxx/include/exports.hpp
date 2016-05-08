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

#ifndef PHLOXAR_EXPORTS_HPP
#define PHLOXAR_EXPORTS_HPP

#if !defined _CRT_SECURE_NO_DEPRECATE && _MSC_VER > 1300
#define _CRT_SECURE_NO_DEPRECATE
#endif


#if (defined WIN32 || defined _WIN32 || defined WINCE)   && defined DSO_EXPORTS
#define PHLOXAR_EXPORTS __declspec(dllexport)
#else
#define PHLOXAR_EXPORTS
#endif

#endif //PHLOXAR_EXPORTS_HPP
