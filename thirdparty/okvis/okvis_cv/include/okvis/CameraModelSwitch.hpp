#ifndef CAMERA_MODEL_SWITCH_HPP
#define CAMERA_MODEL_SWITCH_HPP

#include <okvis/ModelSwitch.hpp>

#ifndef DISTORTION_MODEL_NO_NODISTORTION_SWITCH_CASES
#define DISTORTION_MODEL_NO_NODISTORTION_SWITCH_CASES                          \
  case okvis::cameras::NCameraSystem::Equidistant:                             \
    DISTORTION_MODEL_CASE(                                                     \
        okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion>)  \
    break;                                                                     \
  case okvis::cameras::NCameraSystem::RadialTangential:                        \
    DISTORTION_MODEL_CASE(okvis::cameras::PinholeCamera<                       \
                          okvis::cameras::RadialTangentialDistortion>)         \
    break;                                                                     \
  case okvis::cameras::NCameraSystem::RadialTangential8:                       \
    DISTORTION_MODEL_CASE(okvis::cameras::PinholeCamera<                       \
                          okvis::cameras::RadialTangentialDistortion8>)        \
    break;                                                                     \
  case okvis::cameras::NCameraSystem::FOV:                                     \
    DISTORTION_MODEL_CASE(                                                     \
        okvis::cameras::PinholeCamera<okvis::cameras::FovDistortion>)          \
    break;                                                                     \
  case okvis::cameras::NCameraSystem::EUCM:                                    \
    DISTORTION_MODEL_CASE(okvis::cameras::EUCM)                                \
    break;                                                                     \
  default:                                                                     \
    MODEL_DOES_NOT_EXIST_EXCEPTION                                             \
    break;
#endif

#ifndef DISTORTION_MODEL_SWITCH_CASES
#define DISTORTION_MODEL_SWITCH_CASES                                          \
  case okvis::cameras::NCameraSystem::Equidistant:                             \
    DISTORTION_MODEL_CASE(                                                     \
        okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion>)  \
    break;                                                                     \
  case okvis::cameras::NCameraSystem::RadialTangential:                        \
    DISTORTION_MODEL_CASE(okvis::cameras::PinholeCamera<                       \
                          okvis::cameras::RadialTangentialDistortion>)         \
    break;                                                                     \
  case okvis::cameras::NCameraSystem::RadialTangential8:                       \
    DISTORTION_MODEL_CASE(okvis::cameras::PinholeCamera<                       \
                          okvis::cameras::RadialTangentialDistortion8>)        \
    break;                                                                     \
  case okvis::cameras::NCameraSystem::NoDistortion:                            \
    DISTORTION_MODEL_CASE(                                                     \
        okvis::cameras::PinholeCamera<okvis::cameras::NoDistortion>)           \
    break;                                                                     \
  case okvis::cameras::NCameraSystem::FOV:                                     \
    DISTORTION_MODEL_CASE(                                                     \
        okvis::cameras::PinholeCamera<okvis::cameras::FovDistortion>)          \
    break;                                                                     \
  case okvis::cameras::NCameraSystem::EUCM:                                    \
    DISTORTION_MODEL_CASE(okvis::cameras::EUCM)                                \
    break;                                                                     \
  default:                                                                     \
    MODEL_DOES_NOT_EXIST_EXCEPTION                                             \
    break;
#endif

#endif // CAMERA_MODEL_SWITCH_HPP
