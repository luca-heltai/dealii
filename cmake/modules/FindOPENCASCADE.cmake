## ---------------------------------------------------------------------
## $Id$
##
## Copyright (C) 2012 - 2014 by the deal.II authors
##
## This file is part of the deal.II library.
##
## The deal.II library is free software; you can use it, redistribute
## it, and/or modify it under the terms of the GNU Lesser General
## Public License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
## The full text of the license can be found in the file LICENSE at
## the top level of the deal.II distribution.
##
## ---------------------------------------------------------------------


#
# Try to find the OpenCASCADE (OCC) library. This scripts supports the
# OpenCASCADE Community Edition (OCE) library, which is a cmake based
# OCC library. You might try the original OpenCASCADE library, but your
# mileage may vary.
#
# This module exports:
#
#   OPENCASCADE_DIR
#   OPENCASCADE_INCLUDE_DIRS
#   OPENCASCADE_LIBRARIES
#   OPENCASCADE_VERSION
#


SET(OPENCASCADE_DIR "" CACHE PATH "An optional hint to a OpenCASCADE installation")
SET_IF_EMPTY(OPENCASCADE_DIR "$ENV{OPENCASCADE_DIR}")
SET_IF_EMPTY(OPENCASCADE_DIR "$ENV{OCC_DIR}")
SET_IF_EMPTY(OPENCASCADE_DIR "$ENV{OCE_DIR}")


DEAL_II_FIND_PATH(OPENCASCADE_INCLUDE_DIR Standard_Version.hxx
  HINTS ${OPENCASCADE_DIR}
  PATH_SUFFIXES include include/oce inc
  )

IF(EXISTS ${OPENCASCADE_INCLUDE_DIR}/Standard_Version.hxx)
  FILE(STRINGS "${OPENCASCADE_INCLUDE_DIR}/Standard_Version.hxx" OPENCASCADE_VERSION
    REGEX "#define OCC_VERSION _T"
    )
ENDIF()

# These seem to be pretty much the only required ones.
SET(OPENCASCADE_LIBRARIES  
    TKFillet
    TKMesh
    TKernel
    TKG2d
    TKG3d
    TKMath
    TKIGES
    TKSTL
    TKShHealing
    TKXSBase
    TKBool
    TKBO
    TKBRep
    TKTopAlgo
    TKGeomAlgo
    TKGeomBase
    TKOffset
    TKPrim
    TKSTEP
    TKSTEPBase
    TKSTEPAttr
    TKHLR
    TKFeat
  )

SET(_libraries "")
FOREACH(_library ${OPENCASCADE_LIBRARIES})
  LIST(APPEND _libraries ${_library})
  DEAL_II_FIND_LIBRARY(${_library}
    NAMES ${_library}
    HINTS ${OPENCASCADE_DIR}
    PATH_SUFFIXES lib${LIB_SUFFIX} lib64 lib
    NO_DEFAULT_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_CMAKE_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH
    )
ENDFOREACH()


DEAL_II_PACKAGE_HANDLE(OPENCASCADE
  LIBRARIES
    REQUIRED ${_libraries}
  INCLUDE_DIRS
    REQUIRED OPENCASCADE_INCLUDE_DIR
  USER_INCLUDE_DIRS
    REQUIRED OPENCASCADE_INCLUDE_DIR
  CLEAR
    OPENCASCADE_LIBRARIES ${_libraries}
    TRILINOS_SUPPORTS_CPP11 TRILINOS_HAS_C99_TR1_WORKAROUND
  )
