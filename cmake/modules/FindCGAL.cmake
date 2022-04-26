## ---------------------------------------------------------------------
##
## Copyright (C) 2017 by the deal.II authors
##
## This file is part of the deal.II library.
##
## The deal.II library is free software; you can use it, redistribute
## it, and/or modify it under the terms of the GNU Lesser General
## Public License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
## The full text of the license can be found in the file LICENSE.md at
## the top level directory of deal.II.
##
## ---------------------------------------------------------------------

#
# Try to find the CGAL libraries
#
# This module exports
#
#   CGAL_INCLUDE_DIRS
#

SET(CGAL_DIR "" CACHE PATH "An optional hint to a CGAL installation")
SET_IF_EMPTY(CGAL_DIR "$ENV{CGAL_DIR}")

DEAL_II_FIND_PATH(CGAL_INC CGAL/config.h
  HINTS ${CGAL_DIR}
  PATH_SUFFIXES include
  )

DEAL_II_PACKAGE_HANDLE(CGAL
  INCLUDE_DIRS REQUIRED CGAL_INC
  USER_INCLUDE_DIRS REQUIRED CGAL_INC
  CLEAR CGAL_INC
  )
