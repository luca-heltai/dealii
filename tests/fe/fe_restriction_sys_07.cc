// ---------------------------------------------------------------------
//
// Copyright (C) 2007 - 2014 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------


#include "../tests.h"
#include "fe_restriction_common.h"



int
main()
{
  initlog();
  deallog.threshold_double(1.e-10);

  CHECK_SYS2 (FE_DGQArbitraryNodes<2>(QIterated<1>(QTrapez<1>(),3)), 1,
              FE_Nedelec<2>(0), 2,
              2);
}
