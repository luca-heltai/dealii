//----------------------------  sparse_mic.templates.h  ---------------------------
//    Copyright (C) 1998, 1999, 2000, 2001, 2002
//    by the deal.II authors and Stephen "Cheffo" Kolaroff
//
//    This file is subject to QPL and may not be  distributed
//    without copyright and license information. Please refer
//    to the file deal.II/doc/license.html for the  text  and
//    further information on this license.
//
//----------------------------  sparse_mic.templates.h  ---------------------------
#ifndef __deal2__sparse_mic_templates_h
#define __deal2__sparse_mic_templates_h


#include <base/memory_consumption.h>
#include <lac/sparse_mic.h>
#include <lac/vector.h>


template <typename number>
SparseMIC<number>::SparseMIC ()
                :
                diag(0),
                inv_diag(0),
                inner_sums(0)
{};



template <typename number>
SparseMIC<number>::SparseMIC (const SparsityPattern &sparsity)
                :
                SparseLUDecomposition<number> (sparsity),
                diag(0),
                inv_diag(0),
                inner_sums(0)
{};



template <typename number>
void
SparseMIC<number>::reinit ()
{
  if (true)
    {
      std::vector<number> tmp;
      tmp.swap (diag);
    };
  if (true)
    {
      std::vector<number> tmp;
      tmp.swap (inv_diag);
    };
  if (true)
    {
      std::vector<number> tmp;
      tmp.swap (inner_sums);
    };
  
  SparseLUDecomposition<number>::reinit ();
}



template <typename number>
void SparseMIC<number>::reinit (const SparsityPattern& sparsity)
{
  if (true)
    {
      std::vector<number> tmp;
      tmp.swap (diag);
    };
  if (true)
    {
      std::vector<number> tmp;
      tmp.swap (inv_diag);
    };
  if (true)
    {
      std::vector<number> tmp;
      tmp.swap (inner_sums);
    };
  SparseLUDecomposition<number>::reinit (sparsity);
}



template <typename number>
template <typename somenumber>
void SparseMIC<number>::decompose (const SparseMatrix<somenumber> &matrix,
				   const double                    strengthen_diagonal)
{

  SparseLUDecomposition<number>::decompose(matrix, strengthen_diagonal);

  Assert (matrix.m()==matrix.n(), ExcMatrixNotSquare ());
  Assert (m()==n(),               ExcMatrixNotSquare ());
  Assert (matrix.m()==m(),        ExcSizeMismatch(matrix.m(), m()));

  Assert (strengthen_diagonal>=0, ExcInvalidStrengthening (strengthen_diagonal));

  if (strengthen_diagonal > 0)
    strengthen_diagonal_impl ();

                                   // MIC implementation: (S. Margenov lectures)
                                   // x[i] = a[i][i] - sum(k=1, i-1,
                                   //              a[i][k]/x[k]*sum(j=k+1, N, a[k][j]))
	
                                   // TODO: for sake of siplicity,
                                   // those are placed here A better
                                   // implementation would store this
                                   // values in the underlying sparse
                                   // matrix itself.
  diag.resize (m());
  inv_diag.resize (m());
  inner_sums.resize (m());

                                   // precalc sum(j=k+1, N, a[k][j]))
  for(unsigned int row=0; row<m(); row++) {
    inner_sums[row] = get_rowsum(row);
  }

  const unsigned int* const col_nums = get_sparsity_pattern().get_column_numbers();
  const unsigned int* const rowstarts = get_sparsity_pattern().get_rowstart_indices();

  for(unsigned int row=0; row<m(); row++) {
    number temp = diag_element(row);
    number temp1 = 0;
    const unsigned int * const first_after_diagonal = prebuilt_lower_bound[row];

    unsigned int k = 0;
    for (const unsigned int * col=&col_nums[rowstarts[row]+1];
         col<first_after_diagonal; ++col, k++)
      temp1 += matrix.global_entry (col-col_nums)/diag[k]*inner_sums[k];

    diag[row] = temp - temp1;
    inv_diag[row] = 1.0/diag[row];
    Assert(diag[row]>0, ExcInternal());
  }
};



template <typename number>
inline number
SparseMIC<number>::get_rowsum (const unsigned int row) const
{
  Assert(m()==n(), ExcMatrixNotSquare());
                                   // get start of this row. skip the
                                   // diagonal element
  const unsigned int * const column_numbers = get_sparsity_pattern().get_column_numbers();
  const unsigned int * const rowstart_indices = get_sparsity_pattern().get_rowstart_indices();
  const unsigned int * const rowend = &column_numbers[rowstart_indices[row+1]];

                                   // find the position where the part
                                   // right of the diagonal starts
  const unsigned int * const first_after_diagonal = prebuilt_lower_bound[row];
  number rowsum =  0;
  for (const unsigned int * col=first_after_diagonal; col!=rowend; ++col)
    rowsum += global_entry (col-column_numbers);

  return rowsum;	
};



template <typename number>
template <typename somenumber>
void
SparseMIC<number>::vmult (Vector<somenumber>       &dst,
                          const Vector<somenumber> &src) const
{
  SparseLUDecomposition<number>::vmult (dst, src);
  Assert (dst.size() == src.size(), ExcSizeMismatch(dst.size(), src.size()));
  Assert (dst.size() == m(), ExcSizeMismatch(dst.size(), m()));

  const unsigned int N=dst.size();
  const unsigned int * const rowstart_indices = get_sparsity_pattern().get_rowstart_indices();
  const unsigned int * const column_numbers   = get_sparsity_pattern().get_column_numbers();
                                   // We assume the underlying matrix A is:
                                   // A = X - L - U, where -L and -U are
                                   // strictly lower- and upper- diagonal
                                   // parts of the system.
                                   // 
                                   // Solve (X-L)X{-1}(X-U) x = b
                                   // in 3 steps:
  dst = src;
  for (unsigned int row=0; row<N; ++row)
    {
                                       // Now: (X-L)u = b

                                       // get start of this row. skip
                                       // the diagonal element
      const unsigned int * const rowstart = &column_numbers[rowstart_indices[row]+1];
      const unsigned int * const fad = prebuilt_lower_bound[row];
      for (const unsigned int * col=rowstart; col!=fad; ++col)
        dst(row) -= global_entry (col-column_numbers) * dst(*col);
      
      dst(row) *= inv_diag[row];
    };

                                   // Now: v = Xu
  for(unsigned int row=0; row<N; row++)
    dst(row) *= diag[row];

                                   // x = (X-U)v
  for (int row=N-1; row>=0; --row)
    {
				       // get end of this row
      const unsigned int * const rowend = &column_numbers[rowstart_indices[row+1]];
      const  unsigned int * const fad = prebuilt_lower_bound[row];
      for (const unsigned int * col=fad; col!=rowend; ++col)
        dst(row) -= global_entry (col-column_numbers) * dst(*col);

      dst(row) *= inv_diag[row];
    };
};



template <typename number>
unsigned int
SparseMIC<number>::memory_consumption () const
{
  return (SparseLUDecomposition<number>::memory_consumption () +
          MemoryConsumption::memory_consumption(diag) +
          MemoryConsumption::memory_consumption(inv_diag) +
          MemoryConsumption::memory_consumption(inner_sums));
};



#endif // __deal2__sparse_mic_templates_h
