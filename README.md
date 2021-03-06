CSparse.jl
==========

A [Julia](http://julialang.org) implementation of some of the functions in the
[CSparse](http://www.cise.ufl.edu/research/sparse/CSparse) and
[CXSparse](http://www.cise.ufl.edu/research/sparse/CXSparse/)
libraries developed by
[Tim Davis](http://engineering.tamu.edu/cse/people/davis-tim)

The Julia functions stay very close to the C originals.  Most of the
differences are in changing 0-based indexing to 1-based indexing and
in using the Julia ```CompositeKind``` rather than a pointer to a C
```struct```.  This also allows for checking consistency of
dimensions.

For example, the C function ```cs_lsolve```
```C
#include "cs.h"
/* solve Lx=b where x and b are dense.  x=b on input, solution on output. */
csi cs_lsolve (const cs *L, double *x)
{
    csi p, j, n, *Lp, *Li ;
    double *Lx ;
    if (!CS_CSC (L) || !x) return (0) ;                     /* check inputs */
    n = L->n ; Lp = L->p ; Li = L->i ; Lx = L->x ;
    for (j = 0 ; j < n ; j++)
    {
        x [j] /= Lx [Lp [j]] ;
        for (p = Lp [j]+1 ; p < Lp [j+1] ; p++)
        {
            x [Li [p]] -= Lx [p] * x [j] ;
        }
    }
    return (1) ;
}
```
becomes
```julia
## solve Lx=b where x and b are dense.  x=b on input, solution on output.
function js_lsolve!{T}(L::SparseMatrixCSC{T}, x::StridedVector{T})
    m,n = size(L)
    if m != n error("Matrix L is $m by $n and should be square") end
    if length(x) != n error("Dimension mismatch") end
    Lp = L.colptr; Li = L.rowval; Lx = L.nzval
    for j in 1:n
        x[j] /= Lx[Lp[j]]
        for p in (Lp[j] + 1):(Lp[j+1] - 1)
            x[Li[p]] -= Lx[p] * x[j]
        end
    end
    x
end

js_lsolve{T}(L::SparseMatrixCSC{T}, x::StridedVector{T}) = js_lsolve!(L, copy(x))
```
