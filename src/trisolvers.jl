## solve Lx=b where x and b are dense.  x=b on input, solution on output.
function Base.A_ldiv_B!{T}(L::LowerTriangular{T,SparseMatrixCSC{T}}, x::StridedVector{T})
    Ld = L.data
    n = Base.LinAlg.chksquare(Ld)
    length(x) == n || throw(DimensionMismatch())
    Lp = Ld.colptr; Li = rowvals(Ld); Lx = nonzeros(Ld)
    for j in 1:n
        x[j] /= Lx[Lp[j]]
        for p in (Lp[j] + 1):(Lp[j+1] - 1)
            x[Li[p]] -= Lx[p] * x[j]
        end
    end
    x
end

## solve L'x=b where x and b are dense.  x=b on input, solution on output.
function Ac_ldiv_B!{T}(L::LowerTriangular{T,SparseMatrixCSC{T}}, x::StridedVector{T})
    Ld = L.data
    n = Base.LinAlg.chksquare(Ld)
    length(x) == n || throw(DimensionMismatch())
    Lp = Ld.colptr; Li = rowvals(Ld); Lx = nonzeros(Ld)
    for j in reverse(1:n)
        for p in (Lp[j] + 1):(Lp[j+1] - 1)
            x[j] -= Lx[p] * x[Li[p]]
        end
        x[j] /= Lx[Lp[j]]
    end
    x
end

## solve Ux=b where x and b are dense.  x=b on input, solution on output.
function A_ldiv_B!{T}(U::UpperTriangular{T,SparseMatrixCSC{T}}, x::StridedVector{T})
    n = Base.LinAlg.chksquare(U)
    length(x) == n || throw(DimensionMismatch())
    Ud = U.data
    Up = Ud.colptr; Ui = rowvals(Ud); Ux = nonzeros(Ud)
    for j in reverse(1:n)
        x[j] /= Ux[Up[j+1]-1]
        for p in Up[j]:(Up[j+1]-2)
            x[Ui[p]] -= Ux[p] * x[j]
        end
    end
    x
end

## solve U'x=b where x and b are dense.  x=b on input, solution on output.
function Ac_ldiv_B!{T}(U::UpperTriangular{T,SparseMatrixCSC{T}}, x::StridedVector{T})
    n = Base.LinAlg.chksquare(U)
    length(x) == n || throw(DimensionMismatch)
    Ud = U.data
    Up = Ud.colptr; Ui = rowvals(Ud); Ux = nonzeros(Ud)
    for j in 1:n
        for p in Up[j]:(Up[j+1]-2)
            x[j] -= Ux[p] * x[Ui[p]]
            end
        x[j] /= Ux[Up[j+1]-1]
    end
    x
end

## 1-norm of a sparse matrix = max (sum (abs (A))), largest column sum
function js_norm(A::SparseMatrixCSC)
    Ax  = nonzeros(A)
    nrm = zero(eltype(A))
    for j in 1:size(A,2)
        nrm = max(nrm, sumabs(sub(Ax,nzrange(A,j))))
    end
    nrm
end

## solve Gx=b(:,k), where G is either upper (lo=0) or lower (lo=1) triangular
function js_spsolve{Tv,Ti<:Integer}(G::LowerTriangular{Tv,SparseMatrixCSC{Tv}},
                                   B::SparseMatrixCSC{Tv}, k,
                                   xi::Vector{Ti}, x::Vector{Tv}, pinv::Vector)
    n = Base.LinAlg.chksquare(G)
    Gd = G.data
    Gp = Gd.colptr; Gi = rowvals(Gd); Gx = nonzeros(Gd)
    Bp = B.colptr; Bi = rowvals(B); Bx = nonzeros(B)
    ## must have reach return a 1-based index
    top = js_reach(G, B, k, xi, pinv)   # xi[top..n-1]=Reach(B(:,k))
    ## get xi from reach?
    for p in top:n x[xi[p]] = 0. end    # clear x
    for p in Bp[k]:(Bp[k+1]-1) x[Bi[p]] = Bx[p] end # scatter B
    for px in top:n
        j = xi[px]                       # x[j] is nonzero
        J = pinv[j]                      # j maps to col J of G
        if J < 0 continue end            # column J is empty
        x[j] /= Gx[lo? Gp[J]: Gp[J+1]-1] # x(j) /= G[j,j]
        p = lo ? (Gp[J]+1) : Gp[J]       # lo: L[j,j] 1st entry
        q = lo ? Gp[J+1] : (Gp[J+1]-1)   # up: U[j,j] last entry
        for pp in p:q
            x[Gi[pp]] -= Gx[pp] * x[j]   # x[i] -= G[i,j] * x[j]
        end
    end
    top
end
