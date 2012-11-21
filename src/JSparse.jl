require("sparse")

## solve Lx=b where x and b are dense.  x=b on input, solution on output.
function lsolve!{T}(L::SparseMatrixCSC{T}, x::StridedVector{T})
    m,n = size(L)
    if m != n error("Matrix L is $m by $n and should be square") end
    if length(x) != n error("Dimension mismatch") end
    Lp  = L.colptr
    Li  = L.rowval
    Lx  = L.nzval
    for j in 1:n
        x[j] /= Lx[Lp[j]]
        for p in Lp[j]:(Lp[j + 1] - 1)
            x[Li[p]] -= Lx[p] * x[j]
        end
    end
    x
end

lsolve{T}(L::SparseMatrixCSC{T}, x::StridedVector{T}) = lsolve!(L, copy(x))

## solve L'x=b where x and b are dense.  x=b on input, solution on output.
function ltsolve!{T}(L::SparseMatrixCSC{T}, x::StridedVector{T})
    m,n = size(L)
    if m != n error("Matrix L is $m by $n and should be square") end
    if length(x) != n error("Dimension mismatch") end
    Lp  = L.colptr
    Li  = L.rowval
    Lx  = L.nzval
    for j in 1:n
        for p in Lp[j]:(Lp[j + 1] - 1)
            x[j] -= Lx[p] * x[Li[p]]
        end
        x[j] /= Lx[Lp[j]]
    end
    x
end

ltsolve{T}(L::SparseMatrixCSC{T}, x::StridedVector{T}) = ltsolve!(L, copy(x))

## 1-norm of a sparse matrix = max (sum (abs (A))), largest column sum
function norm(A::SparseMatrixCSC)
    Ap  = A.colptr
    Ax  = A.nzval
    nrm = zero(eltype(A))
    for j in 1:size(A,2)
        nrm = max(nrm, sum(abs(Ax[Ap[j]:(Ap[j+1]-1)])))
    end
    nrm
end

## solve Ux=b where x and b are dense.  x=b on input, solution on output.
function usolve!{T}(U::SparseMatrixCSC{T}, x::StridedVector{T})
    m,n = size(U)
    if m != n error("Matrix U is $m by $n and should be square") end
    if length(x) != n error("Dimension mismatch") end
    Up  = U.colptr
    Ui  = U.rowval
    Ux  = U.nzval
    for j in n:-1:1
        x[j] /= Ux[Up[j+1]-1]
        for p in Up[j]:(Up[j+1]-1)
            x[Ui[p]] -= Ux[p] * x[j]
        end
    end
    x
end

usolve{T}(U::SparseMatrixCSC{T}, x::StridedVector{T}) = usolve!(U, copy(x))

## solve U'x=b where x and b are dense.  x=b on input, solution on output.
function utsolve!{T}(U::SparseMatrixCSC{T}, x::StridedVector{T})
    m,n = size(U)
    if m != n error("Matrix U is $m by $n and should be square") end
    if length(x) != n error("Dimension mismatch") end
    Up  = U.colptr
    Ui  = U.rowval
    Ux  = U.nzval
    for j in 1:n
        for p in Up[j]:(Up[j+1]-1)
            x[j] -= Ux[p] * x[Ui[p]]
            end
        x[j] /= Ux[Up[j+1]-1]
    end
    x
end

