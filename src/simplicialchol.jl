@doc """
Symbolic Cholesky factorization.  The fill-reducing permutation
is calculated externally, for example using Metis.NodeND
"""->
type CholSymb{T<:Union(Int32,Int64)}    # symbolic Cholesky
    pinv::Vector{T}                     # fill reducing perm for Chol
    cp::Vector{T}                       # column pointers
    parent::Vector{T}                   # elimination tree
end

@doc """
Find the nonzero pattern of Cholesky L(k,1:k-1) using etree and triu(A(:,k))
"""->
function ereach{Tv,Ti}(A::Symmetric{Tv,SparseMatrixCSC{Tv,Ti}}, k, parent)
    if A.uplo != 'U'
        error("ereach requires the upper triangle of A to be stored, i.e. A.uplo == 'U'")
    end
    k = convert(Ti,k)
    Ad = A.data
    Ai = rowvals(Ad)
    s = IntSet([k])                     # non-zeros in row k of L
    v = IntSet([k])                     # visited nodes
    for p in nzrange(Ad,k)
        i = Ai[p]                       # A(i,k) is nonzero
        if i > k
            continue                    # use only upper triangle
        end
        while !in(i,v)                  # traverse up etree
            push!(s,i)
            push!(v,i)
        end
    end
    collect(s)
end

function Metis.metisform{Tv,Ti}(S::Symmetric{Tv,SparseMatrixCSC{Tv,Ti}})
    up = S.uplo == 'U'
    Sd = S.data
    n = size(Sd,2)
    I = sizehint!(Cint[],2*nnz(Sd))
    J = sizehint!(Cint[],2*nnz(Sd))
    Si = rowvals(Sd)
    for j in 1:n, i in sub(Si,nzrange(Sd,j))
        if (up ? (i < j) : (i > j))
            push!(I,i)
            push!(J,j)
            push!(I,j)
            push!(J,i)
        end
    end
    sp = sparse(I,J,1.)
    convert(Int32,n),sp.colptr .- one(Int32),rowvals(sp) .- one(Int32)
end

function cholfact{Tv,Ti}(A::Symmetric{Tv,SparseMatrixCSC{Tv,Ti}}, S::CholSymb{Ti})
    n = size(A,2)
    cp = S.cp; pinv = S.pinv; parent = S.parent
## maybe change the natural case (i.e. no permutation) to pinv = Array(Ti, 0)
    C  = (pinv == 1:n) ? A : symperm(A, pinv)
    Cp = C.colptr; Ci = C.rowval; Cx = C.nzval
    Lp = copy(cp); Li = Array(Ti, cp[n+1]); Lx = Array(Tv, cp[n+1])
    c  = copy(cp[1:n])
    s  = Array(Ti, n)                   # Ti workspace
    x  = Array(Tv, n)                   # Tv workspace
    for k in 1:n                        # compute L[k,:] for L*L' = C
        ## Nonzero pattern of L[k,:]
        top = cs_ereach(C, k, parent, s, c) # find pattern of L(k,:)
        x[k] = 0                            # x[1:k] is now zero
        for p in Cp[k]:(Cp[k+1]-1)          # x = full(triu(C(:,k)))
            if (Ci[p] <= k) x[Ci[p]] = Cx[p] end
        end
        d = x[k]                        # d = C[k,k]
        x[k] = zero(Tv)                 # clear x for k+1st iteration
        ## Triangular solve
        while top <= n             # solve L[0:k-1,0:k-1] * x = C[:,k]
            i = s[top]             # s[top..n-1] is pattern of L[k,:]
            lki = x[i]/Lx[Lp[i]]   # L[k,i] = x[i]/L[i,i]
            x[i] = zero(Tv)        # clear x for k+1st iteration
            for p in (Lp[i]+1):(c[i]-1)
                x[Li[p]] -= Lx[p] * lki
            end
            d -= lki * lki ;            # d = d - L[k,i]*L[k,i]
            p = c[i]
            c[i] += 1
            Li[p] = k                   # store L[k,i] in column i
            Lx[p] = lki
        end
        ## Compute L[k,k]
        if (d <= 0) error("Matrix is not positive definite, detected at row $k") end
        p = c[k]
        c[k] += 1
        Li[p] = k                 # store L[k,k] = sqrt(d) in column k
        Lx[p] = sqrt(d)
    end
    SparseMatrixCSC(n, n, Lp, Li, Lx)
end
