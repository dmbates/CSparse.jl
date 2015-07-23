@doc """
Symbolic Cholesky factorization.  The fill-reducing permutation
is calculated externally, for example using Metis.NodeND
"""->
type CholSymb{T<:Integer}    # symbolic Cholesky
    pinv::Vector{T}          # fill reducing perm for Chol
    cp::Vector{T}            # column pointers
    parent::Vector{T}        # elimination tree
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
    for i in sub(Ai,nzrange(Ad,k))
        if i > k
            continue                    # use only upper triangle
        end
        while !in(i,v)                  # traverse up etree
            push!(s,i)
            push!(v,i)
            i = parent[i]
        end
    end
    collect(s)
end

@doc """
Determine the pattern of the sparse Cholesky factor of C

Returns the elimination tree and a lower triangular sparse matrix filled with ones.
"""->
function cholpattern{Tv,Ti}(C::Symmetric{Tv,SparseMatrixCSC{Tv,Ti}})
    C.uplo == 'U' || error("cholpattern requires the upper triangle of C to be stored")
    n = size(C,2)
    parent = etree(C)
    Cd = C.data
    Ci = rowvals(Cd)
    cp = ones(Ti,n+1)
    rv = Ti[]
    s = sizehint!(IntSet([]),n) # nonzeros on column j of L' (row j of L)
    v = sizehint!(IntSet([]),n) # visited nodes
    for j in 1:n
        empty!(s)
        empty!(v)
        push!(s,j)
        push!(v,j)
        for i in sub(Ci,nzrange(Cd,j))
            i > j && break              # upper triangle only
            while !in(i,v)
                push!(s,i)
                push!(v,i)
                i = parent[i]
            end
        end
        append!(rv,collect(s))
        cp[j+1] += convert(Ti,length(rv))
    end
    parent,LowerTriangular(transpose(SparseMatrixCSC(n,n,cp,rv,ones(cp[n+1]))))
end

## write a function that copies the lower triangle of the symmetric matrix to L

function LLcholfact{Tv,Ti}(C::Symmetric{Tv,SparseMatrixCSC{Tv,Ti}})
    parent,L = cholpattern(C)
    n = size(C,2)
    Ct = ltri!(transpose(C.data)) # for now evaluate the lower triangle
    Cti = rowvals(Ct)
    Ctx = nonzeros(Ct)                 # these values will be modified
    Ld = L.data
    Li = rowvals(Ld)
    Lx = fill!(0.,nonzeros(Ld))         # initialize contents of L to zero
    for k in 1:n
        ## spread values of k'th column of A into k'th column of L
        Akr = nzrange(Ct,k)
        Lkr = nzrange(Ld,k)
        Lki = sub(Li,Lkr)
        Lkx = sub(Lx,Lkr)
        if length(Akr) == length(Lkr)
            copy!(Lkx,sub(Ax,Akr))
        else
            Aki = sub(Cti,Akr)
            Akx = sub(Ctx,Akr)
            for i in eachindex(Aki)
                Lkx[searchsortedfirst(Lki,Aki[i])] = Akx[i]
            end
        end
    end
end
                           
function cholsymb{Tv,Ti}(S::Symmetric{Tv,SparseMatrixCSC{Tv,Ti}})
    _,pinv = Metis.nodeND(S)
    C = symperm(S,pinv)
    parent,post = etree(C,true)
    cp = colcounts(C,parent,post)
    CholSymb(pinv,convert(Vector{Ti},cp),parent)
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
