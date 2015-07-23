module CSparse

    using Metis

    export testmat

    include("utilities.jl")
    include("testmat.jl")
    include("simplicialchol.jl")
    include("trisolvers.jl")


## Replace calls to these simple functions by negation, abs, etc.
JS_FLIP(i::Integer) = -i    # CSparse must flip about -1 because 0 is a valid index
JS_UNFLIP(i::Integer) = (i <= 0) ? -i : i        # or abs(i)
JS_MARKED(w::Vector{Integer},j::Integer) = w[j] <= 0
JS_MARK(w::Vector{Integer},j::Integer) = w[j] = -w[j]

function js_chol{Tv,Ti}(A::SparseMatrixCSC{Tv,Ti}, S::CholSymb{Ti})
    n = size(A,2)
    cp = S.cp; pinv = S.pinv; parent = S.parent
## maybe change the natural case (i.e. no permutation) to pinv = Array(Ti, 0)
    C  = (pinv == 1:n) ? A : js_symperm(A, pinv)
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

## depth-first-search of the graph of a matrix, starting at node j
function js_dfs{T}(j::Integer, G::SparseMatrixCSC{T}, top::Integer,
                   xi::Vector{Integer}, pstack::Vector{Integer},
                   pinv::Vector{Integer})
    head = 0; Gp = G.colptr; Gi = G.rowval
    xi[0] = j                        # initialize the recursion stack
    while (head >= 0)
        j = xi[head]       # get j from the top of the recursion stack
        jnew = pinv[j]
        if !JS_MARKED(Gp, j)
            JS_MARK(Gp, j)              # mark node j as visited
            pstack[head] = (jnew < 0) ? 0 : JS_UNFLIP(Gp[jnew])
        end
        done = 1            # node j done if no unvisited neighbors
        p2 = (jnew < 0) ? 0 : JS_UNFLIP(Gp[jnew+1])
        for p in pstack[head]:(p2-1)    # examine all neighbors of j
            i = Gi[p]                   # consider neighbor node i
            if (JS_MARKED(Gp, i)) continue end # skip visited node i
            pstack[head] = p      # pause depth-first search of node j
            xi[++head] = i        # start dfs at node i
            done = 0              # node j is not done
            break                 # break, to start dfs[i]
        end
        if done                 # depth-first search at node j is done
            head -= 1           # remove j from the recursion stack
            top -= 1
            xi[top] = j                # and place in the output stack
        end
    end
    top
end

# compute the etree of A (using triu(A), or A'A without forming A'A
# A root of the etree (which may be a forest) is indicated by 0
function js_etree{Tv,Ti}(A::SparseMatrixCSC{Tv,Ti}, ata::Bool)
    m,n = size(A); Ap = A.colptr; Ai = A.rowval
    parent = zeros(Ti, n); w = zeros(Ti, n + (ata ? m : 0))
    ancestor = 0; prev = n              # offsets into w
    if (ata) w[prev + (1:m)] = 0 end
    for k in 1:n
        parent[k] = 0                   # node k has no parent yet
        w[ancestor + k] = 0             # nor does k have an ancestor
        for p in Ap[k]:(Ap[k+1] - 1)
            i = ata ? w[Ai[p] + prev] : Ai[p]
            while i != 0 && i < k
                inext = w[ancestor + i] # inext = ancestor of i
                w[ancestor + i] = k     # path compression
                if (inext == 0) parent[i] = k end # no anc., parent is k
                i = inext
            end
            if (ata) w[Ai[p] + prev] = k end
        end
    end
    parent
end

js_etree(A::SparseMatrixCSC) = js_etree(A, false)


## post order a forest
function js_post{T<:Union(Int32,Int64)}(parent::Vector{T})
    n = length(parent)
    head = zeros(T,n)                   # empty linked lists
    next = zeros(T,n)
    for j in n:-1:1                  # traverse nodes in reverse order
        if (parent[j] == 0) continue end # j is a root
        next[j] = head[parent[j]]      # add j to list of its parent
        head[parent[j]] = j
    end
    return head, next
    stack = zeros(T,n)
    post = zeros(T,n)
    k = 1
    for j in 1:n
        if (parent[j] != 0) continue end # skip j if it is not a root
        k = js_tdfs(j, k, head, next, post, stack) ;
    end
    post
end

## xi[top:n] = nodes reachable from graph of G*P' via nodes in B[:,k]
## xi[n:2n] used as workspace
function js_reach{Tv,Ti}(G::SparseMatrixCSC{Tv,Ti}, B::SparseMatrixCSC{Tv,Ti},
                         k::Integer, xi::Vector{Ti}, pinv::Vector{Ti})
    n = size(G,2); Bp = B.colptr; Bi = B.rowval; Gp = G.colptr
    top = n
    for p in Bp[k]:(Bp[k+1]-1)
        if !JS_MARKED(Gp, Bi[p])      # start a dfs at unmarked node i
            top = js_dfs(Bi[p], G, top, xi, xi+n, pinv)
        end
    end
    for p in top:n JS_MARK(Gp, xi[p]) end  #restore G
    top
end


# depth-first search and postorder of a tree rooted at node j
function js_tdfs{T<:Union(Int32,Int64)}(j::Integer, k::Integer, head::Vector{T},
                 next::Vector{T}, post::Vector{T}, stack::Vector{T})
    top = 1
    stack[1] = j                        # place j on the stack
    while (top > 0)                     # while (stack is not empty)
        p = stack[top]                  # p = top of stack
        i = head[p]                     # i = youngest child of p
        if (i == 0)
            top -= 1              # p has no unordered children left
            post[k] = p           # node p is the kth postordered node
            p += 1
        else
            head[p] = next[i]           # remove i from children of p
            top += 1
            stack[top] = i              # start dfs on child node i
        end
    end
    k
end

## This is essentially a copy of the transpose function in sparse.jl
function js_transpose{Tv,Ti}(A::SparseMatrixCSC{Tv,Ti})
    m,n = size(A)
    Ap = A.colptr; Ai = A.rowval; Ci = similar(Ai); Ax = A.nzval; Cx = similar(Ax)
    w  = zeros(Ti, m + 1)
    w[1] = one(Ti)
    for i in Ai w[i + 1] += 1 end         # row counts
    Cp = cumsum(w)
    w[:] = Cp[:]
    for j in 1:n, p in Ap[j]:(Ap[j+1] - 1)
        i = Ai[p]; q = w[i]; w[i] += 1
        Ci[q] = j
        Cx[q] = Ax[p]
    end
    SparseMatrixCSC(n, m, Cp, Ci, Cx)
end

## Copied from extras/suitesparse.jl
function _jl_convert_to_0_based_indexing!(S::SparseMatrixCSC)
    for i in 1:length(S.rowval) S.rowval[i] -= 1 end
    for p in 1:length(S.colptr) S.colptr[p] -= 1 end
    S
end

function _jl_convert_to_1_based_indexing!(S::SparseMatrixCSC)
    for i in 1:length(S.rowval) S.rowval[i] += 1 end
    for p in 1:length(S.colptr) S.colptr[p] += 1 end
    S
end

_jl_convert_to_0_based_indexing(S) = _jl_convert_to_0_based_indexing!(copy(S))
_jl_convert_to_1_based_indexing(S) = _jl_convert_to_1_based_indexing!(copy(S))

type cs{Tv<:Union(Float64,Complex128),Ti<:Union(Int32,Int64)} # the CXSparse cs struct
    nzmax::Ti
    m::Ti
    n::Ti
    p::Ptr{Ti}
    i::Ptr{Ti}
    x::Ptr{Tv}
    nz::Ti
end

function cs{Tv<:Union(Float64,Complex128),Ti<:Union(Int32,Int64)}(A::SparseMatrixCSC{Tv,Ti})
    if A.colptr[1] != 0 error("Sparse matrix must be in 0-based indexing") end
    cs{Tv,Ti}(convert(Ti,A.colptr[end]), convert(Ti,A.m), convert(Ti,A.n),
              pointer(A.colptr), pointer(A.rowval), pointer(A.nzval), convert(Ti, -1))
end

## Primary CXSparse-based functions, cs_cholsol, cs_lusol, cs_print, cs_qrsol
for (cholsol, lusol, prt, qrsol, vtyp, ityp) in
    ((:cs_ci_cholsol, :cs_ci_lusol, :cs_ci_print, :cs_ci_qrsol, :Complex128, :Int32),
     (:cs_di_cholsol, :cs_di_lusol, :cs_di_print, :cs_di_qrsol, :Float64, :Int32),
     (:cs_cl_cholsol, :cs_cl_lusol, :cs_cl_print, :cs_cl_qrsol, :Complex128, :Int64),
     (:cs_dl_cholsol, :cs_dl_lusol, :cs_dl_print, :cs_dl_qrsol, :Float64, :Int64))
    @eval begin
        ## A is symmetric but only triu(A) is passed
        function cs_cholsol!(A::SparseMatrixCSC{$vtyp,$ityp}, b::Vector{$vtyp}, order::Integer)
            if !(0 <= order <= 3) error("order = $order is not in the range 0 to 3") end
            m,n = size(A)
            if m != n || n != length(b) error("Dimension mismatch") end
            st = ccall(($(string(cholsol)),"libcxsparse"), $ityp, ($ityp, Ptr{Void}, Ptr{$vtyp}),
                       order, pack(cs(_jl_convert_to_0_based_indexing!(A))).data, b)
            _jl_convert_to_1_based_indexing!(A)
            if st == 0 error("Failure in cholsol") end
            b
        end

        ## A must be square
        function cs_lusol!(A::SparseMatrixCSC{$vtyp,$ityp}, b::Vector{$vtyp}, order::Integer, tol::Real)
            if !(0 <= order <= 3) error("order = $order is not in the range 0 to 3") end
            m,n = size(A)
            if m != n || n != length(b) error("Dimension mismatch") end
            st = ccall(($(string(lusol)),"libcxsparse"), $ityp, ($ityp, Ptr{Void}, Ptr{$vtyp}, Float64),
                       order, pack(cs(_jl_convert_to_0_based_indexing!(A))).data, b, tol)
            _jl_convert_to_1_based_indexing!(A)
            if st == 0 error("Failure in lusol") end
            b
        end

        function cs_print(A::SparseMatrixCSC{$vtyp,$ityp}, brief::Bool)
            ccall(($(string(prt)),"libcxsparse"), Void, (Ptr{Void}, $ityp),
                  pack(cs(_jl_convert_to_0_based_indexing!(A))).data, brief)
            _jl_convert_to_1_based_indexing!(A)
            None
        end

        ## Overdetermined least squares system when m >= n
        function cs_qrsol!(A::SparseMatrixCSC{$vtyp,$ityp}, b::Vector{$vtyp}, order::Integer)
            if (order != 0 && order != 3) error("order = $order but must be 0 or 3") end
            if size(A, 1) != length(b) error("Dimension mismatch") end
            st = ccall(($(string(qrsol)),"libcxsparse"), $ityp, ($ityp, Ptr{Void}, Ptr{$vtyp}),
                       order, pack(cs(_jl_convert_to_0_based_indexing!(A))).data, b)
            _jl_convert_to_1_based_indexing!(A)
            if st == 0 error("Failure in qrsol") end
            b
        end
    end
end

## default ordering for cs_cholsol is 1 (i.e. amd(A))
cs_cholsol!{T<:Union(Float64,Complex128)}(A::SparseMatrixCSC{T}, b::Vector{T}) = cs_cholsol!(A, b, 1)
function cs_cholsol{T<:Union(Float64,Complex128)}(A::SparseMatrixCSC{T}, b::Vector{T}, ord::Integer)
    cs_cholsol!(A, copy(b), ord)
end
function cs_cholsol{T<:Union(Float64,Complex128)}(A::SparseMatrixCSC{T}, b::Vector{T})
    cs_cholsol(A, copy(b), 1)
end

## default ordering for cs_lusol is 2.
## There are other cases with order=1 but I haven't yet deciphered them
cs_lusol!{T<:Union(Float64,Complex128)}(A::SparseMatrixCSC{T}, b::Vector{T}) = cs_lusol!(A, b, 2, 1.)
function cs_lusol{T<:Union(Float64,Complex128)}(A::SparseMatrixCSC{T}, b::Vector{T}, ord::Integer)
    cs_lusol!(A, copy(b), ord, ord == 1 ? 0.001 : 1.)
end
function cs_lusol{T<:Union(Float64,Complex128)}(A::SparseMatrixCSC{T}, b::Vector{T})
    cs_lusol!(A, copy(b), 2, 1.)
end

cs_qrsol!{T<:Union(Float64,Complex128)}(A::SparseMatrixCSC{T}, b::Vector{T}) = cs_lusol!(A, b, 3)
function cs_qrsol{T<:Union(Float64,Complex128)}(A::SparseMatrixCSC{T}, b::Vector{T}, ord::Integer)
    cs_qrsol!(A, copy(b), ord)
end
function cs_qrsol{T<:Union(Float64,Complex128)}(A::SparseMatrixCSC{T}, b::Vector{T})
    cs_qrsol(A, copy(b), 3)
end

for (amd, etree, post, counts, norm, vtyp, ityp) in
    ((:cs_ci_amd, :cs_ci_etree, :cs_ci_post, :cs_ci_counts, :cs_ci_norm, :Complex128, :Int32),
     (:cs_di_amd, :cs_di_etree, :cs_di_post, :cs_di_counts, :cs_di_norm, :Float64, :Int32),
     (:cs_cl_amd, :cs_cl_etree, :cs_cl_post, :cs_cl_counts, :cs_cl_norm, :Complex128, :Int64),
     (:cs_dl_amd, :cs_dl_etree, :cs_dl_post, :cs_dl_counts, :cs_dl_norm, :Float64, :Int64))
    @eval begin
        ## Approximate minimal degree ordering
        function cs_amd(A::SparseMatrixCSC{$vtyp,$ityp}, order::Integer)
            if !(0 < order < 4) error("Valid values of order are 1:Chol, 2:LU, 3:QR") end
            ppt   = ccall(($(string(amd)),"libcxsparse"), Ptr{$ityp}, ($ityp, Ptr{Void}),
                          order, pack(cs(_jl_convert_to_0_based_indexing!(A))).data)
            _jl_convert_to_1_based_indexing!(A)
            pointer_to_array(ppt, (size(A,2),)) + 1
        end

        ## cs_counts returns all the information from cs_etree plus the column counts
        function cs_counts(A::SparseMatrixCSC{$vtyp,$ityp}, col::Bool)
            n = size(A, 2)
            cspk = pack(cs(_jl_convert_to_0_based_indexing!(A)))
            etrpt = ccall(($(string(etree)),"libcxsparse"), Ptr{$ityp},
                          (Ptr{Void}, $ityp), cspk.data, col)
            pospt = ccall(($(string(post)),"libcxsparse"), Ptr{$ityp},
                          (Ptr{$ityp}, $ityp), etrpt, n)
            coupt = ccall(($(string(counts)),"libcxsparse"), Ptr{$ityp},
                          (Ptr{Void}, Ptr{$ityp}, Ptr{$ityp}, $ityp),
                          cspk.data, etrpt, pospt, col)
            _jl_convert_to_1_based_indexing!(A)
            (pointer_to_array(etrpt, (n,)) + 1, pointer_to_array(pospt, (n,)) + 1,
             pointer_to_array(coupt, (n,)))
        end
        
        ## returns the elimination tree and the post-ordering permutation
        function cs_etree(A::SparseMatrixCSC{$vtyp,$ityp}, col::Bool)
            ept = ccall(($(string(etree)),"libcxsparse"), Ptr{$ityp}, (Ptr{Void}, $ityp),
                        pack(cs(_jl_convert_to_0_based_indexing!(A))).data, col)
            _jl_convert_to_1_based_indexing!(A)
            n = size(A, 2)
            popt  = ccall(($(string(post)),"libcxsparse"), Ptr{$ityp},
                          (Ptr{$ityp}, $ityp), ept, n)
            pointer_to_array(ept, (n,)) + 1, pointer_to_array(popt, (n,)) + 1
        end
        
        ## 1-norm of a sparse matrix (better to use js_norm, this is just for illustration)
        function cs_norm(A::SparseMatrixCSC{$vtyp,$ityp})
            res = ccall(($(string(norm)),"libcxsparse"), Float64, (Ptr{Void},),
                        pack(cs(_jl_convert_to_0_based_indexing!(A))).data)
            _jl_convert_to_1_based_indexing!(A)
            res
        end
    end
end

cs_amd(A::SparseMatrixCSC) = cs_amd(A, 1)
cs_counts(A::SparseMatrixCSC) = cs_counts(A, false)
cs_etree(A::SparseMatrixCSC) = cs_etree(A, false)
cs_print(A::SparseMatrixCSC) = cs_print(A, true)

type cs_symb{Tv<:Union(Float64,Complex128),Ti<:Union(Int32,Int64)} # the CXSparse cs_symbolic struct
    pinv::Ptr{Ti}
    q::Ptr{Ti}
    parent::Ptr{Ti}
    cp::Ptr{Ti}
    leftmost::Ptr{Ti}
    m2::Ti
    lnz::Float64
    unz::Float64
end

type cs_num{Tv<:Union(Float64,Complex128),Ti<:Union(Int32,Int64)} # the CXSparse cs_numeric struct
    L::Ptr{cs{Tv,Ti}}
    U::Ptr{cs{Tv,Ti}}
    pinv::Ptr{Ti}
    B::Ptr{Float64}
end

for (chol, qr, schol, sqr, symperm, vtyp, ityp) in
    ((:cs_ci_chol, :cs_ci_qr, :cs_ci_schol, :cs_ci_sqr, :cs_ci_symperm, :Complex128, :Int32),
     (:cs_di_chol, :cs_di_qr, :cs_di_schol, :cs_di_sqr, :cs_di_symperm, :Float64, :Int32),
     (:cs_cl_chol, :cs_cl_qr, :cs_cl_schol, :cs_cl_sqr, :cs_cl_symperm, :Complex128, :Int64),
     (:cs_dl_chol, :cs_dl_qr, :cs_dl_schol, :cs_dl_sqr, :cs_dl_symperm, :Float64, :Int64))
    @eval begin
        ## Numeric Cholesky factorization
        function cs_chol(A::SparseMatrixCSC{$vtyp,$ityp}, S::cs_symb{$vtyp,$ityp})
            pt  = ccall(($(string(chol)),"libcxsparse"), Ptr{Uint8}, (Ptr{Void}, Ptr{Void}),
                        pack(cs(_jl_convert_to_0_based_indexing!(A))).data, pack(S).data)
            _jl_convert_to_1_based_indexing!(A)
            outtyp = cs_num{$vtyp,$ityp}
            unpack(IOString(pointer_to_array(pt, (sum(map(sizeof, outtyp.types)),))), outtyp)
        end
        ## Symbolic Cholesky factorization
        function cs_schol(A::SparseMatrixCSC{$vtyp,$ityp}, order::Integer)
            if !(0 <= order <= 3) error("order = $order is not in the range 0 to 3") end
            pt  = ccall(($(string(schol)),"libcxsparse"), Ptr{Uint8}, ($ityp, Ptr{Void}),
                        order, pack(cs(_jl_convert_to_0_based_indexing!(A))).data)
            _jl_convert_to_1_based_indexing!(A)
            outtyp = cs_symb{$vtyp,$ityp}
            unpack(IOString(pointer_to_array(pt, (sum(map(sizeof, outtyp.types)),))), outtyp)
        end
        ## Symbolic QR decomposition
        function cs_sqr(A::SparseMatrixCSC{$vtyp,$ityp}, order::Integer)
            if !(0 <= order <= 3) error("order = $order is not in the range 0 to 3") end
            pt  = ccall(($(string(sqr)),"libcxsparse"), Ptr{Uint8}, ($ityp, Ptr{Void}, $ityp),
                        order, pack(cs(_jl_convert_to_0_based_indexing!(A))).data, 1)
            _jl_convert_to_1_based_indexing!(A)
            outtyp = cs_symb{$vtyp,$ityp}
            unpack(IOString(pointer_to_array(pt, (sum(map(sizeof, outtyp.types)),))), outtyp)
        end
    end
end

cs_schol(A::SparseMatrixCSC) = cs_schol(A, 1)
cs_sqr(A::SparseMatrixCSC) = cs_sqr(A, 3)

end
