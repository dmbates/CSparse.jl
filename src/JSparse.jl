require("sparse")

## Replace calls to these simple functions by negation, abs, etc.
JS_FLIP(i::Integer) = -i    # CSparse must flip about -1 because 0 is a valid index
JS_UNFLIP(i::Integer) = (i <= 0) ? -i : i        # or abs(i)
JS_MARKED(w::Vector{Integer},j::Integer) = w[j] <= 0
JS_MARK(w::Vector{Integer},j::Integer) = w[j] = -w[j]

type CholSymb{T<:Union(Int32,Int64)}    # symbolic Cholesky
    pinv::Vector{T}                     # fill reducing perm for Chol
    cp::Vector{T}                       # column pointers
    parent::Vector{T}                   # elimination tree
end

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
        top = cs_ereach (C, k, parent, s, c) # find pattern of L(k,:)
        x[k] = 0                             # x[1:k] is now zero
        for p in Cp[k]:(Cp[k+1]-1)           # x = full(triu(C(:,k)))
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
        j = xi [head]     # get j from the top of the recursion stack
        jnew = pinv[j]
        if !JS_MARKED(Gp, j)
            JS_MARK(Gp, j)              # mark node j as visited
            pstack[head] = (jnew < 0) ? 0 : JS_UNFLIP(Gp[jnew])
        end
        done = 1            # node j done if no unvisited neighbors
        p2 = (jnew < 0) ? 0 : JS_UNFLIP(Gp[jnew+1])
        for p in pstack[head]:(p2-1)    # examine all neighbors of j
            i = Gi [p]                # consider neighbor node i
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

# find nonzero pattern of Cholesky L(k,1:k-1) using etree and triu(A(:,k))
function js_ereach{Tv,Ti}(A::SparseMatrixCSC{Tv,Ti}, k::Integer, parent::Vector{Ti},
                          s::Vector{Ti}, w::Vector{Ti})
    m,n = size(A); Ap = A.colptr; Ai = A.rowval
    top = n
    JS_MARK(w, k)                       # mark node k as visited
    for p in Ap[k]:(Ap[k+1] - 1)
        i = Ai [p]               # A(i,k) is nonzero
        if i > k continue end    # only use upper triangular part of A
        len = 0
        while JS_MARKED(w,i)            # traverse up etree
            s[len] = i                  # L(k,i) is nonzero
            len += 1
            JS_MARK(w,i)                # mark i as visited
            i = parent [i]
        end
        while (len > 0)
            top -= 1
            len -= 1
            s[top] = s[len]             # push path onto stack
        end
    end
    for p in top:(n - 1) JS_MARK(w,s[p]) end # unmark all nodes
    JS_MARK(w,k)                             # unmark node k
    top                      # s [top..n-1] contains pattern of L(k,:)
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

## solve L'x=b where x and b are dense.  x=b on input, solution on output.
function js_ltsolve!{T}(L::SparseMatrixCSC{T}, x::StridedVector{T})
    m,n = size(L)
    if m != n error("Matrix L is $m by $n and should be square") end
    if length(x) != n error("Dimension mismatch") end
    Lp = L.colptr; Li = L.rowval; Lx = L.nzval
    for j in n:-1:1
        for p in (Lp[j] + 1):(Lp[j+1] - 1)
            x[j] -= Lx[p] * x[Li[p]]
        end
        x[j] /= Lx[Lp[j]]
    end
    x
end

js_ltsolve{T}(L::SparseMatrixCSC{T}, x::StridedVector{T}) = js_ltsolve!(L, copy(x))

## 1-norm of a sparse matrix = max (sum (abs (A))), largest column sum
function js_norm(A::SparseMatrixCSC)
    Ap  = A.colptr; Ax  = A.nzval
    nrm = zero(eltype(A))
    for j in 1:size(A,2)
        nrm = max(nrm, sum(abs(Ax[Ap[j]:(Ap[j+1]-1)])))
    end
    nrm
end

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

## solve Gx=b(:,k), where G is either upper (lo=0) or lower (lo=1) triangular
function js_spsolve{T}(G::SparseMatrixCSC{T}, B::SparseMatrixCSC{T}, k::Integer,
                    xi::Vector{Integer}, x::Vector{T}, pinv::Vector{Integer}, lo::Bool)
    Gp = G.colptr; Gi = G.rowval; Gx = G.nzval; n = size(G, 2)
    Bp = B.colptr; Bi = B.rowval; Bx = B.nzval
    ## must have reach return a 1-based index
    top = js_reach (G, B, k, xi, pinv)  # xi[top..n-1]=Reach(B(:,k))
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

## solve Ux=b where x and b are dense.  x=b on input, solution on output.
function js_usolve!{T}(U::SparseMatrixCSC{T}, x::StridedVector{T})
    m,n = size(U)
    if m != n error("Matrix U is $m by $n and should be square") end
    if length(x) != n error("Dimension mismatch") end
    Up = U.colptr; Ui = U.rowval; Ux = U.nzval
    for j in n:-1:1
        x[j] /= Ux[Up[j+1]-1]
        for p in Up[j]:(Up[j+1]-2)
            x[Ui[p]] -= Ux[p] * x[j]
        end
    end
    x
end

js_usolve{T}(U::SparseMatrixCSC{T}, x::StridedVector{T}) = js_usolve!(U, copy(x))

## solve U'x=b where x and b are dense.  x=b on input, solution on output.
function js_utsolve!{T}(U::SparseMatrixCSC{T}, x::StridedVector{T})
    m,n = size(U)
    if m != n error("Matrix U is $m by $n and should be square") end
    if length(x) != n error("Dimension mismatch") end
    Up = U.colptr; Ui = U.rowval; Ux = U.nzval
    for j in 1:n
        for p in Up[j]:(Up[j+1]-2)
            x[j] -= Ux[p] * x[Ui[p]]
            end
        x[j] /= Ux[Up[j+1]-1]
    end
    x
end

js_utsolve{T}(U::SparseMatrixCSC{T}, x::StridedVector{T}) = js_utsolve!(U, copy(x))

## Use the home directory for Julia, once I discover how to obtain it
const TestMats = "/home/bates/build/julia/deps/SuiteSparse-4.0.2/CXSparse/Matrix/"
function testmat(nm::ASCIIString)
    fd = open(file_path(TestMats, nm), "r")
    r1 = split(readline(fd))
    rows = [int32(r1[1])]
    cols = [int32(r1[2])]
    cmpx = length(r1) == 4
    if !cmpx && length(r1) != 3 error("$(file_path(TestMats, nm)) is not a CXSparse test matrix") end
    nzs = cmpx ? [float(r1[3]) + im*float(r1[4])] : [float(r1[3])]
    for rr = EachLine(fd)
        vv = split(rr)
        push(rows, int32(vv[1]))
        push(cols, int32(vv[2]))
        push(nzs, cmpx ? float(vv[3]) + im*float(vv[4]) : float(vv[3]))
    end
    close(fd)
    sparse(rows + one(Int32), cols + one(Int32), nzs)
end

#const _jl_libcsparse = dlopen("libcsparse.so")
#const cs_pvec = dlsym(_jl_libcsparse, :cs_pvec)
#const cs_print = dlsym(_jl_libcsparse, :cs_print)

## Interface to the CSparse and CXSparse libraries
const csp = dlopen("/home/bates/build/julia/deps/SuiteSparse-4.0.2/CSparse/Lib/libcsparse.so")

const csp_amd    = dlsym(csp, :cs_amd)
const csp_counts = dlsym(csp, :cs_counts)
const csp_dfree  = dlsym(csp, :cs_dfree)
const csp_etree  = dlsym(csp, :cs_etree)
const csp_nfree  = dlsym(csp, :cs_nfree)
const csp_post   = dlsym(csp, :cs_post)
const csp_sfree  = dlsym(csp, :cs_sfree)
const csp_spfree = dlsym(csp, :cs_spfree)

load("strpack")

## Copied from extras/suitesparse.jl
function _jl_convert_to_0_based_indexing!(S::SparseMatrixCSC)
    for i=1:(S.colptr[end]-1); S.rowval[i] -= 1; end
    for i=1:length(S.colptr); S.colptr[i] -= 1; end
    return S
end

function _jl_convert_to_1_based_indexing!(S::SparseMatrixCSC)
    for i=1:length(S.colptr); S.colptr[i] += 1; end
    for i=1:(S.colptr[end]-1); S.rowval[i] += 1; end
    return S
end

_jl_convert_to_0_based_indexing(S) = _jl_convert_to_0_based_indexing!(copy(S))
_jl_convert_to_1_based_indexing(S) = _jl_convert_to_1_based_indexing!(copy(S))

type cs                                 # the CSparse cs struct
    nzmax::Int
    m::Int
    n::Int
    p::Ptr{Int}
    i::Ptr{Int}
    x::Ptr{Float64}
    nz::Int
end

function cs(A::SparseMatrixCSC{Float64,Int})
    if A.colptr[1] != 0 error("Sparse matrix must be in 0-based indexing") end
    i = A.rowval
    cs(length(i), A.m, A.n, pointer(A.colptr), pointer(i), pointer(A.nzval), -1)
end

function cs_etree(A::SparseMatrixCSC{Float64,Int}, col::Bool)
    n = size(A, 2)
    iostr = IOString()
    _jl_convert_to_0_based_indexing!(A)
    pack(iostr, cs(A))
    ept   = ccall(csp_etree, Ptr{Int}, (Ptr{Void}, Int), iostr.data, col)
    _jl_convert_to_1_based_indexing!(A)
    popt  = ccall(csp_post, Ptr{Int}, (Ptr{Int}, Int), ept, n)
    etree = pointer_to_array(ept, (n,)) + 1
    post  = pointer_to_array(popt, (n,)) + 1
    c_free(ept)
    c_free(popt)
    etree, post
end

cs_etree(A::SparseMatrixCSC{Float64,Int}) = cs_etree(A, false)

## order 1:Chol, 2:LU, 3:QR
function cs_amd(A::SparseMatrixCSC{Float64,Int}, order::Int)
    order = min(max(1, order), 3)
    n = size(A, 2)
    iostr = IOString()
    _jl_convert_to_0_based_indexing!(A)
    pack(iostr, cs(A))
    ppt   = ccall(csp_amd, Ptr{Int}, (Int, Ptr{Void}), order, iostr.data)
    _jl_convert_to_1_based_indexing!(A)
    p = pointer_to_array(ppt, (n,)) + 1
    c_free(ppt)
    p
end

cs_amd(A::SparseMatrixCSC{Float64,Int}) = cs_amd(A, 1)

function cs_counts(A::SparseMatrixCSC{Float64,Int}, col::Bool)
    n = size(A, 2)
    iostr = IOString()
    _jl_convert_to_0_based_indexing!(A)
    pack(iostr, cs(A))
    etrpt = ccall(csp_etree, Ptr{Int}, (Ptr{Void}, Int), iostr.data, col)
    pospt = ccall(csp_post, Ptr{Int}, (Ptr{Int}, Int), etrpt, n)
    coupt = ccall(csp_counts, Ptr{Int}, (Ptr{Void}, Ptr{Int}, Ptr{Int}, Int),
                  iostr.data, etrpt, pospt, col)
    _jl_convert_to_1_based_indexing!(A)
    pointer_to_array(etrpt, (n,)) + 1, pointer_to_array(pospt, (n,)) + 1,pointer_to_array(coupt, (n,))
end

cs_counts(A::SparseMatrixCSC{Float64,Int}) = cs_counts(A, false)

