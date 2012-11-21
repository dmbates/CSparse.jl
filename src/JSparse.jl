require("sparse")

# compute the etree of A (using triu(A), or A'A without forming A'A
function js_etree{Tv,Ti}(A::SparseMatrixCSC{Tv,Ti}, ata::Bool)
    m,n = size(A); Ap = A.colptr; Ai = A.rowval
    parent = zeros(Ti, n); w = zeros(Ti, n + (ata ? m : 0))
    ancestor = 0; prev = n              # offsets into w
    if (ata) w[prev + (1:m)] = -1 end
    for k in 1:n
        parent[k] = -1                  # node k has no parent yet
        w[ancestor + k] = -1            # nor does k have an ancestor
        for p in Ap[k]:(Ap[k+1] - 1)
            i = ata ? w[Ai[p] + prev] : Ai[p]
            while i != -1 && i < k
                inext = w[ancestor + i] # inext = ancestor of i
                w[ancestor + i] = k     # path compression
                if (inext == -1) parent[i] = k end # no anc., parent is k
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
        p2 = (jnew < 0) ? 0 : JS_UNFLIP (Gp [jnew+1]) ;
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

## xi [top...n-1] = nodes reachable from graph of G*P' via nodes in B[:,k]
## xi [n...2n-1] used as workspace
function js_reach{T}(G::SparseMatrixCSC{T}, B::SparseMatrixCSC{T}, k::Integer,
                     xi::Vector{Integer}, pinv::Vector{Integer})
    n = size(G,2); Bp = B.colptr; Bi = B.rowval; Gp = G.colptr
    top = n
    for p in Bp[k]:(Bp[k+1]-1)
        if !JS_MARKED(Gp, Bi[p])      # start a dfs at unmarked node i
            top = js_dfs (Bi [p], G, top, xi, xi+n, pinv)
        end
    end
    for p in top:n JS_MARK (Gp, xi [p]) end  #restore G
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
