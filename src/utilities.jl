function Base.etree{Tv,Ti}(A::Symmetric{Tv,SparseMatrixCSC{Tv,Ti}},postorder::Bool=false)
    etree(A.uplo == 'U' ? A.data : A.data', postorder)
end

function Base.symperm{Tv,Ti}(A::Symmetric{Tv,SparseMatrixCSC{Tv,Ti}},invperm)
    Symmetric(symperm(A.uplo == 'U' ? A.data : A.data',invperm),:U)
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
