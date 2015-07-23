using Metis,CSparse
using Base.Test

function specialize(A::SparseMatrixCSC)
    issym(A) && return Symmetric(triu!(A),:U)
    ishermitian(A) && return Hermitian(triu!(A),:U)
    istril(A) && return LowerTriangular(A)
    istriu(A) && return UpperTriangular(A)
    A
end

cd(joinpath(dirname(@__FILE__),"..","Matrix")) do
    global mats = [specialize(testmat(nm)) for nm in readdir()]
end

@test size(mats[1]) == (219,85)

@test size(mats[2]) == (48,48)
const o48 = ones(48)
@test_approx_eq mats[2]*(mats[2]\o48) o48
@test_approx_eq Ac_mul_B(mats[2],Ac_ldiv_B(mats[2],o48)) o48

const bc01t = mats[2]'
@test istriu(bc01t)
@test isa(bc01t,UpperTriangular)
@test size(bc01t) == (48,48)
@test_approx_eq bc01t*(bc01t\o48) o48
@test_approx_eq Ac_mul_B(bc01t,Ac_ldiv_B(bc01t,o48)) o48

@test size(mats[3]) == (4884,4884)
const o4884 = ones(4884)
@test_approx_eq mats[3]*(mats[3]\o4884) o4884
@test_approx_eq Ac_mul_B(mats[3],Ac_ldiv_B(mats[3],o4884)) o4884

const bc16t = mats[3]'
@test istriu(bc16t)
@test isa(bc16t,UpperTriangular)
@test size(bc16t) == (4884,4884)
@test_approx_eq bc16t*(bc16t\o4884) o4884
@test_approx_eq Ac_mul_B(bc16t,Ac_ldiv_B(bc16t,o4884)) o4884

const sc1 = Symmetric(triu(mats[2].data'mats[2].data),:U)
const perm1,iperm1 = nodeND(sc1)
const tr1 = etree(sc1)
const sc2 = symperm(sc1,iperm1)
const tr2,postperm = etree(sc2,true)
const sc3 = symperm(sc2,invperm(postperm))
const tr3 = etree(sc3)
@show(tr1)
@show(tr2)
@show(tr3)
