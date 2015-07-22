const matrix_directory = Pkg.dir("CSparse", "Matrix")

function testmat(nm)
    fd = open(joinpath(matrix_directory, nm), "r")
    r1 = split(readline(fd))
    rows = [parse(Int32,r1[1])]
    cols = [parse(Int32,r1[2])]
    cmpx = length(r1) == 4
    if !cmpx && length(r1) != 3
        error("$(file_path(TestMats, nm)) is not a CXSparse test matrix")
    end
    nzs = cmpx ? [float(r1[3]) + im*float(r1[4])] : [float(r1[3])]
    for rr = EachLine(fd)
        vv = split(rr)
        push!(rows,parse(Int32,vv[1]))
        push!(cols,parse(Int32,vv[2]))
        push!(nzs, cmpx ? float(vv[3]) + im*float(vv[4]) : float(vv[3]))
    end
    close(fd)
    sparse(rows .+ one(Int32), cols .+ one(Int32), nzs)
end
