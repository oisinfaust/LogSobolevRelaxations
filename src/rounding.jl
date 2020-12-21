"""
    round_sol(relaxation::LSRelaxation{Rational{BigInt}})

Project an approximate floating point solution onto the exact linear space of rational 
solutions.
"""
function round_sol!(relaxation::LSRelaxation{Rational{BigInt}})
    @assert eltype(relaxation.solution) <: AbstractFloat
    x_r = convert.(Rational{BigInt}, relaxation.solution)
    data = relaxation.sdpdata
    A, b = (data.A, data.b)

    gram = Nemo.matrix(Nemo.QQ, Matrix(A*A'))
    rhs = Nemo.matrix(Nemo.QQ, reshape(A*x_r - b, :, 1))
    relaxation.solution = x_r - A' * Rational.(Matrix(Nemo.solve(gram, rhs)))[:]
end
"""
    round_sol!(relaxation::LSRelaxation{Float64})

Project an approximate floating point solution so that it satisfies the linear 
constraints of the SDP to machine precision.
"""
function round_sol!(relaxation::LSRelaxation{Float64})
    @assert eltype(relaxation.solution) <: AbstractFloat
    data = relaxation.sdpdata
    A, b = (data.A, data.b)
    relaxation.solution = relaxation.solution .- (A \ Vector(A*relaxation.solution - b))
end