module OddCycleUtils

import LogSobolevRelaxations: LSRelaxation, GramBasis, check_pos_def, round_sol!
import Nemo
import DynamicPolynomials
using LinearAlgebra: dot

export RealCyclotomicFieldElem, get_cycle_basis

"""
    struct RealCyclotomicFieldElem{N} <: Real
        v::Nemo.nf_elem
    end

Represents an element of the number field ``\\mathbb{Q}[2\\cos 2π/N]``.

The implementation is just a wrapper for an Antic `nf_elem` object accessed via Nemo.
"""
struct RealCyclotomicFieldElem{N} <: Real

    v::Nemo.nf_elem

    function RealCyclotomicFieldElem{N}(x::Union{Integer, Rational}) where N
        @assert N isa Int64 && N > 0
        K = Nemo.MaximalRealSubfield(N, "theta")[1]
        return new{N}(K(x))
    end
    function RealCyclotomicFieldElem{N}(x::Nemo.nf_elem) where N
        @assert N isa Int64 && N > 0
        K = Nemo.MaximalRealSubfield(N, "theta")[1]
        Nemo.parent(x) == K ? new{N}(x) : error("Incompatible number field")
    end

end
    
Base.show(io::IO, a::RealCyclotomicFieldElem{N}) where N =  print(io, "(",  a.v, ")")
Base.:-(a::RealCyclotomicFieldElem{N}) where N = RealCyclotomicFieldElem{N}(-a.v)
Base.inv(a::RealCyclotomicFieldElem{N}) where N = RealCyclotomicFieldElem{N}(1//a.v)
Base.:+(a::RealCyclotomicFieldElem{N}, b::RealCyclotomicFieldElem{N}) where N = RealCyclotomicFieldElem{N}(a.v + b.v)
Base.:-(a::RealCyclotomicFieldElem{N}, b::RealCyclotomicFieldElem{N}) where N = RealCyclotomicFieldElem{N}(a.v - b.v)
Base.:*(a::RealCyclotomicFieldElem{N}, b::RealCyclotomicFieldElem{N}) where N = RealCyclotomicFieldElem{N}(a.v * b.v)
Base.://(a::RealCyclotomicFieldElem{N}, b::RealCyclotomicFieldElem{N}) where N = RealCyclotomicFieldElem{N}(a.v // b.v)
Base.:/(a::RealCyclotomicFieldElem{N}, b::RealCyclotomicFieldElem{N}) where N = RealCyclotomicFieldElem{N}(a.v // b.v)
Base.:(==)(a::RealCyclotomicFieldElem{N}, b::RealCyclotomicFieldElem{N}) where N = a.v == b.v

function Base.:<(a::RealCyclotomicFieldElem{N}, b::RealCyclotomicFieldElem{N}) where N 
    K =  Nemo.MaximalRealSubfield(N, "theta")[1]
    R = Nemo.parent(K.pol)
    theta_val = 2*Nemo.cospi(Nemo.QQ(2//N), Nemo.ArbField())
    R(a.v)(theta_val) < R(b.v)(theta_val)
end

function Base.Float64(x::RealCyclotomicFieldElem{N}) where N 
    theta_val = 2*Rational(cospi(big(2)//N))
    R = Nemo.parent(Nemo.parent(x.v).pol)
    Float64(Rational(R(x.v)(theta_val)))
end

Base.copy(x::RealCyclotomicFieldElem{N}) where N = RealCyclotomicFieldElem{N}(deepcopy(x.v))
RealCyclotomicFieldElem{N}(a::RealCyclotomicFieldElem{N}) where N = copy(a)

Base.promote_rule(::Type{RealCyclotomicFieldElem{N}}, ::Type{Rational{S}}) where {N, S<:Integer} = RealCyclotomicFieldElem{N}
Base.promote_rule(::Type{RealCyclotomicFieldElem{N}}, ::Type{<:Integer}) where N = RealCyclotomicFieldElem{N}
Base.promote_rule(::Type{RealCyclotomicFieldElem{N}}, ::Type{Float64}) where N = Float64    

function round_sol!(relaxation::LSRelaxation{RealCyclotomicFieldElem{N}}) where N
    x_flt = relaxation.solution
    @assert eltype(x_flt) <: AbstractFloat
    n_var = length(x_flt)
    data = relaxation.sdpdata
    A, b = (data.A, data.b)

    K =  Nemo.MaximalRealSubfield(N, "theta")[1]
    combined = Nemo.matrix(K, map(val -> val.v, Matrix([A b])))
    rank, combined = Nemo.rref(combined)
    combined = RealCyclotomicFieldElem{N}.(Matrix(combined))

    dependent = Dict(findfirst(combined[row, :] .!= 0) => row for row in 1:rank)
    indep = Dict(i => indep for (indep, i) in enumerate(filter(v->!(v in keys(dependent)), 1:n_var+1)))

    Q = zeros(RealCyclotomicFieldElem{N}, n_var + 1, n_var + 1 - rank)
    for (i, indep) in indep
        Q[i, indep] = 1
        for (j, depj) in dependent
            Q[j, indep] = - combined[depj, i]
        end
    end
    M = Q[1:end-1, 1:end-1]
    u = Q[1:end-1, end]
    z = Float64.(M) \ (Float64.(u) + Float64.(x_flt))
    z = convert.(Rational{BigInt}, z)

    relaxation.solution = M*z - u
end

"""
    check_pos_def(Q::Matrix{RealCyclotomicFieldElem{N}}) where N

Verify a matrix is positive definite using interval arithmetic.
"""
function check_pos_def(Q::Matrix{RealCyclotomicFieldElem{N}}) where N
    K =  Nemo.MaximalRealSubfield(N, "theta")[1]
    R = Nemo.parent(K.pol)
    theta_val = 2*Nemo.cospi(Nemo.QQ(2//N), Nemo.ArbField())
    X = map(val -> R(val.v)(theta_val), Q)
    X = Nemo.matrix(Nemo.ArbField(), X)
    L = similar(X)
    b = ccall((:arb_mat_cho, Nemo.libarb), Cint,
            (Ref{Nemo.arb_mat}, Ref{Nemo.arb_mat}, Int),
            L, X, precision(Nemo.base_ring(X)))
    return b != 0 
end

"""
    get_cycle_basis(x::Vector{DynamicPolynomials.PolyVar{true}})

Symmetry adapted basis for sums of squares polynomials invariant
under dihedral permutations of its elements, only valid when `N` is an odd 
positive integer.

Suppose that ``q(x)`` is an `N`-variate polynomial which is a sum of squares of 
polynomials in a polynomial subspace ``V\\subset \\mathbb{R}[x]_{N}``.
Then one can find a basis of ``V`` with respect to which ``q`` has a block diagonal 
positive semidefinite representation.
This function returns such a basis for the specific choice 
    
``W = \\left\\{q\\in\\R[x]_{n,3}\\; \\middle\\vert \\; \\begin{array}{l}
q(0) = 0,\\\\
 ϕ^\\top\\nabla q(0) = 0,\\\\
 ψ^\\top\\nabla q(0) = 0,\\\\
\\dfrac{d^3q}{dx_idx_jdx_k}(0) = 0\\qquad \\forall\\; i,\\, j,\\, k \\text{ distinct}
\\end{array}
\\right\\}``
    
where ``ϕ=\\sum_{1}^N\\cos(2j\\pi/N)x_j`` and ``ψ=\\sum_{1}^N\\sin(2j\\pi/N)x_j``.
The coefficients of the polynomials in the resulting basis are of type 
[`RealCyclotomicFieldElem{N}`](@ref).

We also impose a particular sparsity pattern on each Gram matrix.

See 
 - Gatermann, K. & Parrilo, P. A.
*Symmetry groups, semidefinite programs, and sums of squares*
Journal of Pure and Applied Algebra, **2004**, 192, 95-128
 - Section [TODO] of our paper for an explanation of ``ϕ, ψ`` above.`
"""
function get_cycle_basis(x::Vector{DynamicPolynomials.PolyVar{true}})

    n = length(x)
    @assert n%2 == 1
    mm = Int((n-1)/2)
    K, theta = Nemo.MaximalRealSubfield(n, "theta")
    R = Nemo.parent(K.pol)
    θ = (1//2) * Nemo.gen(R)

    isomorphic_to_perm_rep = [vcat(pre, zeros(Int64, n-2)) for pre in [[1,0], [2,0], [3,0]]]
    for k in 1:mm
        newpattern = zeros(Int64, n)
        newpattern[1 + k] = 1
        newpattern[end - k + 1] = 1
        push!(isomorphic_to_perm_rep, newpattern)
    end

    isomorphic_to_reg_rep = [zeros(Int64, n) for k in 1:mm]
    for k in 1:mm
        isomorphic_to_reg_rep[k][1] = 2
        isomorphic_to_reg_rep[k][1+k] = 1
    end

    T = [DynamicPolynomials.Polynomial{true, RealCyclotomicFieldElem{n}}[] for k in 1:n+1]

    for pattern in isomorphic_to_perm_rep
        mons = [prod(x.^pattern[mod1.((1:n) .- j, n)]) for j in 0:n-1]
        push!(T[end], sum(mons))
        for k in 1:mm
            char1 = RealCyclotomicFieldElem{n}.(K.([Nemo.chebyshev_t(j*k, θ) for j in 1:n]))
            char2 = RealCyclotomicFieldElem{n}.(K.([Nemo.chebyshev_u(j*k - 1, θ) for j in 1:n]))
            push!(T[2*k-1], dot(char1, mons))
            push!(T[2*k], dot(char2, mons))
        end
    end

    for pattern in isomorphic_to_reg_rep
        mons1 = [prod(x.^pattern[mod1.((1:n) .- j, n)]) for j in 0:n-1]
        mons2 = [prod(x.^pattern[mod1.((n:-1:1) .+ j, n)]) for j in 1:n]
        push!(T[end], sum(mons1) + sum(mons2))
        push!(T[end-1], sum(mons1) - sum(mons2))
        for k in 1:mm
            char1 = RealCyclotomicFieldElem{n}.(K.([Nemo.chebyshev_t(j*k, θ) for j in 1:n]))
            char2 = RealCyclotomicFieldElem{n}.(K.([Nemo.chebyshev_u(j*k - 1, θ) for j in 1:n]))
            push!(T[2*k-1], dot(char1, mons1) + dot(char1, mons2))
            push!(T[2*k-1], dot(char2, mons1) - dot(char2, mons2))
            push!(T[2*k], dot(char2, mons1) + dot(char2, mons2))
            push!(T[2*k], - dot(char1, mons1) + dot(char1, mons2))
        end
    end

    T[1] = T[1][2:end] # ϕ⋅x  = T[1][1]
    T[2] = T[2][2:end] # ψ⋅x ∝ T[2][1]

    filter!(t->length(t)>=1, T)
    reverse!.(T)

    v_bases = GramBasis{RealCyclotomicFieldElem{n}}[]

    for t in T
        diag_block_idxs = []
        for (i, poly) in enumerate(t)
            mon = DynamicPolynomials.monomial(DynamicPolynomials.terms(poly)[1])
            if length(DynamicPolynomials.effective_variables(mon)) >= 2 && DynamicPolynomials.degree(mon) >= 2
                push!(diag_block_idxs, i)
            end
        end
        sp = Set{CartesianIndex{2}}([CartesianIndex(i, j) for i ∈ diag_block_idxs for j ∈ diag_block_idxs if i != j])
        push!(v_bases, GramBasis(t, sp))
    end
    v_bases

end

end