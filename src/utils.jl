"""
    from_transition_matrix(trn_mat::AbstractMatrix{Rational{T}}, x::Vector{PolyVar{true}}) where T<:Integer

Return the Dirichlet form (as a polynomial), invariant distribution (as a vector), and the spectral 
gap of a Markov chain (assumed irreducible) given its transition matrix.
"""
function from_transition_matrix(trn_mat::AbstractMatrix{Rational{T}}, x::Vector{PolyVar{true}}) where T<:Integer
    # Laplacian
    L = I - trn_mat
    inv_dist = [1//1; - (L[:, 2:end]' * L[:, 2:end]) \ (L[:, 2:end]' * L[:, 1])]
    inv_dist .//= sum(inv_dist)
    dir_form = (inv_dist.*x)'*L*x
    # spectral gap
    λ = 0.5*real.(eigvals(Float64.(inv_dist' .* L' ./inv_dist .+ L)))[2]
    (dir_form, inv_dist, λ)
end
"""
    from_transition_matrix(trn_mat::AbstractMatrix{Float64}, x::Vector{PolyVar{true}})

Return the Dirichlet form (as a polynomial), invariant distribution (as a vector), and the spectral 
gap of a Markov chain (assumed irreducible) given its transition matrix.
"""
function from_transition_matrix(trn_mat::AbstractMatrix{Float64}, x::Vector{PolyVar{true}})
    # Laplacian
    L = I - trn_mat
    @assert abs(eigvals(L)[1]) < 1e-10
    inv_dist = eigvecs(L')[:, 1]
    inv_dist ./= sum(inv_dist)
    dir_form = (inv_dist.*x)'*L*x
    # spectral gap
    λ = 0.5*real.(eigvals(inv_dist' .* L' ./ inv_dist .+ L))[2]
    (dir_form, inv_dist, λ)
end

"""
    get_pade(t::DynamicPolynomials.PolyVar{true}, k::Integer)

Return a tuple `(2*(1+t)^2 * num, den)`, such that the quoteient of  `num` and 
`den` is the ``[k, k+1]`` Pade approximant of ``log(1+t)`` at 0. 
"""
function get_pade(t::PolyVar{true}, k::Integer)
    c(j) = (-1)^(j+1) // (j+2)
    C = [c(i + j) for i=0:k-1, j=0:k-1]
    q = - C \ c.(k:2k-1)
    Q = 1//1 + q'*t.^(k:-1:1)
    tay = c.(-1:k-1)' * t.^(1:(k+1))
    P = removemonomials(Q * tay, t.^(k+2:2k+1))
    (num, den) = 2*(1+t)^2*P, Q
end

"""
    get_linind_cols(A::SparseMatrixCSC{T,Int64}) where T

Find a set of linearly independent columns.

This function is guaranteed to be correct only if `T` subtypes `Rational{<:Integer}`.
"""
function get_linind_cols(A::SparseMatrixCSC{T,Int64}) where T
    get_linind_cols(Float64.(A))
end
function get_linind_cols(A::SparseMatrixCSC{Float64,Int64})
    F = qr(A)
    linind = []
    track = 0
    for i in 1:size(F.R)[2]
        nt = findlast(v -> abs(v) > 1e-15, F.R[:, i])
        if !isnothing(nt) && nt > track
            push!(linind, i)
            track = nt
        end
    end
    linind = F.pcol[linind]
end
function get_linind_cols(A::SparseMatrixCSC{Rational{T},Int64}) where T<:Integer
    r, RREF = Nemo.rref(Nemo.matrix(Nemo.QQ, Matrix(A)))
    RREF = Matrix(RREF)
    linind = [findfirst(RREF[i, :] .== 1) for i in 1:r]
end

"""
    check_pos_def(Q::Matrix{Rational{T}}) where T<:Integer

Verify a rational matrix is positive definite using interval arithmetic.
"""
function check_pos_def(Q::Matrix{Rational{T}}) where T<:Integer
    RR = Nemo.ArbField()
    X = Nemo.matrix(RR, Q)
    L = similar(X)
    b = ccall((:arb_mat_cho, Nemo.libarb), Cint,
            (Ref{Nemo.arb_mat}, Ref{Nemo.arb_mat}, Int),
            L, X, precision(Nemo.base_ring(X)))
    return b != 0 
end

"""
    check_pos_def(Q::Matrix{Float64})

Verify a floating point matrix is positive definite.
"""
function check_pos_def(Q::Matrix{Float64})
    return all(eigvals(Q) .> 0)
end

"""
    ldl(Q::Matrix{T}) where T

Compute LDL^T factorization of a matrix, presumed positive definite.
"""
function ldl(Q::Matrix{T}) where T
    n = size(Q)[1]
    @assert size(Q)[2] == n
    L = zeros(T, n, n)
    D = zeros(T, n)
    for i in 1:n
        D[i] = Q[i,i] - (i == 1 ? 0 : sum(D[j]*L[i,j]^2 for j in 1:i-1))
        if D[i] == 0
            D = D[1:i-1]
            L = L[:, 1:i-1]
            break
        end
        for j in i+1:n
            L[j, i] = (Q[j,i] - (i==1 ? 0 : sum(D[k]*L[j,k]*L[i,k] for k in 1:i-1)))/D[i]
        end
    end
    for k in 1:minimum(size(L))
        L[k, k] = 1
    end
    @assert all(Q .== L*diagm(D)*L')
    (D, L)
end

"""
    rat_upper_bound_sqrt(s::Rational{BigInt}, denom::BigInt=big(2)^33)

Return a rational upper bound on the square root of a number.
"""
function rat_upper_bound_sqrt(s::Rational{<:Integer}; denom::BigInt=big(2^33))
    BigInt(ceil(denom * sqrt(s))) // denom
end

"""
    default_relaxation(kernel::Matrix{<:Real})   

Build a [PadeRelaxation](@PadeRelaxation) SoS relaxation of the given type for estimating 
the log-Sobolev constant of the given kernel, with sensible default parameters.
The relaxation will be parametrized by the `eltype` of `kernel`.
The defaults are such that the relaxation has the form

``
\\begin{aligned}
\\operatorname{minimize} \\gamma \\text{ s.t. } & γ\\mathcal{E}(x,x) - \\sum_{i=1}^n \\pi_i p(x_i) + 
\\left(\\sum_{i=1}^n\\pi_i(x_i^2+2x_i)\\right)h(x) \\in \\Sigma(V)\\\\
& h \\in V_h,
``

where 
- ``V`` is the space of degree 3 polynomials in ``x_1, ..., x_n`` with no constant 
term and no monomial ``x_ix_jx_k`` with ``i,j,k`` all different.
- ``V_h`` is the space of degree 3 polynomials where each monomial has at most one variable.
"""
function default_relaxation(kernel::Matrix{<:Real})
    @assert size(kernel)[1] == size(kernel)[2]
    n = size(kernel)[1]
    @polyvar x[1:n] 
    dir_form, inv_dist, λ = from_transition_matrix(kernel, x)
    mons = monomials(x, 1:3, mon -> length(effective_variables(mon)) <= 2)
    v_bases = [GramBasis(mons)]    
    h_basis = monomials(x, 0:4, mon -> length(effective_variables(mon)) <= 1)
    relaxation = PadeRelaxation(dir_form, inv_dist, v_bases, h_basis)
end

"""
    solve_relaxation!(relaxation::LSRelaxation, optimizer; tol=1e-8)

Convenience function grouping together the steps required to solve a
    sum-of-squares relaxation.
"""
function solve_relaxation!(relaxation::LSRelaxation, optimizer; tol=1e-8)
    lin_reduce!(relaxation)
    set_epsilon!(relaxation, tol)
    solve_sdp!(relaxation, optimizer)
    round_sol!(relaxation)
    @assert(all(check_pos_def(X) for X in extract_mat_vars(relaxation)), 
                "Error: a matrix variable is no longer PSD after projection")
    γ = extract_gamma(relaxation)
    inv(γ)
end