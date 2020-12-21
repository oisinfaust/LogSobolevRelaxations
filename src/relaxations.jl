"""
    struct GramBasis{T}
        dim::Integer
        basis::AbstractVector{<:AbstractPolynomialLike{T}}
        sparsity_pattern::Set{CartesianIndex{2}}
    end

Represents a polynomial basis with respect to which a sum of squares
polynomial can be represented as a quadratic form.
"""
struct GramBasis{T}
    dim::Integer
    basis::AbstractVector{<:AbstractPolynomialLike{T}}
    sparsity_pattern::Set{CartesianIndex{2}}

    function GramBasis(dim::Integer, basis::AbstractVector{<:AbstractPolynomialLike{T}}, 
                       sparsity_pattern::Set{CartesianIndex{2}}) where T
        new{T}(dim, basis, sparsity_pattern)
    end
end
GramBasis(basis::AbstractVector{<:AbstractPolynomialLike}, 
                sparsity_pattern::Set{CartesianIndex{2}}) = GramBasis(length(basis), basis, sparsity_pattern)
GramBasis(basis::AbstractVector{<:AbstractPolynomialLike}) = GramBasis(basis, Set{CartesianIndex{2}}([]))
GramBasis{S}(gb::GramBasis{T}) where {S, T} = GramBasis(gb.dim, polynomial.(gb.basis, S), gb.sparsity_pattern)

"""
    struct PSDVarStructure
        side_length::Integer
        range::UnitRange
        nonzeros::Vector{Integer}
    end

Describes the relative position and sparsity structure of a vectorized positive 
semidefinite variable within a longer vector of variables.

 - `side_length` is the side length of the PSD matrix `X` in question.
 - `range` denotes the indices of the longer vector of variables corresponding 
 to elements of `X`.
 - `nonzeros` is such that `solution[range] .== mat2vec(X)[nonzeros]`.
"""
struct PSDVarStructure
    side_length::Integer
    range::UnitRange
    nonzeros::Vector{Integer}
end

function PSDVarStructure(side_length::Integer, range_start::Integer)
    r = 1:div(side_length*(side_length+1), 2)
    range = range_start .+ r
    PSDVarStructure(side_length, range, collect(r))
end
function PSDVarStructure(side_length::Integer, range_start::Integer, sp::Set{CartesianIndex{2}})
    nonzeros = [binomial(i, 2) + j for i in 1:side_length for j in 1:i if !(CartesianIndex(i, j) ∈ sp)]
    range = range_start .+ (1:length(nonzeros))
    PSDVarStructure(side_length, range, nonzeros)
end

# The following functions map a symmetric matrix between matrix form, 
# vectorized lower triangular form, and sparse vectorized form.
function psdvar2vec(v::Vector{D}, psdvar::PSDVarStructure) where D
    w = zeros(D, div(psdvar.side_length*(psdvar.side_length+1), 2))
    for (i, idx) in enumerate(psdvar.nonzeros)
        w[idx] = v[i]
    end
    return w
end
vec2psdvar(v::Vector, psdvar::PSDVarStructure) = v[psdvar.nonzeros]
function vec2mat(v::Vector, psdvar::PSDVarStructure)
    s = psdvar.side_length
    idxs = [binomial(max(i, j), 2) + min(i, j) for i=1:s, j=1:s]
    return v[idxs]
end
mat2vec(v::Matrix, psdvar::PSDVarStructure) = [v[i, j] for i in 1:psdvar.side_length for j in 1:i]


"""
    mutable struct SDPData{T}
        varlen::Integer
        A::SparseMatrixCSC{T}
        b::SparseVector{T}
        psdvars::Vector{PSDVarStructure}
    end

Describes a semidefinite program with data of type `T`.

The SDP has the form 

``
\\begin{aligned}
\\operatorname{min} z_1 \\text{ s.t. } & Az=b, \\\\
                                       & X_i\\succeq ϵI \\; \\forall i\\in M
\\end{aligned}
``

where the ``X_i`` are symmetric matrices whose lower diagonal elements are
obtained from ``z`` according to the information in `psdvars[i]`.
"""
mutable struct SDPData{T}
    varlen::Integer
    A::SparseMatrixCSC{T}
    b::SparseVector{T}
    psdvars::Vector{PSDVarStructure}
    ϵ::Float64
end

"""
    common_type_for_relaxation(dir_form, inv_dist, v_bases::Vector{GramBasis{T}}, h_basis) where T

Decide which type to store data as.
"""
function common_type_for_relaxation(dir_form, inv_dist, v_bases::Vector{GramBasis{T}}, h_basis) where T
    K = promote_type(coefficienttype(dir_form), eltype(inv_dist), coefficienttype.(h_basis)..., T)
    if K <: AbstractFloat K = Float64 end
    if K <: Rational K = Rational{BigInt} end
    K
end

"""
    get_affine_matrix_block(data::Vector{Polynomial{true, K}}) where K

Return a (sparse) matrix of coefficients, with each column corresponging to an 
entry of `data`, and each row corresponding to a different monomial appearing 
in some entry of `data`.
"""
function get_affine_matrix_block(data::Vector{Polynomial{true, K}}) where K
    full_lookup = Dict{Monomial{true}, Int64}()
    I,J,V = Int64[], Int64[], K[]
    for (i, m) in enumerate(data)
        for term in terms(m)
            row = get!(full_lookup, monomial(term), length(full_lookup) + 1)
            push!(I, row); push!(J, i); push!(V, coefficient(term))
        end
    end
    return sparse(I, J, V, length(full_lookup), length(data))
end


abstract type LSRelaxation{T} end

"""
    mutable struct TaylorRelaxation{T} <: LSRelaxation{T}
        v_bases::Vector{GramBasis}
        h_basis::Vector{Polynomial{true, T}}
        dir_form::Polynomial{true, T}
        inv_dist::Vector{T}
        sdpdata::SDPData{T}
        solution::Union{Nothing, Vector{Float64}, Vector{T}}
    end
    
Represents a polynomial optimization problem

``
\\operatorname{minimize} \\gamma \\text{ s.t. } γ\\mathcal{E}(x,x) - \\sum_{i=1}^n \\pi_i p(x_i) + 
\\left(\\sum_{i=1}^n\\pi_i(x_i^2+2x_i)\\right)h(x) \\in \\bigoplus_j\\Sigma(B_j)
``

where ``p`` is the Taylor expansion of ``2(1+t)^2\\log(1+t)`` around 0.
Here ``\\Sigma(B_j)`` means the cone of sume of squares of polynomials 
spanned by `v_bases[j]`.
"""
mutable struct TaylorRelaxation{T} <: LSRelaxation{T}
    v_bases::Vector{GramBasis{T}}
    h_basis::Vector{Polynomial{true, T}}
    dir_form::Polynomial{true, T}
    inv_dist::Vector{T}
    sdpdata::SDPData{T}
    solution::Union{Nothing, Vector{Float64}, Vector{T}}
    x::Vector{PolyVar{true}}
end
Base.show(io::IO, relaxation::TaylorRelaxation{T}) where T = print(io, 
                                            "Taylor-type relaxation using number type ", string(T))

"""
    mutable struct PadeRelaxation{T} <: LSRelaxation{T}
       ...
    end

Represents a polynomial optimization problem

``
\\operatorname{minimize} \\gamma \\text{ s.t. } γ\\mathcal{E}(x,x) - \\sum_{i=1}^n \\pi_i p_i(x_i) + 
\\left(\\sum_{i=1}^n\\pi_i(x_i^2+2x_i)\\right)h(x) \\in \\bigoplus_j\\Sigma(B_j)
``

where each ``p_i`` can vary subject
to dominating ``2(1+t)^2\\log(1+t)`` on ``t\\in[-1, -1+\\pi_i^{-1/2}]``.
"""
mutable struct PadeRelaxation{T} <: LSRelaxation{T}
    v_bases::Vector{GramBasis{T}}
    h_basis::Vector{Polynomial{true, T}}
    dir_form::Polynomial{true, T}
    inv_dist::Vector{T}
    sdpdata::SDPData{T}
    solution::Union{Nothing, Vector{Float64}, Vector{T}}
    x::Vector{PolyVar{true}}
end
Base.show(io::IO, relaxation::PadeRelaxation{T}) where T = print(io, 
                                "Padé-type relaxation using number type ", string(T))

function TaylorRelaxation(dir_form::Polynomial{true}, inv_dist::Vector, v_bases::Vector{GramBasis{T}}, 
                                            h_basis::AbstractVector{<:AbstractPolynomialLike}) where T
    K = common_type_for_relaxation(dir_form, inv_dist, v_bases, h_basis)
    dir_form = polynomial(dir_form, K)
    inv_dist = K.(inv_dist)
    v_bases = GramBasis{K}.(v_bases)
    h_basis = polynomial.(h_basis, K)

    x = variables(dir_form)
    n = length(x)
    @assert n == length(inv_dist)
    d = maximum(maxdegree.(Iterators.flatten(map(b->b.basis, v_bases))))

    columns = [-dir_form]
    append!(columns, [-m*dot(inv_dist, x.^2 .+ 2*x) for m in h_basis])
    psdvars = PSDVarStructure[]
    for q_basis in v_bases
        psdvar = PSDVarStructure(q_basis.dim, length(columns), q_basis.sparsity_pattern)
        push!(psdvars, psdvar)
        append!(columns, [(1 + Int(i!=j))*q_basis.basis[i]*q_basis.basis[j] for i in 
                    1:q_basis.dim for j in 1:i if !(CartesianIndex(i, j) ∈ q_basis.sparsity_pattern)])
    end

    k = 2*d - 1
    @polyvar t
    # Taylor expansion to order k of 2(1+t)^2*log(1+t)
    p = K(2)t + K(3)t^2 + sum(K(4*(-1)^(i-1)//(i*(i-1)*(i-2)))t^i for i in 3:k)
    push!(columns, -dot(inv_dist, subs(p, t => x[i]) for i in 1:n))
    A = get_affine_matrix_block(columns)

    b = A[:, end]
    A = A[:, 1:end-1]
    sdpdata = SDPData{K}(size(A)[2], A, b, psdvars, 0)
    TaylorRelaxation{K}(v_bases, h_basis, dir_form, K.(inv_dist), sdpdata, nothing, x)
end

function PadeRelaxation(dir_form::Polynomial{true}, inv_dist::Vector, v_bases::Vector{GramBasis{T}}, 
                                h_basis::AbstractVector{<:AbstractPolynomialLike}; k=5) where T
    K = common_type_for_relaxation(dir_form, inv_dist, v_bases, h_basis)
    dir_form = polynomial(dir_form, K)
    inv_dist = K.(inv_dist)
    v_bases = GramBasis{K}.(v_bases)
    h_basis = polynomial.(h_basis, K)

    x = variables(dir_form)
    n = length(x)
    @assert n == length(inv_dist)
    d = maximum(maxdegree.(Iterators.flatten(map(b->b.basis, v_bases))))

    columns = [-dir_form]    
    append!(columns, [-m*dot(inv_dist, x.^2 .+ 2*x) for m in h_basis])
    psdvars = PSDVarStructure[]
    for q_basis in v_bases
        psdvar = PSDVarStructure(q_basis.dim, length(columns), q_basis.sparsity_pattern)
        push!(psdvars, psdvar)
        append!(columns, [(1 + Int(i!=j))*q_basis.basis[i]*q_basis.basis[j] for i in 
                    1:q_basis.dim for j in 1:i if !(CartesianIndex(i, j) ∈ q_basis.sparsity_pattern)])
    end

    final_column = zero(Polynomial{true, K})
    for s in 1:n

        @polyvar t
        num, den = get_pade(t, k)
        append!(columns, [inv_dist[s] * x[s]^j - den*t^j for j in 2:2d])

        if K<:AbstractFloat
            pi_s_upper_bound = 1 / sqrt(inv_dist[s])
        else
            pi_s_upper_bound = rat_upper_bound_sqrt(inv(inv_dist[s]))
            @assert pi_s_upper_bound^2 * inv_dist[s] >= 1 
        end
        # sum of squares certificate that p_s(t) dominates 2(1+t)^2*log(1+t)
        # on the interval [-1, -1 + pi_s_upper_bound].
        if k%2 == 0
            q_data = [(1, t.^(1:d+div(k, 2))), ((t+1)*(pi_s_upper_bound-1-t), t.^(1:d+div(k,2)-1))]
        else
            q_data = [((t+1), t.^(1:d+div(k-1, 2))), ((pi_s_upper_bound-1-t), t.^(1:d+div(k-1, 2)))]
        end
        for (multiplier, q_basis) in q_data
            psdvar = PSDVarStructure(length(q_basis), length(columns))
            push!(psdvars, psdvar)
            append!(columns, [multiplier*(1 + Int(i!=j))*q_basis[i]*q_basis[j] for i in 
                                                                            1:length(q_basis) for j in 1:i])
        end
        final_column += 2t*den - num
    end
    final_column -= 2dot(inv_dist, x)
    push!(columns, final_column)
    A = get_affine_matrix_block(columns)
    
    b = A[:, end]
    A = A[:, 1:end-1]
    sdpdata = SDPData{K}(size(A)[2], A, b, psdvars, 0)
    PadeRelaxation{K}(v_bases, h_basis, dir_form, inv_dist, sdpdata, nothing, x)
end

"""
    lin_reduce!(relaxation::LSRelaxation)

Make rows of the linear constraints in the `sdpdata` field 
of `relaxation` linearly independent by removing a subset.
"""
function lin_reduce!(relaxation::LSRelaxation{K}) where K
    olddata = relaxation.sdpdata
    A = [olddata.A olddata.b]
    # the following check is just for efficiency - either option works
    if size(A)[1] >= size(A)[2]
        linind = get_linind_cols(sparse(A'))
    else
        linind = get_linind_cols(A*A')
    end
    relaxation.sdpdata = SDPData(olddata.varlen, olddata.A[linind, :], olddata.b[linind], olddata.psdvars, olddata.ϵ)
    relaxation
end

"""
    set_epsilon!(relaxation::LSRelaxation, tol::Float64)

Should only be called after [`lin_reduce!`](@ref).

Compute ϵ such that approximate floating point solutions of the perturbed SDP

``
\\begin{aligned}
\\operatorname{min} z_1 \\text{ s.t. } & Az=b, \\\\
                                       & X_i\\succeq ϵI \\; \\forall i\\in M
\\end{aligned}
``

remain positive definite when rounded to exact solutions, and populate 
`relaxation.sdpdata` with it.

See paper for motivation.
"""
function set_epsilon!(relaxation::LSRelaxation, tol::Float64)
    data = relaxation.sdpdata
    A, b = (Float64.(data.A), Float64.(data.b))
    # This should be a conservative estimate
    data.ϵ = sqrt(2*size(A)[1]) / svdvals!(Matrix(A))[end] * tol * (1 + maximum(abs.(b)))
end

function extract_gamma(relaxation::LSRelaxation{K}) where K
    relaxation.solution[1]
end

"""
    extract_mat_vars(relaxation::LSRelaxation)

Return all positive semidefinite variables of a computed solution.
"""
function extract_mat_vars(relaxation::LSRelaxation)
    y = relaxation.solution
    data = relaxation.sdpdata
    @assert length(y) == data.varlen
    vec_of_mats = []
    for psdvar in data.psdvars
        mat = vec2mat(psdvar2vec(y[psdvar.range], psdvar), psdvar)
        push!(vec_of_mats, mat)
    end
    vec_of_mats
end

"""
    extract_certificate_data(relaxation::LSRelaxation)

Returns γ (the reciprocal of the computed bound on the log-Sobolev constant)
as well as the polynomials and Gram matrices required to verify the 
sum-of-squares proof.
"""
function extract_certificate_data(relaxation::TaylorRelaxation)
    γ = relaxation.solution[1]
    x = relaxation.x
    h = polynomial(relaxation.h_basis, relaxation.solution[2:1+length(relaxation.h_basis)])
    psd_vars = extract_mat_vars(relaxation)
    γ, (x, h), psd_vars
end
function extract_certificate_data(relaxation::PadeRelaxation{K}) where K
    γ = relaxation.solution[1]
    x = relaxation.x
    h = dot(relaxation.h_basis, relaxation.solution[2:1+length(relaxation.h_basis)])
    psd_vars = extract_mat_vars(relaxation)
    gram_psd_vars = psd_vars[1:length(relaxation.v_bases)]
    @polyvar t
    p = Polynomial{true, K}[]
    p_psd_vars = Tuple[]
    for i in (1+length(relaxation.v_bases)):2:(length(psd_vars)-1)
        # index of first coefficient of p_i in relaxation.solution
        s = relaxation.sdpdata.psdvars[i-1].range[end] + 1
        # index of last coefficient of p_i in relaxation.solution
        e = relaxation.sdpdata.psdvars[i].range[1] - 1
        push!(p, 2t + dot(relaxation.solution[s:e], t.^(2:(e-s+2))))
        push!(p_psd_vars, (psd_vars[i], psd_vars[i+1]))
    end
    γ, (x, h), gram_psd_vars, (t, p), p_psd_vars
end