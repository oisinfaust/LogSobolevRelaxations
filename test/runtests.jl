using LogSobolevRelaxations
using Test, DynamicPolynomials, MosekTools, SDPAFamily

FEAS_TOL = 1e-12
FEAS_TOL_SDPA = 1e-9

function mosFactory()
    return Mosek.Optimizer(QUIET = true, INTPNT_CO_TOL_PFEAS=FEAS_TOL)
end

function SDPA_GMPFactory()
    return SDPAFamily.Optimizer{Float64}(
        presolve = true,
        params = ( epsilonDash = FEAS_TOL_SDPA, epsilonStar = FEAS_TOL_SDPA)
        )
end

function K_rat(n)
    kernel = [(i!=j)*(1//(n-1)) for i in 1:n, j in 1:n]
    relaxation = default_relaxation(kernel)
    solve_relaxation!(relaxation, mosFactory; tol=FEAS_TOL)
    γ, (x, h), gram_psd_vars, (t, p), p_psd_vars = extract_certificate_data(relaxation)
    Q = gram_psd_vars[1]

    @assert check_pos_def(Q)
    @assert all(check_pos_def(gram1) && check_pos_def(gram2) for (gram1, gram2) in p_psd_vars)

    dir_form = relaxation.dir_form
    v_basis = relaxation.v_bases[1].basis
    u = rat_upper_bound_sqrt(n//1)

    @assert γ*dir_form - (1//n)*sum(subs(p[i], t=>x[i]) for i=1:n) + (1//n)*sum(x.^2+2x)*h ==    
                        v_basis'*Q*v_basis    

    num, den = get_pade(t, 5)
    for i in 1:n
        
        @assert den*p[i] - num == (t+1)*t^2*polynomial(p_psd_vars[i][1], t.^(0:4)) + 
                        (u-1-t)*t^2*polynomial(p_psd_vars[i][2], t.^(0:4))
    end
    inv(γ)
end

function K_flt(n) 
    kernel = [(i!=j)*(1/(n-1)) for i in 1:n, j in 1:n]
    relaxation = default_relaxation(kernel)
    solve_relaxation!(relaxation, mosFactory; tol=FEAS_TOL)

    γ = extract_gamma(relaxation)
    x_min, ub = get_upper_bound(relaxation; repeats=10)
    [inv(γ), ub]
end

function stick(p)
    kernel = [0 1 0; p 0 1-p; 0 1 0]
    relaxation = default_relaxation(kernel)
    lb = solve_relaxation!(relaxation, SDPA_GMPFactory; tol=0.0)
    x_min, ub = get_upper_bound(relaxation, repeats=10)
    [lb, ub]
end

α = (5-2)/(5-1)/log(5-1)
@testset "K_5" begin
    @testset "Rational" begin
        @test abs(α - K_rat(5)::Rational{BigInt}) < 0.0001
    end
@testset "Float" begin
        @test all(abs.(α .- K_flt(5)::Vector{Float64}) .< 0.0001)
    end
end
α = 0.3149
@testset "3-Stick" begin
    @testset "Float" begin
        @test all(abs.(stick(0.1)::Vector{Float64} .- α) .< 0.001)
    end
end


include("../examples/odd_n_cycle_proofs/cycle_basis.jl")
using .OddCycleUtils

function C(n)
    @polyvar x[1:n] 
    kernel = [(abs(mod(i-j+1,n) - 1) == 1)*(1//2) for i in 1:n, j in 1:n]
    dir_form, inv_dist, λ = from_transition_matrix(kernel, x)
    v_bases = get_cycle_basis(x)
    h_basis = monomials(x, 0:4, mon -> length(effective_variables(mon))<=1)
    relaxation = TaylorRelaxation(dir_form, inv_dist, v_bases, h_basis)
    lin_reduce!(relaxation)
    satisfy_sdp!(relaxation, mosFactory)
    round_sol!(relaxation)
    γ, (x, h), Q = extract_certificate_data(relaxation)
    p(t) = 2t + 3t^2 + (2//3)t^3  - (1//6)t^4 + (1//15)t^5
    @assert γ*dir_form - (1//n)*sum(p.(x)) + (1//n)*sum(x.^2+2x)*h == 
                sum(polynomial(Q[i], v_bases[i].basis) for i in 1:length(Q))
    inv(γ)
end

function C_flt(n)
    @polyvar x[1:n] 
    kernel = [(abs(mod(i-j+1,n) - 1) == 1)*(1/2) for i in 1:n, j in 1:n]
    dir_form, inv_dist, λ = from_transition_matrix(kernel, x)
    v_bases = get_cycle_basis(x)
    h_basis = monomials(x, 0:4, mon -> length(effective_variables(mon))<=1)
    relaxation = TaylorRelaxation(dir_form, inv_dist, v_bases, h_basis)
    lin_reduce!(relaxation)
    satisfy_sdp!(relaxation, mosFactory)
    round_sol!(relaxation)
    γ = extract_gamma(relaxation)
    inv(γ)
end

α = sinpi(1/5)^2
@testset "C_5" begin
    @testset "Exact" begin
        @test abs(C(5)::RealCyclotomicFieldElem{5} - α) < 1e-5
    end
    @testset "Float" begin
        @test abs(C_flt(5)::Float64 - α) < 1e-5
    end
end