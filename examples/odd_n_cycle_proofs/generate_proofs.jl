using LogSobolevRelaxations
include(joinpath(@__DIR__, "cycle_basis.jl"))
using .OddCycleUtils
using DynamicPolynomials, MosekTools, SparseArrays

##########################################################################################
# Define optimizer factory
FEAS_TOL = 1e-12
function mosFactory()
    return Mosek.Optimizer(QUIET = true, INTPNT_CO_TOL_PFEAS=FEAS_TOL)
end
function block_diagonal(X::Vector{Matrix{T}}) where T
    Matrix(blockdiag(sparse.(X)...))
end

##########################################################################################
#           Some functions for processing strings for output
function strip_array_type(str::String)
    i = findfirst("[", str)
    str[i[1]:end]
end
function replace_division_operator(str::String)
    replace(str,  r"\/\/" => "/")
end
function to_zero_based_index(str::String)
    replace(str,  r"x\[[0-9]*\]" => (c) -> "x[" * string(parse(Int, c[3:end-1])-1)*"]")
end
##########################################################################################

##########################################################################################
#           Solve relaxation and print SoS proof in desired Sage format
for n_str in ARGS
    n = parse(Int64, n_str)
    println("Starting n = ", n, "..."); flush(stdout)
    @polyvar x[1:n] 
    kernel = [(abs(mod(i-j+1,n) - 1) == 1)*(1//2) for i in 1:n, j in 1:n]
    dir_form, inv_dist, λ = from_transition_matrix(kernel, x)
    v_bases = get_cycle_basis(x)
    h_basis = monomials(x, 0:4, mon -> length(effective_variables(mon))<=1)
    relaxation = TaylorRelaxation(dir_form, inv_dist, v_bases, h_basis)
    lin_reduce!(relaxation)
    satisfy_sdp!(relaxation, mosFactory)
    round_sol!(relaxation)
    
    γ, h, Q = extract_certificate_data(relaxation)
    if  !all(check_pos_def(X) for X in Q)
        println("Aborting n = ", n)
        continue
    end
    ldl_decomp = ldl.(Q)
    
    D = vcat(map(DL -> DL[1], ldl_decomp)...)
    L = block_diagonal(map(DL -> DL[2], ldl_decomp))
    L = [L[:, i] for i in 1:size(L)[2]];
    b = vcat(map(gb -> gb.basis, v_bases)...)
    
    proofdir = joinpath(@__DIR__, "proofs")
    !isdir(proofdir) && mkdir(proofdir)
    filepath = joinpath(proofdir, string("sos_proof_", n, ".sage"))
    open(filepath, "w") do f
        
        D_str = replace_division_operator(strip_array_type(string(D)))
        println(f, "D = ", D_str)
        
        L_str = replace_division_operator(strip_array_type(string(L)))
        println(f, "L = ", L_str)
        
        b_arr = string.(b)
        b_arr = replace_division_operator.(b_arr)
        b_arr = to_zero_based_index.(b_arr)
        b_str = "[" * join(b_arr, ",") * "]"
        println(f, "B = ", b_str)
        
        h_str = to_zero_based_index(replace_division_operator(string(h)))
        println(f, "h = " * h_str)

    end
    
    println("Finished n = ", n, "!")
end
