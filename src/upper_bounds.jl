"""
    refine_minimizer(x_0::Vector{Float64}, dir_form::Polynomial{true}, inv_dist::Vector{<:Number})

Given an initial point of, try to find a better local minimimum of 
``\\frac{\\mathcal{E}(x,x)}{\\mathcal{L}(x)}``(using a BFGS algorithm). 
"""
function refine_minimizer(x_0::Vector{Float64}, dir_form::Polynomial{true}, inv_dist::Vector{<:Number})
    inv_dist = Float64.(inv_dist)
    dir_form = polynomial(dir_form, Float64)
    E(x) = dir_form(x)
    L(x) = dot(inv_dist .* x.^2, 2*log.(abs.(x)) .- log(dot(inv_dist, x.^2)))
    f(x) = E(x) / L(x)
    x_e = Optim.minimizer(Optim.optimize(f, x_0, Optim.Newton(); autodiff = :forward))
    x_e /= sqrt(dot(inv_dist, x_e.^2))
    return (x_e, f(x_e))
end

"""
    get_upper_bound(relaxation::LSRelaxation; repeats=1, rng::Random.AbstractRNG=Random.GLOBAL_RNG)

Locally minimize \\frac{\\mathcal{E}(x,x)}{\\mathcal{L}(x)}``, 
staring from a randomly chosen starting point.

Return the best of `repeats` tries.
"""
function get_upper_bound(relaxation::LSRelaxation; repeats=1, rng::Random.AbstractRNG=Random.GLOBAL_RNG)
    dir_form = relaxation.dir_form
    inv_dist = relaxation.inv_dist
    initial_guesses = rand(rng, length(inv_dist), repeats)
    refined_minimizers = [refine_minimizer(initial_guesses[:, g], dir_form, inv_dist) for g=1:repeats]
    sort!(refined_minimizers, by=m->m[2])
    refined_minimizers[1]
end