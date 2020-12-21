# LogSobolevRelaxations

This repository contains code to obtain rigourous and accurate lower bounds on the logarithmic Sobolev constants of finite Markov chains.
It accompanies the paper [TODO].

The project defines an `import`able module named LogSobolevRelaxations.
It also contains some examples of the use of this module.

## Installation

```bash
$ git clone https://github.com/oisinfaust/LogSobolevRelaxations
$ cd LogSobolevRelaxations
$ julia 
```
Next, from the Julia REPL, type `]` to enter the Pkg REPL, then:
```julia
(@v1.5) pkg> add .
```

## Quick example
___
**A note on solvers**

The module `LogSobolevRelaxations` contains no hardcoded reference to a particular solver.
However, many of the examples in this project use the solver [Mosek](https://www.mosek.com). Mosek is a commercial software package, but it is possible to obtain a [free academic licence](https://www.mosek.com/products/academic-licenses) if you meet certain criteria.
In order to use Mosek, you should put this licence in your home directory in the following location:
`~/mosek/mosek.lic`

It is, of course, also possible to use this project with a different semidefinite program solver, as long as it has a [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl) interface.
___

Let's compute an exact rational lower bound on the log-Sobolev constant of the simple random walk on K<sub>5</sub>.

```julia
julia> using LogSobolevRelaxations

# We need a solver, in this case we choose Mosek.
julia> using MosekTools

julia> function mosFactory()
           return Mosek.Optimizer(QUIET = true, INTPNT_CO_TOL_PFEAS=1e-12)
       end
mosFactory (generic function with 1 method)

julia> kernel = [(i!=j)*1//4 for i=1:5, j=1:5]
5×5 Array{Rational{Int64},2}:
 0//1  1//4  1//4  1//4  1//4
 1//4  0//1  1//4  1//4  1//4
 1//4  1//4  0//1  1//4  1//4
 1//4  1//4  1//4  0//1  1//4
 1//4  1//4  1//4  1//4  0//1

# Define a rational relaxation based on Padé approximation, using sensible default options.
julia> relaxation = default_relaxation(kernel)
Padé-type relaxation using number type Rational{BigInt}

# One could instead define a relaxation using floating-point data only with the line:
# 
#   julia> relaxation = default_relaxation(PadeRelaxationSingle, Float64.(kernel))
#
# Floating-point relaxations can be solved much more quickly, but any solution will lack 
# a certificate in exact rational arithmetic.

# Solve the relaxation.
julia> α_lower = solve_relaxation!(relaxation, mosFactory, tol=1e-12)
617053050204257219891059971901133144269618172736176828820813038280371801366560082851974029882343299
512403390850664351909255363717043458815301654338103211787261479299991689487159370276300918891906844
5460813085289765663399278045485149978624//114055624389781723654203992377744468820060742014759136138
394757973307615359318033942572402022975531558218191578456611342114781758626929798735480825178596046
69946649140789795083025389110489972250134970915091561035943538566417159434446839555

# This rational number is hard understand, let's round it.
julia> Float64(α_lower)
0.5410106283715538

# The real constant, as determined analytically (complete graphs are a rare case when this can be 
# done).
julia> (5-2)/(5-1)/log(5-1)
0.5410106403333613
```

## More examples

In the `examples` folder, there are two IJulia notebooks: [petersen.ipynb](examples/petersen.ipynb) and [3-stick.ipynb](examples/3-stick.ipynb).

The folder `examples/odd_n_cycle_proofs` contains material used to prove that the logarithmic Sobolev constant of the n-cycle is exaclty half of its spectral gap for odd n between 5 and 21.
This folder contains its own [README](examples/odd_n_cycle_proofs/README.md).