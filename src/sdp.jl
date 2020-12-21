const MOI = MathOptInterface

"""
    solve_sdp(relaxation::LSRelaxation, optimizer; sense=:Min)

Solve the SDP

``
\\begin{aligned}
\\operatorname{min} z_1 \\text{ s.t. } & Az=b, \\\\
                                       & X_i\\succeq ϵI \\; \\forall i\\in M
\\end{aligned}
``

and populate ``relaxation.solution`` with the result.

For technical reaons, the ``ϵI`` term is absorbed into the linear constraint.     
"""
function solve_sdp!(relaxation::LSRelaxation, optimizer; sense=:Min)
    data = relaxation.sdpdata
    A, b, ϵ = (Float64.(data.A), Float64.(data.b), data.ϵ)
    model = Model(optimizer)
    @variable(model, var[1:data.varlen])
    # eye will be a vector with ones at every index corresponding to a
    # diagonal element of a psd matrix, and zeros elsewhere.
    # Then the constraints can be written 
    # A(z - ϵ*eye) = b - ϵ*A*eye, (X_i - ϵI) > 0
    # and solved in terms of (z - ϵ*eye).
    eye = zeros(data.varlen)
    cpsd = []
    for psdvar in data.psdvars
        side_len = psdvar.side_length
        triangle_size = div(side_len*(side_len+1), 2)
        for i in 1:psdvar.side_length
            eye_vec = mat2vec(diagm(ones(side_len)), psdvar)
            eye[psdvar.range] = vec2psdvar(eye_vec, psdvar)
        end
        if length(psdvar.nonzeros) == 1
            push!(cpsd, @constraint(model, var[psdvar.range[1]] >= 0))
        elseif length(psdvar.nonzeros) == triangle_size
            triangle = var[psdvar.range]
            push!(cpsd, @constraint(model, triangle ∈ MOI.PositiveSemidefiniteConeTriangle(side_len)))
        else
            triangle = @variable(model, [i=1:triangle_size])
            @constraint(model, triangle .== 0)
            triangle[psdvar.nonzeros] = var[psdvar.range]
            push!(cpsd, @constraint(model, triangle ∈ MOI.PositiveSemidefiniteConeTriangle(side_len)))
        end
    end
    b_ϵ = b - ϵ*A*eye
    clin = @constraint(model, A*var .== b_ϵ)
    if sense == :Min
        @objective(model, Min, var[1])
    elseif sense == :Satisfy
        @objective(model, Min, 0)
    else
        error("If sense is set, it must be one of :Min or :Satisfy")
    end
    optimize!(model)
    @assert Int(primal_status(model)) == 1 "Error: failed to find SDP solution"
    y = value.(var)
    y += ϵ*eye
    relaxation.solution = y
    relaxation
end

"""
satisfy_sdp(relaxation::LSRelaxation, optimizer::Function)

Solve the relevant semidefinite feasibility problem and populate the `solution`
field of `relaxation`.
"""
function satisfy_sdp!(relaxation::LSRelaxation, optimizer::Function)
    solve_sdp!(relaxation::LSRelaxation, optimizer; sense=:Satisfy)
end