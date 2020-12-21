import sys

for n_str in sys.argv[1:]:
    n = int(n_str)

    K.<theta> = CyclotomicField(n).maximal_totally_real_subfield()[0]
    R = PolynomialRing(K, n, "x")
    x = R.gens()
    load("proofs/sos_proof_{}.sage".format(n))

    assert(all(d in K and d >= 0 for d in D))
    assert(all(b in R for b in B))
    assert(all(all(number in K for number in l) for l in L))
    assert(h in R)

    m = len(D)
    q = [sum(l[i]*B[i] for i in range(m)) for l in L]
    assert(all(poly in R for poly in q))

    sum_of_squares = sum(n*D[i]*q[i]^2 for i in range(m))

    E = sum((x[i%n] - x[i-1])^2 for i in range(1, n+1))
    taylor_upper_bound = sum((2*y + 3*y^2 + (2/3)*y^3 - (1/6)*y^4 + (1/15)*y^5) for y in x)
    translated_sphere_constraint = sum(y^2 + 2*y for y in x)
    if E / (1 - theta/2) - taylor_upper_bound + (translated_sphere_constraint * h) == sum_of_squares:
        print("n = {}: Success - proof is valid".format(n))
    else:
        print("n = {}: Failure - proof is invalid".format(n))


