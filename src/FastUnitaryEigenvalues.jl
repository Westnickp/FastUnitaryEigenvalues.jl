module FastUnitaryEigenvalues

import LinearAlgebra, Distributions

export 
    # Core transformation struct and methods
    CoreTrans, 
    apply_ct!,
    apply_ct,
    apply_full,
    apply_ctchain!,
    apply_ctchain,
    fuse_ct,
    turnover,
    ct_is_diag,

    # Functions for sampling unitary matrices
    get_unitary_naive,
    get_uuh_naive,
    factorize_uuh,
    sample_factored_form,

    # Unitary QR methods
    get_eig_estimate,
    random_shift,
    get_bulge,
    qr_iterate!,
    uuh_qr!,
    uuh_qr

struct CoreTrans
	# 2x2 matrix A plus position k so action is on index, index+1 rows
	active_part
	index
end

function apply_ct!(C, V; k=0)
	if k == 0
		k = C.index
	end
	V[k:k+1, :] = C.active_part*V[k:k+1, :] 
end

function apply_ct(C, V; k=0)
	if k == 0
		k = C.index
	end
	S = C.active_part*V[k:k+1, :]
end

function apply_full(C::CoreTrans, v)
    # Outputs full vector/columns, does not change in place, O(n * numcols) operations 
    s = v[:, 1:end] .+ 0im # copy
    apply_ct!(C, s)
    return s
end

function fuse_ct(C1::CoreTrans, C2::CoreTrans; k=0)
	A = C1.active_part*C2.active_part
	if k == 0
		k = C1.index
	end
	return CoreTrans(A, k)
end

function apply_ctchain!(Cs, V)
	for j in 0:length(Cs)-1
		C = Cs[length(Cs)-j]
		k = C.index
		apply_ct!(C, V)
	end
end

function apply_ctchain(Cs, V)
	S = V[:, :] .+ 0.0im
	for j in 0:length(Cs)-1
		C = Cs[length(Cs)-j]
		k = C.index
		apply_ct!(C, S)
	end
	S
end

function turnover(C1::CoreTrans, C2::CoreTrans, C3::CoreTrans; updown=true)
    # not implemented
    if updown #updownup to downupdown
		if (C1.index != C2.index-1) || (C2.index != C3.index+1)
			error("Transformations not in appropriate setting")
		end
		k = C1.index
		A = [1.0+0.0im 0 0; 0 1 0; 0 0 1]
		apply_ct!(C3, A; k=1); apply_ct!(C2, A; k=2); apply_ct!(C1, A; k=1)
		G1, r1 = LinearAlgebra.givens(A[:, 1], 2, 3)
		tC1 = [G1[2, 2] G1[2, 3]; G1[3, 2] G1[3, 3]]'
		A = G1*A
		G2, r2 = LinearAlgebra.givens(A[:, 1], 1, 2)
		tC2 = [G2[1, 1] G2[1, 2]; G2[2, 1] G2[2, 2]]'
		A = G2*A
		G3, r3 = LinearAlgebra.givens(A[:, 3], 3, 2)
		tC3 = [G3[2, 2] G3[2, 3]; G3[3, 2] G3[3, 3]]'
		A = G3*A
		
		a = A[1, 1]; b = A[2, 2]; c = A[3, 3]
		tC2[:, 1] = a*tC2[:, 1] 
		tC3[:, 1] = b*tC3[:, 1]
		tC3[:, 2] = c*tC3[:, 2]
		Cs = [CoreTrans(tC1, k+1) CoreTrans(tC2, k) CoreTrans(tC3, k+1)]
		return Cs
    else #downupdown to updownup
		if (C1.k != C2.k+1) || (C2.k != C3.k+1)
			error("Transformations not in appropriate setting")
		end
        return 1
    end
end

function ct_is_diag(C)
	A = C.active_part
	eta = sqrt(abs2(A[2, 1]) + abs2(A[1, 2]))
	machine_tol = eps(Float64) # 
	if eta > machine_tol
		return false
	else
		return true
	end
end

function get_unitary_naive(n)
	A = randn(n, n) + im*randn(n, n)
	Q, R = LinearAlgebra.qr(A)
	L = LinearAlgebra.diag(R)
	L = L./abs.(L)
	return (Q*LinearAlgebra.diagm(L))
end

function get_uuh_naive(n)
	A = get_unitary_naive(n)
	F = LinearAlgebra.hessenberg(A)
	return F.H
end

function factorize_uuh(U)
	n = size(U, 1)
	Cs = Array{CoreTrans}(undef, n-1)
	for k in 1:n-1
		G, r = LinearAlgebra.givens(U[k, k], U[k+1, k], k, k+1)
		C = [conj(G[k, k])*r conj(G[k+1, k]); conj(G[k, k+1])*r conj(G[k+1, k+1])]
		Cs[k] = CoreTrans(C, k)
		U = G*U
		U[k, k] = U[k, k]/r
		if k == n-1
			C[1, 2] = C[1, 2]*U[n, n]
			C[2, 2] = C[2, 2]*U[n, n]
			Cs[k] = CoreTrans(C, k)
			U[n, n] = U[n, n]/U[n, n]
		end
	end
	Cs
end

function sample_factored_form(n; force_det=0)
	cs = Array{CoreTrans}(undef, n-1)
	for j in 1:n-1
		alphaj = (randn(1)+1.0im*randn(1))[1]/sqrt(2) # Complex normal with unit var
		betaj = rand(Distributions.Chisq(2*j))/2 # Complex chisq with j deg free
		wjnrm = sqrt(abs2(alphaj)+abs2(betaj))
		thetaj = angle(alphaj)
		vj = [alphaj + exp(1im*thetaj)*wjnrm; betaj]
		Cj = [1 0; 0 1] - 2*vj*vj'/(vj'*vj)
		dj = -exp(1im*thetaj)
		Cj[:, 1] = Cj[:, 1]*dj
		if j == n-1
			thetan = (rand()*2 - 1)*pi
			Cj[:, 2] = Cj[:, 2]*(-exp(1im*thetan))
		end
		cs[j] = CoreTrans(Cj, j)
	end
	cs
end


function get_eig_estimate(ct_chain, right)
	# Projected/unimodular Wilkinson shift strategy
	C1 = ct_chain[right-1]
	C2 = ct_chain[right]
	A = [1+0.0im 0 0; 0 1 0; 0 0 1]
	apply_ct!(C2, A; k=2)
	apply_ct!(C1, A; k=1)
	target = A[3, 3]
	λ = LinearAlgebra.eigvals(A[2:3, 2:3])
	newtarget = λ[1]
	if abs2(λ[1] - target) > abs2(λ[2] - target)
		newtarget = λ[2]
	end
	return newtarget/abs(newtarget)
end

function random_shift()
	z = randn(1) + 1im*randn(1)
	z = z[1]/abs(z[1])
	z
end

function get_bulge(ct_chain, eig_est, left)
	e1 = [1+0.0im; 0; 0]
	apply_ct!(ct_chain[left+1], e1; k=2)
	apply_ct!(ct_chain[left], e1; k=1)
	e1[1] = e1[1] - eig_est
	
	x = e1[1:2]

	# solve for bulge -- want 
	G, r = LinearAlgebra.givens(x, 1, 2)
	B = [G[1, 1]*abs(r)/r G[1, 2]*abs(r)/r; G[2, 1] G[2, 2]]'
	return CoreTrans(B, 1)
end

function qr_iterate!(ct_chain, bulge, left, right)
	# Fuse adjoint of bulge with first core transformation
	left_fuse = CoreTrans(bulge.active_part', left)
	ct_chain[left] = fuse_ct(left_fuse, ct_chain[left])
	# Chase bulge
	for j in left:right-1
		# Turnover
		C1 = ct_chain[j]
		C2 = ct_chain[j+1]
		C3 = bulge
		newseq = turnover(C1, C2, C3)
		ct_chain[j] = newseq[2]
		ct_chain[j+1] = newseq[3]

		# similarity trans moving bulge to other side
		bulge = newseq[1]
	end
	ct_chain[right] = fuse_ct(ct_chain[right], bulge)
end

function uuh_qr!(ct_chain; maxiter=100, verbose=false)
	left = 1
	right = length(ct_chain)
	iter_counter = 0
	deflation_counter = 1
	small_case_size = 4
	
	d = zeros(length(ct_chain)+1) .+ 0.0im
	while right-left > small_case_size-2
		# Check for deflation
		if ct_is_diag(ct_chain[right])
			d[right+1] = ct_chain[right].active_part[2, 2]
			if verbose
				display("Eigenvalue "*string(deflation_counter)*" found!")
				display("Value "*string(d[right+1]))
				display("Iterates required "*string(iter_counter))
			end
			
			A = ct_chain[right-1].active_part
			A[:, 2] = A[:, 2]*ct_chain[right].active_part[1, 1]
			ct_chain[right-1] = CoreTrans(A, ct_chain[right-1].index)
			right = right - 1
			deflation_counter = deflation_counter + 1
			iter_counter = 0
			if right - left <= small_case_size
				break
			end
		end
		iter_counter = iter_counter + 1
	
		# iterate 
		shift = get_eig_estimate(ct_chain, right)
		bulge = get_bulge(ct_chain, shift, left)
		qr_iterate!(ct_chain, bulge, left, right)

		if iter_counter > maxiter
			error("Not converging; number of deflated eigs is "*string(deflation_counter))
		end
	end
	# In small case... 
	if verbose
		display("Finishing up small case size of "*string(small_case_size)*"x"*string(small_case_size)*" matrices")
	end
	A = LinearAlgebra.Matrix(LinearAlgebra.I(right+1)) .+ 0.0im
	apply_ctchain!(ct_chain[left:right], A)
	d[left:right+1] = LinearAlgebra.eigvals(A)
	return d
end

function uuh_qr(ct_chain; maxiter=100, verbose=false)
	ct_chain = ct_chain[1:end]
    d = uuh_qr!(ct_chain; maxiter=maxiter, verbose=verbose)
    return d
end

end
