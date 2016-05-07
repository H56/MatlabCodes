function A = KLDE(X, labels, K, k1, t1, k2, t2, d)
[W1, W2] = LDECompute(X, labels, k1, t1, k2, t2);
D1 = spdiags(sum(W1, 2),0,speye(size(W1,1)));
D2 = spdiags(sum(W2, 2),0,speye(size(W2,1)));
tmp = K * (D1 - W1) * K;
if ~isempty(find(sum(abs(tmp)) == 0))
    tmp = tmp + 10^(-8) * eye(size(tmp));    
end
[A, D] = eigs( (tmp)^(-1) * (K * (D2 - W2) * K), d);
A = real(A);
