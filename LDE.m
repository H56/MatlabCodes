function V = LDE(X, labels, k1, t1, k2, t2, d)
% G, G' ==> [W, W']
[n, m] = size(X);
[W1, W2] = LDECompute(X, labels, k1, t1, k2, t2);
D1 = spdiags(sum(W1, 2),0,speye(size(W1,1)));
D2 = spdiags(sum(W2, 2),0,speye(size(W2,1)));
tmp = X * (D1 - W1) * X';
if m < n
    tmp = tmp + 10^(-9) * eye(size(tmp));    
end
[V, D] = eigs( (tmp)^(-1) * (X * (D2 - W2) * X' ), d);
V = real(V);
