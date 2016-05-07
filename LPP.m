function A = LPP(X, K, d, t)%, d_pca)
[D, N] = size(X);
Delta = 0;
if(K > D) 
  fprintf(1,'   [note: K>D; regularization will be used]\n'); 
  Delta = 1e-1; % regularlizer in case constrained fits are ill conditioned
end
%STEP1: compute the neighbors
X2 = sum(X .^ 2, 1);
distance = sqrt(bsxfun(@plus, X2, bsxfun(@minus, X2', 2 * (X' * X))));
[~, index] = sort(distance, 1);
neighbors = index(2 : K + 1, :);

W = sparse(N, N);
for i = 1 : N
    W(i, neighbors(:, i)) = exp(-distance(i, neighbors(:, i)) / t);
    W(neighbors(:, i), i) = exp(-distance(i, neighbors(:, i)) / t);
end

D = spdiags(sum(W, 2),0,speye(size(W,1)));
L = D - W;
% Wpca = PCA(X, d_pca);
% D = Wpca' * X * D * X' * Wpca;
% L = Wpca' * X * L * X' * Wpca;
if N < D
    D = D + + 10^(-9) * eye(size(tmp));
end
W = D^-1 * L;
[A, dump] = eig(W);
[dump, index] = sort(abs(diag(dump)));
A = A(:, index);
col = find(dump > 0, 1);
A = A(:, col : col + d - 1);
A = real(A);
% A = Wpca * A;


%-----------------------
% old
%-----------------------
% function A = LLP(X, K, d, t)
% [D, N] = size(X);
% Delta = 0;
% if(K > D) 
%   fprintf(1,'   [note: K>D; regularization will be used]\n'); 
%   Delta = 1e-1; % regularlizer in case constrained fits are ill conditioned
% end
% %STEP1: compute the neighbors
% X2 = sum(X .^ 2, 1);
% distance = sqrt(bsxfun(@plus, X2, bsxfun(@minus, X2', 2 * (X' * X))));
% [~, index] = sort(distance, 1);
% neighbors = index(2 : K + 1, :);
% 
% W = sparse(N, N);
% for i = 1 : N
%     W(i, neighbors(:, i)) = exp(-distance(i, neighbors(:, i)) / t);
%     W(neighbors(:, i), i) = exp(-distance(i, neighbors(:, i)) / t);
% end
% 
% D = spdiags(sum(W, 2),0,speye(size(W,1)));
% L = D - W;
% D = X * D * X';
% L = X * L * X';
% [U, S] = eig(D);
% for i = 1 : length(S)
%     if S(i, i) > 0
%         S(i, i) = S(i, i)^(-1/2);
%     else 
%         S(i, i) = 0;
%     end
% end        
% W = S * U' * L * U * S;
% [A, dump] = eig(W);
% [dump, index] = sort(real(diag(dump)));
% A = A(:, index);
% col = find(dump > 0, 1);
% A = A(:, col : col + d - 1);
% A = U * S * A;
% A = real(A);
