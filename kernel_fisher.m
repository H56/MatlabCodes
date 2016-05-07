function [Y F] = kernel_fisher(X, L, K, Dpca, Dfisher)
% X: columns of data
% L: labels, tow dimension, [start, end]
% d_pac: PCA wanted dimension
% d: wanted dimension

% compute means
% K = gaussian_kernel(X, X, 10^7);
% [Apca, D] = eigs(K, [], Dpca);
[Apca, D] = eig(K);
[D, index] = sort(diag(D), 'descend');
Apca = Apca(:, index);
Apca = Apca(:, 1 : Dpca);

[~, n] = size(X);
c = length(L);
Z = zeros(n, c);
for i = 1 : c
    Z(L(i, 1) : L(i, 2), i) = sqrt(L(i, 2) - L(i, 1) + 1);
end
Z = Z * Z';
SB = Apca'* K * Z * K' * Apca;
SW = Apca' * K * K * Apca;
% [A, D] = eigs((Apca' * K * K * Apca)^(-1) * Apca' * K * Z * K * Apca, [], Dfisher);
% [Wf, D] = eigs(SB, SW, Dfisher);
[Wf, D] = eig(SW^-1 * SB);
[D, index] = sort(diag(D), 'descend');
Wf = Wf(:, index);
Wf = Wf(:, 1 : Dfisher);
Wf = real(Wf);
F = Apca * Wf;
Y = F' * K;