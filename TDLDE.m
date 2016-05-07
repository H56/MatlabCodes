function [L, R] = TDLDE(A, labels, n1, n2, k1, t1, k2, t2, times, d)
len = length(A);
[n, m] = size(A{1});
X = zeros(n * m, len);
for i = 1 : len
    X(:, i) = reshape(A{i}, [], 1);
end
[W1, W2] = LDECompute(X, labels, k1, t1, k2, t2);

L = rand(n1, n);
for i = 1 : times
    L1 = zeros(m, m);
    L2 = L1;
    for i = 1 : m
        for i = 1 : m
            if W1(i, j) ~= 0
                L1 = L1 + W1(i, j) * (A{i} - A{j})' * L * L' * (A{i} - A{j});
            end
            if W2(i, j) ~= 0
                L2 = L2 + W2(i, j) * (A{i} - A{j})' * L * L' * (A{i} - A{j});
            end
        end
    end
    [R, D] = eigs(L1^(-1) * L2, n1);
    
    L1 = zeros(n, n);
    L2 = L1;
    for i = 1 : m
        for i = 1 : m
            if W1(i, j) ~= 0
                L1 = L1 + W1(i, j) * (A{i} - A{j}) * L * L' * (A{i} - A{j})';
            end
            if W2(i, j) ~= 0
                L2 = L2 + W2(i, j) * (A{i} - A{j}) * L * L' * (A{i} - A{j})';
            end
        end
    end
    [L, D] = eigs(L1^(-1) * L2, n2);
end