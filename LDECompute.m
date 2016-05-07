function [W1, W2] = LDECompute(X, labels, k1, t1, k2, t2)
[n, m] = size(X);
[IDX, distance] = KNNIndex(X, m - 1);

neighbors1 = zeros(k1, m);
neighbors2 = zeros(k2, m);
for i = 1 : m
    ni1 = 1;
    ni2 = 1;
    for j = 1 : m - 1
        if labels(i) == labels(IDX(j))
            if ni1 <= k1
                neighbors1(ni1, i) = IDX(j);
                ni1 = ni1 + 1;
            end
        else
            if ni2 <= k2
                neighbors2(ni2, i) = IDX(j);
                ni2 = ni2 + 1;
            end
        end
        if ni1 > k1 && ni2 > k2
            break;
        end          
    end
end

W1 = sparse(m, m);
for i = 1 : m
    W1(i, neighbors1(:, i)) = exp(-distance(i, neighbors1(:, i)) / t1);
    W1(neighbors1(:, i), i) = W1(i, neighbors1(:, i));
end

W2 = sparse(m, m);
for i = 1 : m
    W2(i, neighbors2(:, i)) = exp(-distance(i, neighbors2(:, i)) / t2);
    W2(neighbors2(:, i), i) = W2(i, neighbors2(:, i));
end
