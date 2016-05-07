function [A] = FNPE(X, Ls, K, d)
% X = bsxfun(@minus, X, mean(X, 2));
[D, N] = size(X);
n = length(X);
c = length(Ls);
if K > min(Ls(:, 2) - Ls(:, 1) + 1)
    K = min(Ls(:, 2) - Ls(:, 1) + 1) - 1;
end
mu_i = zeros(n, c);
SB = zeros(length(X));
for i = 1 : c
    num = Ls(i, 2) - Ls(i, 1) + 1;
    Xc = X(:, Ls(i, 1) : Ls(i, 2));
    % c-th class mean
    mu_i(:, i) = mean(Xc, 2);
    
    X2 = sum(Xc.^2,1);
    distance = repmat(X2, num, 1) + repmat(X2', 1, num) - 2 * Xc' * Xc;

    [~, index] = sort(distance, 1);
    neighborhood = index(2 : K + 1, :);
    if(K>D) 
      fprintf(1,'   [note: K>D; regularization will be used]\n'); 
      tol=1e-3; % regularlizer in case constrained fits are ill conditioned
    else
      tol=0;
    end

    W = zeros(K,num);
    for ii=1:num
       z = Xc(:,neighborhood(:,ii))-repmat(Xc(:,ii),1,K); % shift ith pt to origin
       C = z'*z;                                        % local covariance
       C = C + eye(K,K)*tol*trace(C);                   % regularlization (K>D)
       W(:,ii) = pinv(C) * ones(K,1);                           % solve Cw=1
       W(:,ii) = W(:,ii)/sum(W(:,ii));                  % enforce sum(w)=1
    end;

    M = sparse(1:num,1:num,ones(1,num),num,num,4*K*num); 
    for ii=1:num
       w = W(:,ii);
       jj = neighborhood(:,ii);
       M(ii,jj) = M(ii,jj) - w';
       M(jj,ii) = M(jj,ii) - w;
       M(jj,jj) = M(jj,jj) + w * w';
    end;
    SB = SB + Xc * M * Xc';
end

mu2 = sum(mu_i .^ 2, 1);
mean_d = sqrt(abs(bsxfun(@plus, mu2, mu2') -2 * (mu_i' * mu_i)));
Wmean = zeros(c);
for i = 1 : c
    Wmean(i, :) = 1;
    Wmean(i, i) = 0;
end
Dmean = spdiags(sum(Wmean, 2),0,speye(size(Wmean,1)));
Lmean = Dmean - Wmean;
SW = mu_i * Lmean * mu_i';

[~, P] = PCA(X, 20);
[A, dump] = eig(pinv(P * SW * P') * (P * SB * P'));

% [A, dump] = eig(pinv(SW) * SB);
[dump, index] = sort(diag(abs(dump)));
A = A(:, index);
col = find(dump > 0, 1);
F = A(:, col : col + d - 1);
A = P' * F;

