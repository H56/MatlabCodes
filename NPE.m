function [A] = NPE(X,K,d)
[D,N] = size(X);
X2 = sum(X.^2,1);
distance = repmat(X2,N,1)+repmat(X2',1,N)-2*X'*X;

[sorted,index] = sort(distance);
neighborhood = index(find(sum(sorted, 2) > 0) : (1+K), :);
if(K>D) 
  fprintf(1,'   [note: K>D; regularization will be used]\n'); 
  tol=1e-3; % regularlizer in case constrained fits are ill conditioned
else
  tol=0;
end

W = zeros(K,N);
for ii=1:N
   z = X(:, neighborhood(:,ii)) - repmat(X(:,ii),1,K); % shift ith pt to origin
   C = z'*z;                                        % local covariance
   C = C + eye(K,K)*tol*trace(C);                   % regularlization (K>D)
   W(:,ii) = pinv(C) * ones(K,1);                           % solve Cw=1
   W(:,ii) = W(:,ii)/sum(W(:,ii));                  % enforce sum(w)=1
end;

M = sparse(1:N,1:N,ones(1,N),N,N,4*K*N); 
for ii=1:N
   w = W(:,ii);
   jj = neighborhood(:,ii);
   M(ii,jj) = M(ii,jj) - w';
   M(jj,ii) = M(jj,ii) - w;
   M(jj,jj) = M(jj,jj) + w * w';
end;
[A, D] = eig(X * M * X');
[D, IX] = sort((diag(D)));
A = A(:, IX);
start = find(D > 0, 1);
A = A(:, start : d + start);
fprintf(1,'Done.\n');

