clear; close all; clc;
d_pca = 20;
d = 16;
data = load('Yale_64x64.mat');
X = data.fea';
label = data.gnd;
[label, index] = sort(label);
X = X(:, index);
L = [];
pre = [];
for i = 1 : length(label)
    if isempty(pre)
        pre = i;
    end
    if label(pre) ~= label(i)
        L = [L; [pre, i - 1]];
        pre = i;
    end
end
L = [L; [pre, i]];
%[Y, F] = fisher(X, L, d_pca, d);
% F = FLPP(X, L, 20, 16, inf, inf);
% F = LPP(X, 20, 16, inf);
% F = FNPE(X, L, 20, 16);
% F = NPE(X, 20, 16);
F = kernel_fisher(X, L, 160, 16);
figure;
m = ceil(sqrt(d));
for i = 1 : d
    subplot(m, m, i);
    imshow(mat2gray(reshape(F(:, i), [], 64)));
end
disp('');