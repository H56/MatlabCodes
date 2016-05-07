clear; close all; clc;
d_pca = 20;
d = 40;
result = 0;
count = 400;
% data = load('Yale_64x64.mat');
% data = load('mnist.mat');
load('at&t faces.mat');
% X = data.data_train(:, 1 : 5000);
% label = data.labels_train(1 : 5000);
[trainLabels, index] = sort(trainLabels);
trainData = trainData(:, index);
X = trainData;
label = trainLabels';
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

L1 = L - [(0 : length(L) - 1)', (1 : length(L))'];
D = [20 40];
fuck = [];
for j = [2 * 10^-7, 2 * 10^-8]
    for d = D
        for i = 1 : count
            index = randi(10, length(L), 1) + [0 : 10 : length(label) - 10]';
            train_data = X;
            train_labels = label;
            test_data = train_data(:, index);
            test_labels = train_labels(index);
            train_data(:, index) = [];
            train_labels(index) = [];

            train_data = bsxfun(@minus, train_data, mean(train_data, 2));
%             K = gaussian_kernel(train_data, train_data, j);
            K = sigmoid(train_data, train_data, j, 0);
%             K = polynomial_kernel(train_data, train_data, j);
%             [Y, F] = kernel_PCA(train_data, K, d);
            [Y, F] = kernel_fisher(train_data, L1, K, 40, d);
            test_data = bsxfun(@minus, test_data, mean(test_data, 2));
%             K = gaussian_kernel(train_data, test_data, j);
%             K = polynomial_kernel(train_data, test_data, j);
            K = sigmoid(train_data, test_data, j, 0);
            F = real(F);
            ret = predict(fitcknn(Y', train_labels), K' * F);
            result = result + sum(test_labels == ret);
            disp([num2str(i) 'th knn test']);
        end
        fuck = [fuck result / (length(ret) * count) * 100];
        disp(result / (length(ret) * count) * 100); 
        result = 0;
    end
end
disp(fuck); 