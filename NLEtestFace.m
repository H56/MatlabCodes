clear; clc;
% load the source
% load('at&t_faces.mat');
% data = trainData;
% labels = trainLabels';
% [labels, index] = sort(labels);
% data = data(:, index);
% itemCount = 10;
% [dim, num] = size(data);
load('Yale_64x64.mat');
data = fea';
labels = gnd;
[labels, index] = sort(labels);
data = data(:, index);
itemCount = 10;
[dim, num] = size(data);

% parameters
K = 8;
K1 = 60;
d = 38;
times = 100;
result_NLE = zeros(1, times);
result_LPP = zeros(1, times);
for t = 1 : times
    testIndex = randi(itemCount, 1, ceil(num / 11)) + [0 : 11 : num - 1];
    test = data(:, testIndex);
    testLabels = labels(testIndex);
    train = data;
    train(:, testIndex) = [];
    trainLabels = labels;
    trainLabels(testIndex) = [];
    % ---------------------
    % functions
    % ---------------------

    % NLE(X,      K, d,  t,   K2, la, d_pca)
    [Y, P] = NLE(train, K, d, 1, K1, 1, 41);
    % Y = LE(all_data, K, d, 4.9 * 10^2);
    % Y = LLE(all_data, K, d);
    % P = LPP(train, K, d, inf);
    % P = LPP(train, K, d, 2e+7, 40);
    % [Y, P] = PCA(train, 20);
    % train = Y(:, 1 : 2000);
    % test = Y(:, 2001 : end);

    ret = predict(fitcknn(train' * P, trainLabels), test' * P);
    % ret = knnclassify(test' * P, train' * P, trainLabel);
    % ret = predict(fitcknn(train', trainLabel), test');
    result = sum(ret == testLabels) / numel(testLabels) * 100;
    disp(['NPE: ' num2str(result) '%, ' num2str(t) 'th test']);
    result_NLE(t) = result;
    

    P = LPP(train, K, d, inf, 41);
    ret = predict(fitcknn(train' * P, trainLabels), test' * P);
    result = sum(ret == testLabels) / numel(testLabels) * 100;
    disp(['LPP: ' num2str(result) '%, ' num2str(t) 'th test']);
    result_LPP(t) = result;
end
disp(['NLE worst accuracy: ' num2str(min(result_NLE)) '%']);
disp(['NLE Average accuracy: ' num2str(mean(result_NLE)) '%']);
disp(['LPP worst accuracy: ' num2str(min(result_LPP)) '%']);
disp(['LPP Average accuracy: ' num2str(mean(result_LPP)) '%']);
