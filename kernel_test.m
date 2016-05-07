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

ret = predict(fitcknn(train' * P, trainLabels), test' * P);
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



    ret = predict(fitcknn(train', trainLabels), test');
    result = sum(ret == testLabels) / numel(testLabels) * 100;
    disp(['NPE: ' num2str(result) '%, ' num2str(t) 'th test']);
    result_NLE(t) = result;
    

    train = train.^3;
    test = test.^3;
    ret = predict(fitcknn(train', trainLabels), test');
    result = sum(ret == testLabels) / numel(testLabels) * 100;
    disp(['LPP: ' num2str(result) '%, ' num2str(t) 'th test']);
    result_LPP(t) = result;
end
disp(['NLE worst accuracy: ' num2str(min(result_NLE)) '%']);
disp(['NLE Average accuracy: ' num2str(mean(result_NLE)) '%']);
disp(['LPP worst accuracy: ' num2str(min(result_LPP)) '%']);
disp(['LPP Average accuracy: ' num2str(mean(result_LPP)) '%']);