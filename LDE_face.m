load('Yale_64x64.mat');
data = fea';
labels = gnd;
[labels, index] = sort(labels);
data = data(:, index);
itemCount = 10;
[dim, num] = size(data);
times = 10;

for t = 1 : times
    testIndex = randi(itemCount, 1, ceil(num / (itemCount + 1))) + [0 : itemCount + 1 : num - 1];
    test = data(:, testIndex);
    testLabels = labels(testIndex);
    train = data;
    train(:, testIndex) = [];
    trainLabels = labels;
    trainLabels(testIndex) = [];
%     trainIndex = randi(itemCount - 1, 1, ceil(size(train, 2) / itemCount)) + [0 : itemCount : size(train, 2) - 1];
%     head = train(:, trainIndex);
%     train(:, trainIndex) = [];
%     train = [head, train];
%     trainLabels = trainLabels(trainIndex);
    % ---------------------
    % functions
    % ---------------------
    train = bsxfun(@minus, train, mean(train, 1));
    test = bsxfun(@minus, test, mean(test, 1));


    V = LDE(train, trainLabels, 7, inf, 4, inf, 500);
    ret = predict(fitcknn((V' * train)', trainLabels), (V' * test)');
    result = sum(ret == testLabels) / length(testLabels) * 100;
    disp(['LDE: ' num2str(result) '%']);
    
    V = LDE(train, trainLabels, 7, inf, 4, inf, 500);
    P = LPP(train, 7, 100, inf);
    ret = predict(fitcknn((V' * train)', trainLabels), (V' * test)');
    result = sum(ret == testLabels) / length(testLabels) * 100;
    disp(['LDE: ' num2str(result) '%']);
    
%     ret = predict(fitcknn(train', trainLabels), test');
%     result = sum(ret == testLabels) / numel(testLabels) * 100;
%     disp(['NPE: ' num2str(result) '%, ' num2str(t) 'th test']);
%     result_NLE(t) = result;
    

%     train = train.^3;
%     test = test.^3;
%     ret = predict(fitcknn(train', trainLabels), test');
%     result = sum(ret == testLabels) / numel(testLabels) * 100;
%     disp(['LPP: ' num2str(result) '%, ' num2str(t) 'th test']);
%     result_LPP(t) = result;
end
disp(['NLE worst accuracy: ' num2str(min(result_NLE)) '%']);
disp(['NLE Average accuracy: ' num2str(mean(result_NLE)) '%']);
disp(['LPP worst accuracy: ' num2str(min(result_LPP)) '%']);
disp(['LPP Average accuracy: ' num2str(mean(result_LPP)) '%']);