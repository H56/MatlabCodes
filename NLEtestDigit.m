% load the source
clear; clc;
K = 10;
K1 = 60;
d = 38;

source = load('2k2k.mat');
train = source.fea(source.trainIdx, :)';
trainLabel = source.gnd(source.trainIdx);
test = source.fea(source.testIdx, :)';
testLabel = source.gnd(source.testIdx);

all_data = [train, test];
all_labels = [trainLabel, testLabel];

for K = 1 : 1 : 50
%        NLE(X,      K, d,  t,   K2, la, d_pca)
% [Y, P] = NLE(train, K, d, 1, K1, 1, 41);
% Y = LE(all_data, K, d, 4.9 * 10^2);
% Y = LLE(all_data, K, d);
% P = LPP(train, K, d, inf);
P = LPP(train, K, d, 2e+7, 43);
% [Y, P] = PCA(train, 20);
% train = Y(:, 1 : 2000);
% test = Y(:, 2001 : end);

ret = predict(fitcknn(train' * P, trainLabel), test' * P);
% ret = knnclassify(test' * P, train' * P, trainLabel);
% ret = predict(fitcknn(train', trainLabel), test');
disp(['Accuracy: ' num2str(sum(ret == testLabel) / numel(testLabel) * 100) '%, when K =' num2str(K)]);
end