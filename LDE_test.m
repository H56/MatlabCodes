clear; close all; clc;

load('mnist.mat');
len_train = 2000;
len_test = 1000;

train = data_train(:, 1 : len_train);
train_l = labels_train(1 : len_train);

test = data_test(:, 1 : len_test);
test_l = labels_test(1 : len_test);

% V = LDE(train, train_l, 13, inf, 4, inf, 15);
% ret = predict(fitcknn((V' * train)', train_l), (V' * test)');
% result = sum(ret == test_l) / len_test * 100;
% disp(['LDE: ' num2str(result) '%']);

train = bsxfun(@minus, train, mean(train, 2));
test = bsxfun(@minus, test, mean(test, 2));
K = gaussian_kernel(train, train, 1);
A = KLDE(train, train_l, K, 10, inf, 5, inf, 8);
KT = gaussian_kernel(train, test, 1);
ret = predict(fitcknn((A' * K)', train_l), (A' * KT)');
result = sum(ret == test_l) / len_test * 100;
disp(['LDE: ' num2str(result) '%']);