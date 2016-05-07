clear; close all; clc;

load('mnist.mat');
len_train = 2000;
len_test = 1000;

train = data_train(:, 1 : len_train);
train_l = labels_train(1 : len_train);

test = data_test(:, 1 : len_test);
test_l = labels_test(1 : len_test);

train = bsxfun(@minus, train, mean(train, 2));
test = bsxfun(@minus, test, mean(test, 2));
A = SDA(train, train_l(1 : 1000), 70, 0.2, 20);
ret = predict(fitcknn((A' * train)', train_l), (A' * test)');
result = sum(ret == test_l) / len_test * 100;
disp(['LDE: ' num2str(result) '%']);
