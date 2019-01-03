function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
%p = zeros(size(X, 1), 1);%这里注释掉，避免重复分配

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
X=[ones(m,1) X];
z1=X*Theta1';%m*(n+1) X (n+1)*25 = m*25
%一开始忘了加sigmoid激活函数
z1=sigmoid(z1);
z1=[ones(m,1) z1];
z2=z1*Theta2';%m*26 X 26*10 = m*10
z2=sigmoid(z2);

[~,p]=max(z2,[],2);


% =========================================================================


end
