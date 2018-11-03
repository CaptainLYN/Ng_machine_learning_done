function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
n=length(theta);
for i=1:m
    J=J-y(i)*log(sigmoid(X(i,:)*theta))-(1-y(i))*log(1-sigmoid(X(i,:)*theta));
    for j=1:n
    grad(j)=grad(j)+(sigmoid(X(i,:)*theta)-y(i))*X(i,j);
    end
end
J=1/m*J;
grad=1/m.*grad;

t=0;
for j=2:n  %惯例上不对theta1惩罚，但是即使惩罚了也影响不大，只是一个常数项
    t=t+theta(j)^2;
    grad(j)=grad(j)+lambda/m*theta(j);
end
t=lambda/(2*m)*t;
J=J+t;





% =============================================================

end
