function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%---------
%PART 1 A
%---------
%VARIABLE INITIALIZATION

% Add the bias constant.
X = [ones(length(X),1) X];

%Initialize Y
temp_y = zeros(m,num_labels);

%Convert Y to truth vectors (m X num_labels)
for i=1:m
	temp_y(i,y(i)) = 1;
end


%IMPLEMENTATION OF COST FUNCTION
%Get activation function of the output layer (m X num_labels)
temp_a = sigmoid([ones(length(sigmoid(X * Theta1')),1) sigmoid(X * Theta1')] * Theta2');

%Calculate the cost function.
temp_J = (((-1 * log(temp_a)) .* temp_y) - (log(1-temp_a) .* (1-temp_y)))/m;
J = ones(1, size(temp_J(:))) * temp_J(:);


%--------
%PART 1 B
%--------
%REGULARIZATION

Theta1_tmp = Theta1(:,2:end);
Theta2_tmp = Theta2(:,2:end);

Theta1_sq = Theta1_tmp(:)' * Theta1_tmp(:);
Theta2_sq = Theta2_tmp(:)' * Theta2_tmp(:);

reg = ( ones(1, size(Theta1_sq(:))) * Theta1_sq(:) + ones(1, size(Theta2_sq(:))) * Theta2_sq(:) ) * lambda / (2*m);

J = J + reg;

%--------
%PART 2 A
%--------
%UNREGULARIZED BACKPROPAGATION
%Step 1 - Gradient of the final layer wrt activation function
deltaFinal = temp_a - temp_y;

%Step 2 - Gradient of the hidden layer wrt activation function
%deltaHidden = ((deltaFinal * Theta2) .* sigmoidGradient([ones(length(sigmoid(X * Theta1')),1) sigmoid(X * Theta1')]))(:,2:end);
deltaHidden = (deltaFinal * Theta2(:,2:end)) .* sigmoidGradient(X * Theta1');

%Step 3 - Activation of the Hidden layer
aHidden = [ones(length(sigmoid(X * Theta1')),1) sigmoid(X * Theta1')];

%Theta2_grad calculation - Accumulation of gradient wrt weights between l2 - l3
[rowaH, colaH] = size(aHidden);
[rowdF, coldF] = size(deltaFinal);
counter = 0;
for j = 1:colaH
	for i = 1:coldF	
		counter = counter + 1;
		if (j == 1)
			Theta2_grad_temp(:,counter) = (deltaFinal(:,i)' * aHidden(:,j))/m;
		else %PART 2B
			Theta2_grad_temp(:,counter) = (deltaFinal(:,i)' * aHidden(:,j) + lambda * Theta2(i,j))/m;
		end		
	end	
end

Theta2_grad = reshape(Theta2_grad_temp(1:end), size(Theta2_grad));


%Theta1_grad calculation - Accumulation of gradient wrt weights between l1 - l2
[rowaH, colaH] = size(X);
[rowdF, coldF] = size(deltaHidden);
counter = 0;

for j = 1:colaH %401 4
	for i = 1:coldF	 %25 5
		counter = counter + 1;
		if (j == 1)
			Theta1_grad_temp(:,counter) = (deltaHidden(:,i)' * X(:,j))/m;
		else %PART 2B
			Theta1_grad_temp(:,counter) = (deltaHidden(:,i)' * X(:,j) + lambda * Theta1(i,j))/m;	
		end
	end	
end

Theta1_grad = reshape(Theta1_grad_temp(1:end), size(Theta1_grad));


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
