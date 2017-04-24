% Perform the Householder algorithm on the passed in symmetric
% matrix (if the matrix, A, is not symmetric the function returns in
% failures)
% Returns the tridiagonal matrix which has the same eigenvalues as the
% matrix A
function [A, eigenvector_info] = egaebel_householder_method(A)

    if size(A, 1) ~= size(A, 2)
        fprintf('Must pass symmetric matrix! Returning.....\n')
        return
    end
    
    N = size(A, 1);
    eigenvector_info = eye(N);
    I = eye(N);
    for k = 1:(N - 2)

        % Sum the squares of the column, k
        col_sum = 0;
        for j = (k + 1):N
            col_sum = col_sum + A(j, k)^2;
        end
        
        % Compute alpha
        alpha = sign(A(k + 1, k)) * sqrt(col_sum);
        
        % Compute w
        w = zeros(N, 1);
        w(k + 1, 1) = (A(k + 1, k) + alpha);
        for j = (k + 2):N
            w(j, 1) = A(j, k);
        end
        
        % Compute P and new A
        if (transpose(w) * w) == 0
            P = I;
        else
            P = I - (2 * w * transpose(w) / (transpose(w) * w));
        end
        
        A = P * A * P;
        
        % Compute eigenvector_info to return
        eigenvector_info = eigenvector_info * P;
    end
end