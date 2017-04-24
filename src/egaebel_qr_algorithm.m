% Perform the QR algorithm and return a vector of eigenvalues along
% with a matrix of eigenvectors where the columns are the eigenvectors
% NOTE: This algorithm has not been tested on non-square matrices and 
% probably won't work for them.
function [eigenvalues, eigenvectors] = ...
                        egaebel_qr_algorithm(A, tolerance, MAX_ITERATIONS)

    N = size(A, 1); 
    eigenvectors = eye(N);
                    
    % Check if the matrix is tridiagonal, if it isn't convert it
    if ~egaebel_check_tridiagonal(A)
        [A, eigenvectors] = egaebel_householder_method(A);
    end

    % Initialize a & b vectors to represent tridiagonal matrix
    a = zeros(N, 1);
    b = zeros(N, 1);
    for i = 1:N
        a(i) = A(i, i);
        % Skip first index, because it makes life easier when indexing
        if i > 1
            b(i) = A(i, (i - 1));
        end
    end

    % Set size to 0 initially, so we can incrementally tack on the end
    %eigenvalues = zeros(0, 1);
    eigenvalues = zeros(N, 1);

    % n is going to be dynamic
    n = N;
    
    % Values after processing each row
    z = zeros(n, 1);
    q = zeros(n, 1);
    r = zeros(n, 1);
    
    % Inner variables used to compute R
    x = zeros(n, 1);
    y = zeros(n, 1);
    
    sigma_vec = zeros(n, 1);
    
    % Rotation matrix elements
    c = zeros(n, 1);
    s = zeros(n, 1);
    
    % Loop & loop vars
    k = 1;
    SHIFT = 0;
    sigma = 0;
    while k <= MAX_ITERATIONS && n > 0
       
        % If we have a 1x1 matrix now, we're done with the eigenvalues! 
        % We need only continue to finish computing the eigenvectors
        if n == 1
            eigenvalues(1, 1) = a(1) + SHIFT;
            break
        end
        
        % temp variables to help solve the polynomial for a 2x2 matrix's 
        % eigenvalues
        trace_2x2 = a(n - 1) + a(n); %-(a(n - 1) + a(n));        
        det_2x2 = a(n) * a(n - 1) - b(n)^2;
        if abs(trace_2x2^2 - (4 * det_2x2)) < 10^-9
            disc_2x2 = 0;
        elseif ~isreal(sqrt(abs(trace_2x2^2 - (4 * det_2x2))))
            disc_2x2 = 0;
        else
            disc_2x2 = sqrt(trace_2x2^2 - (4 * det_2x2));
        end
        
        % Find eigenvalues of 2x2 matrix
        mu_1 = (1 / 2) * (trace_2x2 + disc_2x2);
        mu_2 = (1 / 2) * (trace_2x2 - disc_2x2);

        % Compute shift accumulation (using 2x2 eigenvalues)
        min_diff = abs(mu_1 - a(n));
        sigma = mu_1;
        if abs(mu_2 - a(n)) <= min_diff
            sigma = mu_2;
        end
        % Accumulate shift
        SHIFT = SHIFT + sigma;
        % Perform shift
        d = zeros(n, 1);
        for j = 1:n
            d(j) = a(j) - sigma;
        end
        
        % Compute z, q, r (new values along each row) (values of the R
        % matrix)
        x(1) = d(1);
        y(1) = b(2);
        for j = 2:n
            
            z(j - 1) = sqrt(x(j - 1)^2 + b(j)^2);
            if z(j - 1) == 0
                % Compute rotation matrix elements
                s(j) = 0.7071;
                c(j) = 0.7071;

                sigma_vec(j) = 0.7071;
            else
                % Compute rotation matrix elements
                s(j) = b(j) / z(j - 1);
                c(j) = x(j - 1) / z(j - 1);

                sigma_vec(j) = b(j) / z(j - 1);
            end
            
            
            % Compute new row values
            q(j - 1) = c(j) * y(j - 1) + s(j) * d(j);
            x(j) = -sigma_vec(j) * y(j - 1) + c(j) * d(j);
            if j ~= n
                r(j - 1) = sigma_vec(j) * b(j + 1);
                y(j) = c(j) * b(j + 1);
            end
        end
        z(n) = x(n);
        
        % Compute updates to a and b values representing tridiagonal matrix
        a(1) = sigma_vec(2) * q(1) + c(2) * z(1);
        b(2) = sigma_vec(2) * z(2);
        for j = 2:(n - 1)
            a(j) = sigma_vec(j + 1) * q(j) + c(j) * c(j + 1) * z(j);
            b(j + 1) = sigma_vec(j + 1) * z(j + 1);
        end
        a(n) = c(n) * z(n);
        
        % Find eigenvectors
        for i = 2:n
            col1 = eigenvectors(:, i - 1) * c(i) + ...
                    eigenvectors(:, i) * s(i);
            eigenvectors(:, i) = -s(i) * eigenvectors(:, i - 1) + ...
                                    c(i) * eigenvectors(:, i);
            eigenvectors(:, i - 1) = col1;
        end
        
        % find last row's eigenvalue
        if abs(b(n)) <= tolerance
            eigenvalues(n) = a(n) + SHIFT;
            n = n - 1;
        end
        
        k = k + 1;
    end
    
    % Pair off eigenvectors and eigenvalues so they can be sorted by the
    % eigenvalues
    eigens_to_sort = cell(size(eigenvectors, 1), 2);
    for i = 1:size(eigenvalues, 1)
        eigens_to_sort{i, 1} = eigenvectors(:, i);
        eigens_to_sort{i, 2} = eigenvalues(i);
    end

    % Sort
    sorted_eigens = sortrows(eigens_to_sort, 2);

    % Separate back out
    for i = 1:size(eigenvalues, 1)
        eigenvalues(i) = sorted_eigens{i, 2};
        eigenvectors(:, i) = sorted_eigens{i, 1};
    end
end

% We assume that A is symmetric
% We check if the matrix A is tridiagonal
% Return true if tridiagonal, false if not tridiagonal
function [is_tridiagonal] = egaebel_check_tridiagonal(A)

    N = size(A, 1);

    is_tridiagonal = true;
    for i = 1:(N - 1)
        % Check tridiagonal elements
        if A(i + 1, i) ~= A(i, i + 1)
            is_tridiagonal = false;
            return
        end
        % Check columns to the right of the tridiagonal elements
        for j = (i + 2):N
            if A(i, j) ~= 0
                is_tridiagonal = false;
                return
            end
        end
        % Check columns to the left of the tridiagonal elements
        for j = 1:(i - 2)
            if A(i, j) ~= 0
                is_tridiagonal = false;
                return
            end
        end
    end
end