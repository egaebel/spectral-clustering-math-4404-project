% Rigorously test my QR algorithm implementation
function test_egaebel_qr_algorithm

    % egaebel_qr_algorithm(A, tolerance, MAX_ITERATIONS)

    % Tridiagonal matrix---------------------------------------------------
    fprintf('\n\nTridiagonal matrix-----------------------------------:\n')
    A = [3 1 0;
         1 3 1;
         0 1 3];
    tolerance = 10^-20;
    MAX_ITERATIONS = 10000;
    
    % With tridiagonal matrix passed
    [eigenvalues, eigenvectors] = egaebel_qr_algorithm(A, ... 
                                                        tolerance, ...
                                                        MAX_ITERATIONS)
    
    fprintf('\n\nEig results from MATLAB\n')
    [eigenvectors, eigenmatrix] = eig(A)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Non tridiagonal matrix-----------------------------------------------
      
    tol_is_equal = @(x, y) (all(abs(x - y) <= (1*10^-7)));
     
    % Generate random symmetric matrices and check algorithms--------------
    fprintf('random tests---------------------------------------------\n')
    % Number of iterations per matrix size
    for j = 1:10
        % Different matrix sizes
        for k = 2:200
            B = zeros(k);
            % Construct matrices
            for l = 1:size(B, 1)
                for p = 1:size(B, 1)
                    rand_num = rand();
                    % Add some larger numbers, randomly (coin flip)
                    if rand() > 0.5
                        for z = 1:abs((rand() * 10) - 5)
                            rand_num = rand_num * 10;
                        end
                    end
                    B(l, p) = rand_num;
                    B(p, l) = rand_num;
                end
            end
            
            % Actually do test
            %fprintf('\n\nQR Algorithm results\n')
            [eigenvalues, eigenvectors] = ...
                                egaebel_qr_algorithm(B, ...
                                                   tolerance, ...
                                                   MAX_ITERATIONS);

            %fprintf('\n\nChecking my Eigenvalue/vector pairs..\n')
            for i = 1:size(eigenvalues, 1)
                lhs = B * eigenvectors(:, i);
                rhs = eigenvectors(:, i) * eigenvalues(i);
                %fprintf('-------equality?-----------\n')
                if tol_is_equal(lhs, rhs)
                    %don't do anything
                else
                    fprintf('iteration: %d\n', j + k)
                    lhs
                    rhs
                    fprintf('mismatches at indices...\n')
                    for m = 1:size(lhs)
                        if lhs(m) ~= rhs(m)
                            m
                            lhs(m)
                            rhs(m)
                        end
                    end
                    fprintf('LHS != RHS in eigenvalue/vector check!\n')
                    return;
                end
            end
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    fprintf('\n\nNon-tridiagonal matrix-------------------------------:\n')
    A = [4 1 -2  2;
         1 2  0  1;
        -2 0  3 -2;
         2 1 -2 -1];
    
    fprintf('\n\nQR Algorithm results\n')
    [eigenvalues, eigenvectors] = egaebel_qr_algorithm(A, ...
                                                        tolerance, ...
                                                        MAX_ITERATIONS)

    fprintf('\n\nChecking my Eigenvalue/vector pairs...\n')
    for i = 1:size(eigenvalues, 1)
        lhs = A * eigenvectors(:, i);
        rhs = eigenvectors(:, i) * eigenvalues(i);
        %fprintf('-------equality?-----------\n')
        if tol_is_equal(lhs, rhs)
            %Don't do anything
        else
            fprintf('LHS != RHS in eigenvalue/vector check!\n')
            lhs
            rhs
            break;
        end
    end
                                                    
    fprintf('\n\nEig results of A from MATLAB\n')
    [eigenvectors, eigenmatrix] = eig(A)
end