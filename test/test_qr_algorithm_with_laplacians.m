function test_qr_algorithm_with_laplacians()
   
    L = [1.0000         0         0         0         0         0         0         0         0         0;
            0    1.0000   -0.3333   -0.3333   -0.3333         0         0         0         0         0;
            0   -0.3333    1.0000   -0.3333   -0.3333         0         0         0         0         0;
            0   -0.3333   -0.3333    1.0000   -0.3333         0         0         0         0         0;
            0   -0.3333   -0.3333   -0.3333    1.0000         0         0         0         0         0;
            0         0         0         0         0    1.0000         0         0         0         0;
            0         0         0         0         0         0    1.0000   -0.3333   -0.3333   -0.3333;
            0         0         0         0         0         0   -0.3333    1.0000   -0.3333   -0.3333;
            0         0         0         0         0         0   -0.3333   -0.3333    1.0000   -0.3333;
            0         0         0         0         0         0   -0.3333   -0.3333   -0.3333    1.0000];

    tolerance = 10^-9;
    MAX_ITERATIONS = 1000;
    L
    
    [eigenvalues, eigenvectors] = egaebel_qr_algorithm(L, ...
                                                        tolerance, ...
                                                        MAX_ITERATIONS);
    
    eigenvalues
    eigenvectors
    
    tol_is_equal = @(x, y) (all(abs(x - y) <= (10^-5)));
    fprintf('\n\nChecking my Eigenvalue/vector pairs...for my QR!!!!\n')
    for i = 1:size(eigenvalues, 1)
        lhs = L * eigenvectors(:, i);
        rhs = eigenvectors(:, i) * eigenvalues(i);
        %fprintf('-------equality?-----------\n')
        if tol_is_equal(lhs, rhs)
            %Don't do anything
        else
            fprintf('LHS != RHS in eigenvalue/vector check!\n')
            lhs
            rhs
            abs(lhs - rhs) <= (1*10^-7)
            return
        end
    end
    fprintf('Verified!\n')

    fprintf('---------------------------\n')
    fprintf('compared to eig\n')
    
    [eig_eigenvectors, eig_eigenvalue_matrix] = eig(L);
    eig_eigenvalues = diag(eig_eigenvalue_matrix);
    
    eig_eigenvalue_matrix
    eig_eigenvectors
    
    for i = 1:size(eigenvalues, 1)
        if tol_is_equal(eigenvalues, eig_eigenvalues)
            % DO nothing
        else
            fprintf('Failure on eigenvalue comp\n')
            eigenvalues
            eig_eigenvalues
            return
        end
    end
    
    eigenvalues
    eig_eigenvalues
    fprintf('Verified vs EIG!\n')
    return
    
    %
    fprintf('---------------------------\n')
    fprintf('compared to eig on householder transform\n')
    L
    H = egaebel_householder_method(L);
    H
    [eig_eigenvectors, eig_eigenvalue_matrix] = eig(H);
    eig_eigenvalues = diag(eig_eigenvalue_matrix);
    
    eig_eigenvalue_matrix
    eig_eigenvectors
    %}
    
    tol_is_equal = @(x, y) (all(abs(x - y) <= (1*10^-7)));
    fprintf('\n\nChecking my Eigenvalue/vector pairs...\n')
    for i = 1:size(eig_eigenvalues, 1)
        lhs = L * eig_eigenvectors(:, i);
        rhs = eig_eigenvectors(:, i) * eig_eigenvalues(i);
        %fprintf('-------equality?-----------\n')
        if tol_is_equal(lhs, rhs)
            %Don't do anything
        else
            fprintf('LHS != RHS in eigenvalue/vector check!\n')
            lhs
            rhs
            return
        end
    end
    fprintf('Verified!\n')
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %{
    fprintf('\n\nNon-tridiagonal matrix from book---------------------:\n')
    A = [4 1 -2  2;
         1 2  0  1;
        -2 0  3 -2;
         2 1 -2 -1];
    
    fprintf('\n\nQR Algorithm results\n')
    [eigenvalues, eigenvectors] = egaebel_qr_algorithm(A, ...
                                                        tolerance, ...
                                                        MAX_ITERATIONS)

    fprintf('\n\nEig results of A from MATLAB\n')
    [eigenvectors, eigenmatrix] = eig(A);
    eigenvalues = diag(eigenmatrix);
    eigenvalues
    eigenvectors
                      
    tol_is_equal = @(x, y) (all(abs(x - y) <= (1*10^-7)));
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
    %}
end