% Performs spectral clustering on the passed in data set
% The data set is an array of vectors supporting n-dimensional data
function [clusters, means, eigenvalues, eigenvectors] = ...
            egaebel_norm_spectral_clustering(data_set, ...
                                                gen_similarity_graph, ...
                                                K)

    W = gen_similarity_graph(data_set);
    
    D = find_degree_matrix(W);
    D_inv = invert_diagonal(D);
    
    % Compute normalized Laplacian matrix
    % Generalized eigenvalue problem:
    % Lu = (lambda)Du
    % Equivalent to the random walk Laplacian's eigenvectors
    L = eye(size(W, 1)) - D_inv * W;

    % Compute k eigenvalues/vectors of L
    % k = number of clusters desired
    %[eigenvectors, eigenvalue_matrix] = eig(L);
    %eigenvalues = diag(eigenvalue_matrix);
    MAX_ITERATIONS = 1000;
    [eigenvalues, eigenvectors] = egaebel_qr_algorithm(L, ...
                                                        10^-9, ...
                                                        MAX_ITERATIONS);
    fprintf('initial eigens...\n')
    eigenvalues(1:3)
    eigenvectors(:, 1:3)
    [eig_eigenvectors, ~] = eig(L);
    eig_eigenvectors(:, 1:3)
    %{
    % Find K largest eigenvectors, with non-repeated eigenvalues
    % Must round floating points to work with unique properly
    decimal_digits = 7;
    D = 10^decimal_digits;
    unique_eigenvalues = zeros(size(eigenvalues));
    for i = 1:size(eigenvalues, 1)
        unique_eigenvalues(i) = round(eigenvalues(i) * D) / D;
    end
    unique_eigenvalues = unique(unique_eigenvalues, 'stable');
    
    tol_is_equal = @(x, y) (all(abs(x - y) <= (10^-7)));
    % Find largest eigenvectors for each eigenvalue
    largest_eig_idx = zeros(size(unique_eigenvalues, 1), 1);
    % over unique eigenvalues
    for i = 1:size(unique_eigenvalues, 1)
        
        cur_eigenvalue = unique_eigenvalues(i);
        % over all eigenvalues
        for j = 1:size(eigenvalues, 1)
            
            % if on current eigenvalue
            if tol_is_equal(eigenvalues(j), cur_eigenvalue)
    
                if largest_eig_idx(i, 1) == 0 || ...
                    (norm(eigenvectors(:, j)) >= ...
                        norm(eigenvectors(:, largest_eig_idx(i, 1))))

                    largest_eig_idx(i, 1) = j;
                end
            end
        end
    end

    % Find norm of each largest eigenvector
    largest_eig_norm = zeros(size(largest_eig_idx, 1), 2);
    for i = 1:size(largest_eig_idx, 1)
        largest_eig_norm(i, 1) = norm(eigenvectors(:, largest_eig_idx(i)));
        largest_eig_norm(i, 2) = largest_eig_idx(i);
    end
    
    % Sort
    largest_eig_norm = sortrows(largest_eig_norm, -1);
    
    % Recover indices
    k_largest_eig_idx = zeros(K, 1);
    cluster_eigenvectors = zeros(size(eigenvectors, 1), K);
    i = 1;
    while i <= K
        %k_largest_eig_idx(i)
        
        k_largest_eig_idx(i) = largest_eig_norm(i, 2);
        cluster_eigenvectors(:, i) = eigenvectors(:, k_largest_eig_idx(i));
        i = i + 1;
    end
    %}
    
    % Trim down to K columns of eigenvectors (also remove eigenvalues)
    while K < size(eigenvectors, 2)
        eigenvectors(:, K + 1) = [];
        eigenvalues(K + 1, :) = [];
    end
    
    cluster_eigenvectors = eigenvectors;

    cluster_eigenvectors
    transpose(eigenvalues)
    
    cluster_eigenvectors(1) * cluster_eigenvectors(2)
    cluster_eigenvectors(1) * cluster_eigenvectors(3)
    cluster_eigenvectors(2) * cluster_eigenvectors(3)
    
    % Cluster points
    [~, cluster_indices, means] = egaebel_kmeans(cluster_eigenvectors, ...
                                                   K, ...
                                                   MAX_ITERATIONS);

    %%%%% -------- Data Formatting -------- %%%%%
    % eigenvector indices match data set indices
    % cluster_indices matches the data_set by index
    for i = 1:size(cluster_indices, 1)
        data_set(i, 3) = cluster_indices(i);
    end
    
    % Sort by cluster number
    sorted_data = sortrows(data_set, 3);
    
    % Split into separate cluster vectors
    j = 1;
    i = 1;
    cur_cluster = sorted_data(1, 3);
    clusters = cell(K, 1);
    % While there's still data
    while j <= size(sorted_data, 1)
        % If this data is in the current cluster
        if sorted_data(j, 3) == cur_cluster
            % Place data into clusters
            clusters{cur_cluster}(i, 1) = sorted_data(j, 1);
            clusters{cur_cluster}(i, 2) = sorted_data(j, 2);
            j = j + 1;
            i = i + 1;
        else
            cur_cluster = sorted_data(j, 3);
            i = 1;
        end
    end
    
end

% Takes a diagonal matrix and inverts it
% Simply take the reciprocol of all diagonal elements
% This function does not verify that D is diagonal, so be sure!
% D is also assumed to be symmetric
function D = invert_diagonal(D)

    for i = 1:size(D, 1)
        if D(i, i) ~= 0
            D(i, i) = 1 / D(i, i);
        end
    end
end

% Takes a matrix A representing an undirected graph
% (NOTE: since A represents an undirected graph, A is symmetric)
% Returns the degree matrix D which has the degree
% of each vertex along the diagonal and 0s everywhere else
% i.e. d_ii = degree(v_i) in the graph G = (V, E) represented
% by the adjacency matrix A
function D = find_degree_matrix(A)

    D = zeros(size(A, 1));
    for i = 1:size(A, 1)
        degree = 0;
        for j = 1:size(A, 2)
            degree = degree + A(i, j);
        end
        D(i, i) = degree;
    end
end