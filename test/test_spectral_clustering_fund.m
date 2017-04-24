function test_spectral_clustering_fund()
    
    X = [0 1 1 0 1;
         1 0 1 0 0;
         1 1 0 0 1;
         0 0 0 0 1
         1 0 1 1 0];
     issymmetric(X)
     
     fund_unnorm_spectral_clustering(X, 3);
end

% Performs spectral clustering on the passed in data set
% The data set is an array of vectors supporting n-dimensional data
function [clusters, means, eigenvalues, eigenvectors] = ...
            fund_unnorm_spectral_clustering(A, K)

    D = find_degree_matrix(A);
    
    % Compute un-normalized Laplacian matrix
    L = D - A;
    
    % Compute normalized Laplacian
    D_inv = invert_diagonal(D);
    
    % Compute normalized Laplacian matrix
    L = sqrt(D_inv) * L * sqrt(D_inv);
    
    % Compute normalized Laplacian

    % Compute k eigenvalues/vectors of L
    % k = number of clusters desired
    %[eigenvectors, eigenvalue_matrix] = eig(L);
    %eigenvalues = diag(eigenvalue_matrix);
    MAX_ITERATIONS = 10000;
    [eigenvalues, eigenvectors] = egaebel_qr_algorithm(L, ...
                                                        10^-9, ...
                                                        MAX_ITERATIONS);
    fprintf('initial eigens...\n')
    eigenvalues
    eigenvectors
    [eig_eigenvectors, eig_eigenvalues] = eig(L);
    diag(eig_eigenvalues)
    eig_eigenvectors
    return
    % Trim down to K columns of eigenvectors (also remove eigenvalues)
    while K < size(eigenvectors, 2)
        eigenvectors(:, K + 1) = [];
        eigenvalues(K + 1, :) = [];
    end
    
    % Cluster points
    MAX_ITERATIONS = 1000;
    [clusters, cluster_indices, means] = egaebel_kmeans(eigenvectors, ...
                                                           K, ...
                                                           MAX_ITERATIONS);
    %cluster_indices
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
    while j <= size(sorted_data, 1)
        if sorted_data(j, 3) == cur_cluster
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