% Performs spectral clustering on the passed in data set
% The data set is an array of vectors supporting n-dimensional data
function [clusters, means, original_row_indices, eigenvalues, eigenvectors] = ...
            egaebel_unnorm_spectral_clustering(data_set, ...
                                                gen_similarity_graph, ...
                                                K)

    W = gen_similarity_graph(data_set);

    D = find_degree_matrix(W);
    
    % Compute un-normalized Laplacian matrix
    L = D - W;

    % Compute k eigenvalues/vectors of L
    % k = number of clusters desired
    MAX_ITERATIONS = 10000;
    [eigenvalues, eigenvectors] = egaebel_qr_algorithm(L, ...
                                                        10^-9, ...
                                                        MAX_ITERATIONS);
    
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
    cluster_index = size(data_set, 2) + 1;
    original_row_index = cluster_index + 1;
    for i = 1:size(cluster_indices, 1)
        %data_set(i, 3) = cluster_indices(i);
        data_set(i, cluster_index) = cluster_indices(i);
        data_set(i, original_row_index) = i;
    end
    
    % Sort by cluster number
    sorted_data = sortrows(data_set, cluster_index);
    
    % Split into separate cluster vectors
    j = 1;
    i = 1;
    original_row_indices = zeros(size(data_set, 1));
    cur_cluster = sorted_data(1, cluster_index);
    clusters = cell(K, 1);
    while j <= size(sorted_data, 1)
        if sorted_data(j, cluster_index) == cur_cluster
            clusters{cur_cluster}(i, :) = sorted_data(j, 1:cluster_index - 1);
            original_row_indices(j) = sorted_data(j, original_row_index);
            j = j + 1;
            i = i + 1;
        else
            cur_cluster = sorted_data(j, cluster_index);
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