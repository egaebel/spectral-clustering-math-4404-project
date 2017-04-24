function [clusters, cur_clusters, means] = egaebel_kmeans(X, K, ...
                                                            MAX_ITERATIONS)

    means = kmeans_plus_plus(X, K);
    %means = forgy_initialization(X, K);
    norms = zeros(K, 1);
    cur_clusters = zeros(size(X, 1), 1);

    count = 0;
    while count < MAX_ITERATIONS
        
        num_assignments = 0;
        
        % Assign each data point to a cluster based on differenced 2-norm 
        % O(|X|^2 + |X| K)
        for j = 1:size(X, 1)
            
            % O(K + |X|)
            for p = 1:K
                norms(p) = norm((X(j, :) - means(p, :)), 2);
            end
            [~, indices] = min(norms);

            % Assign to cluster
            if cur_clusters(j) ~= indices(1)
                cur_clusters(j) = indices(1);
                num_assignments = num_assignments + 1;
            end
        end
        
        
        % Sum over elements in clusters and count number of elements
        % O(|X|)
        cluster_sums = zeros(K, 1);
        cluster_sizes = zeros(K, 1);
        for j = 1:size(X, 1)
            
            cluster_sums(cur_clusters(j)) = ...
                cluster_sums(cur_clusters(j)) + X(j);
            
            cluster_sizes(cur_clusters(j)) = ...
                cluster_sizes(cur_clusters(j)) + 1;
        end
        
        % Update means
        % O(K)
        for p = 1:size(means, 1)
            
            means(p) = cluster_sums(p) / cluster_sizes(p);
        end
        
        count = count + 1;
        
        % Exit condition
        if num_assignments == 0
            break;
        end
    end
    
    fprintf('K-Means Iterations: %d\n', count)
    
    %%%%% -------- Data Formatting -------- %%%%%
    % Add cluster info to data vector (making it Nx(K + 1))
    cluster_index = size(X, 2) + 1;
    for i = 1:size(X, 1)
        X(i, cluster_index) = cur_clusters(i);
    end
    
    % Sort by cluster number
    sorted_X = sortrows(X, size(X, 2));
    
    % Split into separate cluster vectors
    j = 1;
    i = 1;
    cur_cluster = sorted_X(1, size(X, 2));
    clusters = cell(K, 1);
    while j <= size(sorted_X, 1)
        if sorted_X(j, size(X, 2)) == cur_cluster
            clusters{cur_cluster}(i, :) = sorted_X(j, 1:end - 1);
            j = j + 1;
            i = i + 1;
        else
            cur_cluster = sorted_X(j, size(X, 2));
            i = 1;
        end
    end
end

function [means] = kmeans_plus_plus(X, K)
    
    % Must round floating points to work with unique properly
    decimal_digits = 9;
    D = 10^decimal_digits;
    for i = 1:size(X, 1)
        for j = 1:size(X, 2)
            X(i, j) = round(X(i, j) * D) / D;
        end
    end

    % I do not want duplicate means
    X = unique(X, 'rows', 'stable');

    random_index = randi(size(X, 1), 1, 1);
    means = zeros(K, size(X, 2));

    means(1, :) = X(random_index, :);
    X(random_index, :) = [];
    min_distances = zeros(size(X, 1), 2);
    means_chosen = 1;
    % Over all clusters -- O(K)
    for i = 2:K
        
        probs = zeros(size(X, 1), 1);
        % Over all data to find distances -- O(N)
        for j = 1:size(X, 1)
            % Get min distance to a cluster for point X(j)
            min_distance = norm(X(j) - means_chosen(1));
            min_distance_index = 1;
            % Over all means -- O(K) (max)
            for k = 2:means_chosen 
                temp_dist = norm(X(j) - means_chosen(k));
                if temp_dist < min_distance
                    min_distance = temp_distance;
                    min_distance_index = k;
                end
            end
            min_distances(j, :) = [min_distance^2, min_distance_index];
        end
        
        % Over all data to compute probs O(N)
        for j = 1:size(X, 1)
            % Assign probabilities to Xs
            probs(j, 1) = min_distances(j, 1) / sum(min_distances(:, 1));
        end

        % Select next mean
        [selection, index] = select_with_prob(X, probs);
        means(i, :) = selection(1, :);
        X(index, :) = [];
    end
    
    fprintf('Exiting K-Means-Plus-Plus\n')
    means
    %{
    1. Choose one center uniformly at random from among the data points.
    2. For each data point x, compute D(x), the distance between x and the 
               nearest center that has already been chosen.
    3. Choose one new data point at random as a new center, 
               using a weighted probability distribution where 
               a point x is chosen with probability proportional to D(x)^2.
    Repeat Steps 2 and 3 until k centers have been chosen.
    Now that the initial centers have been chosen, 
               proceed using standard k-means clustering.
    %}
end

function [val, index] = select_with_prob(X, P)
    x = cumsum([0 P(:).'/sum(P(:))]);
    x(end) = 1e3 * eps + x(end);
    [a a] = histc(rand, x);
    val = X(a, :);
    index = a;
end

% Takes a dataset X and an integer k
% Selects k random means from X and returns them
function [random_means] = forgy_initialization(X, K)

    random_means = zeros(K, size(X, 2));
    random_indices = randi(size(X, 1), K, 1);

    for i = 1:K
        random_means(i, :) = X(random_indices(i, 1), :);
    end
end