function test_spectral_clustering()
    
    % Setup paths for my machine
    % Change main_project_dir to wherever 'src' and 'test' are located
    main_project_dir = '/home/egaebel/grad-docs/spring-2015/math-4404/project/';
    cd(main_project_dir);
    cur_dir = pwd();
    source_dir = sprintf('%s/src/', main_project_dir);
    test_dir = sprintf('%s/test/', main_project_dir);
    addpath(source_dir);
    addpath(test_dir);
    cd(cur_dir);

    close_figures();

    %
    K = 3;

    % Generate Data
    xs = cell(K, 1);
    ys = cell(K, 1);
    
    [xs{1}, ys{1}] = fuzzy_circle(1);
    if K > 1
        [xs{2}, ys{2}] = fuzzy_circle(10);
    end
    if K > 2
        [xs{3}, ys{3}] = fuzzy_circle(20);
    end
    if K > 3
        [xs{4}, ys{4}] = fuzzy_circle(40);
    end
    if K > 4
        [xs{5}, ys{5}] = fuzzy_circle(80);
    end
    if K > 5
        [xs{6}, ys{6}] = fuzzy_circle(160);
    end
    
    % Combine Xs into the Y
    for j = 1:K
        for k = 1:size(xs{j}, 1)
            t = ys{j}(k, 1);
            ys{j}(k, 1) = xs{j}(k);
            ys{j}(k, 2) = t;
        end
    end
    
    % Loop to print ys values for visual check
    for j = 1:K
        transpose(ys{j});
    end

    % Concatenate all data into one Nx2 vector
    data = vertcat(ys{:});
    
    % Clean data (check for Inf and NaN)
    i = 1;
    while i <= size(data, 1)
        for j = 1:size(data, 2)
            if isinf(data(i, j)) || isnan(data(i, j))
                data(i, :) = [];
                % fprintf('HOUSE KEEP-ING!\n')
            end
        end
        i = i + 1;
    end

    % Un-normed spectral clustering
    %
    test_spectral_clustering_procedure(data, ... 
                                        K, ... 
                                        @egaebel_unnorm_spectral_clustering);
    %}
    
    % normed spectral clustering
    %{
    test_spectral_clustering_procedure(data, ... 
                                        K, ... 
                                        @egaebel_norm_spectral_clustering);
    %}
    
    % K-Means
    test_kmeans_comp(data, K)
end


function test_spectral_clustering_procedure(data, K, spectral_fun)

    epsilons = transpose(1.0:1.0:5.0);
    sigmas = transpose(1.0:1.0:10.0);

    epsilon_graph_gens = cell(size(epsilons, 1), 1);
    simga_graph_gens = cell(size(sigmas, 1), 1);
    
    % Generate epsilons
    for i = 1:size(epsilons, 1)
        epsilon_graph_gens{i} = ...
            @(data) generate_epsilon_neighborhood_similarity_graph(data, ...
                                                                    epsilons(i));
    end
    
    % Generate sigmas
    for i = 1:size(sigmas, 1)
        simga_graph_gens{i} = ...
            @(data) generate_gaussian_similarity_graph(data, sigmas(i));
    end

    % Epsilon neighborhood similarity graphs
    %
    for i = 1:size(epsilons, 1)
        fprintf('epsilon %f\n', epsilons(i))
        [clustered_data, means] = spectral_fun(data, ...
                @(data)(generate_epsilon_neighborhood_similarity_graph(data, epsilons(i))), ...
                K);
        % epsilon_graph_gens{i}, ...
        % means
        spectral_clustering_name = ...
                sprintf('Spectral Clustering--Epsilon--%f', epsilons(i));
        plot_clusters(clustered_data, means, spectral_clustering_name)
    end
    %}
    
    % Gaussian similarity graphs
    %
    for i = 1:size(sigmas, 1)
        [clustered_data, means] = spectral_fun(data, ...
                @(data)(generate_gaussian_similarity_graph(data, sigmas(i))), ...
                K);
        % means

        spectral_clustering_name = ...
                sprintf('Spectral Clustering--Gaussian--%f', sigmas(i));
        plot_clusters(clustered_data, means, spectral_clustering_name)
    end
    %}
    
    test_kmeans_comp(data, K)
end

function test_kmeans_comp(data, K)

    % Run K-Means with 6 clusters (6 distributions!)
    MAX_ITERATIONS = 100;
    [clusters, ~, means] = egaebel_kmeans(data, K, MAX_ITERATIONS);
    means
                                                        
    plot_clusters(clusters, means, 'K-Means')
end

function [X, Y] = fuzzy_circle(r)

    Theta = transpose(0:(pi / 16):(2 * pi));

    noise_per_sample = 1;
    % 2 vals per theta, noise_per_sample for each val
    data_size = size(Theta, 1) * 2;
    data_size = data_size + data_size * max(noise_per_sample, 1);
    X = zeros(data_size, 1);
    Y = zeros(data_size, 1);
    i = 1;
    j = 1;
    while i <= size(Theta, 1)
        
        base_index_1 = j;
        base_index_2 = j + 1;
        
        X(j) = r * cos(Theta(i));
        Y(j) = r * sin(Theta(i));
        j = j + 1;
        
        X(j) = X(j - 1);
        Y(j) = -Y(j - 1);
        j = j + 1;
        
        % Add noise
        for k = 1:noise_per_sample
            
            if randn > 0
                X(j) = X(base_index_1) + normpdf(randn);
                Y(j) = Y(base_index_1) + normpdf(randn);
                j = j + 1;
            else
                X(j) = X(base_index_1) - normpdf(randn);
                Y(j) = Y(base_index_1) - normpdf(randn);
                j = j + 1;
            end
            
            if randn > 0
                X(j) = X(base_index_2) + normpdf(randn);
                Y(j) = Y(base_index_2) + normpdf(randn);
                j = j + 1;
            else
                X(j) = X(base_index_2) - normpdf(randn);
                Y(j) = Y(base_index_2) - normpdf(randn);
                j = j + 1;
            end
        end
        
        i = i + 1;
    end

    assert (size(X, 1) == data_size);
    assert (size(X, 1) == size(Y, 1) && size(X, 2) == size(Y, 2));
end

function plot_clusters(clusters, means, fig_name)
    % Plot clusters---------------------------------------------------
    figure('name', fig_name);
    hold on
    clusters
    
    if size(clusters, 1) >= 1 && size(clusters{1}, 1) > 0
        plot(clusters{1}(:, 1), clusters{1}(:, 2), 'bo')
    end
    if size(clusters, 1) >= 2 && size(clusters{2}, 1) > 0
        plot(clusters{2}(:, 1), clusters{2}(:, 2), 'go')
    end
    if size(clusters, 1) >= 3 && size(clusters{3}, 1) > 0
        plot(clusters{3}(:, 1), clusters{3}(:, 2), 'ro')
    end
    if size(clusters, 1) >= 4 && size(clusters{4}, 1) > 0
        plot(clusters{4}(:, 1), clusters{4}(:, 2), 'ko')
    end
    if size(clusters, 1) >= 5 && size(clusters{5}, 1) > 0
        plot(clusters{5}(:, 1), clusters{5}(:, 2), 'yo')
    end
    if size(clusters, 1) >= 6 && size(clusters{6}, 1) > 0
        plot(clusters{6}(:, 1), clusters{6}(:, 2), 'co')
    end

    hold off
end

function close_figures
    % Close all figures.
    window_handles = findall(0);
    delete(window_handles(2:end));
end