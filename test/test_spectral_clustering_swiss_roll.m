function test_spectral_clustering_swiss_roll()

    close_figures();

    % create swiss roll data
    N = 2^8;%2^11; % number of points considered
    t = rand(1,N);
    t = sort(4*pi*sqrt(t))'; 

    %t = sort(generateRVFromRand(2^11,@(x)1/32/pi^2*x,@(x)4*pi*sqrt(x)))';
    z = 8*pi*rand(N,1); % random heights
    x = (t+.1).*cos(t);
    y = (t+.1).*sin(t);
    data = [x,y,z]; % data of interest is in the form of a n-by-3 matrix

    % visualize the data
    cmap = jet(N);
    scatter3(x,y,z);%,20,cmap);
    title('Original data');
    
    for k = 2:6
        test_swiss_spectral_clustering_procedure(data, k, ...
                    @egaebel_unnorm_spectral_clustering);
    end
end

function test_swiss_spectral_clustering_procedure(data, K, spectral_fun)

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
    %{
    for i = 1:size(epsilons, 1)
        fprintf('epsilon %f\n', epsilons(i))
        [clustered_data, means] = spectral_fun(data, ...
                @(data)(generate_epsilon_neighborhood_similarity_graph(data, epsilons(i))), ...
                K);
        % epsilon_graph_gens{i}, ...
        % means
        spectral_clustering_name = ...
                sprintf('Spectral Clustering--Epsilon--%f', epsilons(i));
        scatter3_clusters_plot(clustered_data, means, spectral_clustering_name);
        return
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
        scatter3_clusters_plot(clustered_data, means, spectral_clustering_name)
    end
    %}
    
    %test_kmeans_comp(data, K)
end

function scatter3_clusters_plot(clusters, means, fig_name)
    % Plot clusters---------------------------------------------------
    figure('name', fig_name);
    hold on
    fprintf('it IS here...\n')
    %
    if size(clusters, 1) >= 1 && size(clusters{1}, 1) > 0
        scatter3(clusters{1}(:, 1), clusters{1}(:, 2),  clusters{1}(:, 3), 20, 'bo')
    end
    if size(clusters, 1) >= 2 && size(clusters{2}, 1) > 0
        scatter3(clusters{2}(:, 1), clusters{2}(:, 2),  clusters{2}(:, 3), 20, 'go')
    end
    if size(clusters, 1) >= 3 && size(clusters{3}, 1) > 0
        scatter3(clusters{3}(:, 1), clusters{3}(:, 2),  clusters{3}(:, 3), 20, 'ro')
    end
    if size(clusters, 1) >= 4 && size(clusters{4}, 1) > 0
        scatter3(clusters{4}(:, 1), clusters{4}(:, 2),  clusters{4}(:, 3), 20, 'ko')
    end
    if size(clusters, 1) >= 5 && size(clusters{5}, 1) > 0
        scatter3(clusters{5}(:, 1), clusters{5}(:, 2),  clusters{5}(:, 3), 20, 'yo')
    end
    if size(clusters, 1) >= 6 && size(clusters{6}, 1) > 0
        scatter3(clusters{6}(:, 1), clusters{6}(:, 2), clusters{6}(:, 3), 20, 'co')
    end
    %}
    hold off
end

function test_kmeans_comp(data, K)

    % Run K-Means with 6 clusters (6 distributions!)
    MAX_ITERATIONS = 100;
    [clusters, ~, means] = egaebel_kmeans(data, K, MAX_ITERATIONS);
    means
                                                        
    scatter3_clusters_plot(clusters, means, 'K-Means')
end

function close_figures
    % Close all figures.
    window_handles = findall(0);
    delete(window_handles(2:end));
end