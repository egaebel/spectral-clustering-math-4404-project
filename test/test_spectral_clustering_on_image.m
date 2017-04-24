function test_spectral_clustering_on_image()
    
    close_figures();

    % Setup paths for my machine
    % Change main_project_dir to wherever 'src' and 'test' are located
    main_project_dir = '/home/egaebel/grad-docs/spring-2015/math-4404/project/';
    cd(main_project_dir);
    cur_dir = pwd();
    source_dir = sprintf('%s/src/', main_project_dir);
    test_dir = sprintf('%s/test/', main_project_dir);
    img_dir = sprintf('%s/image/', main_project_dir);
    addpath(source_dir);
    addpath(test_dir);
    addpath(img_dir);
    cd(cur_dir);
    
    img_name = 'bright_bird.jpeg';
    img_name = 'super_bright_bird.jpg';
    img_name = 'bright-bird-grey-back.jpg';
    img_name = 'bright-3-color-bird.jpg';
    %img_name = 'baby_colored_blocks.jpg';
    img_path = sprintf('%s%s', img_dir, img_name);
    img_data = imread(img_path);
    
    img_data = im2double(img_data);
    
    imshow(rgb2ind(img_data, 0.6))
    return
    
    r_img_data = img_data(:, :, 1);
    g_img_data = img_data(:, :, 2);
    b_img_data = img_data(:, :, 3);
    
    bw_img_data = im2bw(img_data, 0.4);
    size(bw_img_data)
    figure('name', 'bw bird');
    imshow(bw_img_data)
    figure('name', 'r bird');
    imshow(r_img_data)
    figure('name', 'g bird');
    imshow(g_img_data)
    figure('name', 'b bird');
    imshow(b_img_data)
    
    % Spectral clustering on img_data
    K = 3;

    for k = 3:K
        [clustered_data, original_row_indices] = ...
            test_spectral_clustering_procedure(g_img_data, k, ...
                                    @egaebel_unnorm_spectral_clustering);
    end
end

function [clustered_data, original_row_indices] = test_spectral_clustering_procedure(data, ...
                                                                K, ...
                                                                spectral_fun)

    epsilons = transpose(3.0:1.0:5.0);
    sigmas = transpose(1.0:0.5:5.0);

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
        [clustered_data, ~, original_row_indices] = spectral_fun(data, ...
                @(data)(generate_epsilon_neighborhood_similarity_graph(data, epsilons(i))), ...
                K);
        % epsilon_graph_gens{i}, ...
        % means
        spectral_clustering_name = ...
                sprintf('Spectral Clustering--Epsilon--%f', epsilons(i));
        display_clustered_image_data(clustered_data, ...
                                        original_row_indices, ...
                                        data, ...
                                        epsilons(i));
    end
    %}
    
    % Gaussian similarity graphs
    %
    for i = 1:size(sigmas, 1)
        [clustered_data, ~, original_row_indices] = spectral_fun(data, ...
                @(data)(generate_gaussian_similarity_graph(data, sigmas(i))), ...
                K);
        % means
        spectral_clustering_name = ...
                sprintf('Image Clustering--Gaussian--%f', sigmas(i));
        display_clustered_image_data(clustered_data, ...
                                        original_row_indices, ...
                                        data, ...
                                        spectral_clustering_name);
    end
    %}
end

function display_clustered_image_data(clustered_data, ...
                                        original_row_indices, ...
                                        img_data, ...
                                        fig_id)

    close_figures();
                                    
    color_matrices = zeros(size(img_data));
    for i = 1:size(clustered_data, 1)

        c = clustered_data{i};

        % Fill color matrix in at original rows
        color_mat = zeros(size(img_data));
        for j = 1:size(c, 1)
            color_mat(original_row_indices(j), :) = c(j, :);
        end

        color_matrices(:, :, i) = color_mat;

    end

    figure('name', fig_id)
    imshow(color_matrices)

    %{
    for j = 1:k
        fig_name = sprintf('cluster %d', j);
        figure('name', fig_name);
        imshow(color_matrices(:, :, j));
    end
    %}
end

function test_kmeans_comp(data, K)

    % Run K-Means with 6 clusters (6 distributions!)
    MAX_ITERATIONS = 100;
    [clusters, ~, means] = egaebel_kmeans(data, K, MAX_ITERATIONS);
    means
                                                        
    plot_clusters(clusters, means, 'K-Means')
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