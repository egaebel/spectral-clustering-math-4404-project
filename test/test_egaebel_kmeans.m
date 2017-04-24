function test_egaebel_kmeans()

    close_figures();

    K = 6;

    X = transpose(0.01:0.005:1);
    ys = cell(K, 1);
    ys{1} = betapdf(X, 0.75, 0.75);
    ys{2} = betapdf(X, 3, 1);
    ys{3} = betapdf(X, 1, 3);
    ys{4} = normpdf(X, 0, 1);
    ys{5} = normpdf(X, 3, 1);
    ys{6} = normpdf(X, -3, 1);
    
    % Combine X into the Y
    for i = 1:size(X, 1)
        
        for j = 1:size(ys, 1)
            
            t = ys{j}(i, 1);
            ys{j}(i, 1) = X(i);
            ys{j}(i, 2) = t;
        end
    end
    
    % Loop to print ys values for visual check
    for j = 1:size(ys, 1)
        transpose(ys{j});
    end

    % Concatenate all data into one Nx2 vector
    data = vertcat(ys{:});
    
    % Clean data (check for Inf and NaN)
    while i <= size(data, 1)
        for j = 1:size(data, 2)
            if isinf(data(i, j)) || isnan(data(i, j))
                data(i, :) = [];
            end
        end
        i = i + 1;
    end
    
    % Run K-Means with 6 clusters (6 distributions!)
    MAX_ITERATIONS = 100;
    [clusters, ~, means] = egaebel_kmeans(data, K, MAX_ITERATIONS);
                                                        
    % Plot clusters---------------------------------------------------
    hold on
    plot(clusters{1}(:, 1), clusters{1}(:, 2), 'bo', ...
            clusters{2}(:, 1), clusters{2}(:, 2), 'go', ...
            clusters{3}(:, 1), clusters{3}(:, 2), 'ro', ...
            clusters{4}(:, 1), clusters{4}(:, 2), 'co', ...
            clusters{5}(:, 1), clusters{5}(:, 2), 'yo', ...
            clusters{6}(:, 1), clusters{6}(:, 2), 'ko')
        
    plot(means(1, 1), means(1, 2), 'bo', ...
            'MarkerFaceColor', 'b', ...    
            'MarkerSize', 20)
    plot(means(2, 1), means(2, 2), 'go', ...
            'MarkerFaceColor', 'g', ...    
            'MarkerSize', 20)
    plot(means(3, 1), means(3, 2), 'ro', ...
            'MarkerFaceColor', 'r', ...    
            'MarkerSize', 20)
    plot(means(4, 1), means(4, 2), 'b+', ...
            'MarkerFaceColor', 'b', ...    
            'LineWidth', 7, ...
            'MarkerSize', 20)
    plot(means(5, 1), means(5, 2), 'g+', ...
            'MarkerFaceColor', 'g', ...    
            'LineWidth', 7, ...
            'MarkerSize', 20)
    plot(means(6, 1), means(6, 2), 'r+', ...
            'MarkerFaceColor', 'r', ...    
            'LineWidth', 7, ...
            'MarkerSize', 20)
    
    hold off
end

function close_figures
    % Close all figures.
    window_handles = findall(0);
    delete(window_handles(2:end));
end