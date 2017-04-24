function W = generate_gaussian_similarity_graph(data_set, sigma)

    W = zeros(size(data_set, 1));
    sim = @(x, y) exp(norm(abs(x) - abs(y)) / (2 * sigma^2));
    
    for i = 1:size(data_set, 1)
        for j = 1:size(data_set, 1)
            if i ~= j
                s = sim(data_set(i, :), data_set(j, :));
                W(i, j) = s;
                W(j, i) = s;
            end
        end
    end
    
    assert (issymmetric(W))
end