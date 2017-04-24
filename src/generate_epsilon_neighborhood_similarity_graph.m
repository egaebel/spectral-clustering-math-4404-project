% Generate a similarity matrix from passed in data points
% based on the epsilon-neighborhood technique
% A quick summary is that this uses some measure of "closeness"
% where the closeness is summarized in a value epsilon
% This "epsilon-closeness" determines the weight of an edge
% between two data points that will be put into this W matrix
function W = generate_epsilon_neighborhood_similarity_graph(data_set, ...
                                                            epsilon)

    W = zeros(size(data_set, 1));
    sim = @(x, y) norm(abs(x) - abs(y)) <= epsilon;
    
    for i = 1:size(data_set, 1)
        for j = 1:size(data_set, 1)
            if i ~= j
                s = sim(data_set(i, :), data_set(j, :));
                if s == 1
                    W(i, j) = 1;
                    W(j, i) = 1;
                end
            end
        end
    end
    
    assert (issymmetric(W))
end