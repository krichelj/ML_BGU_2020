function classifier = learnknn(k, d, m, Xtrain, Ytrain)

    function label = knn(x, k, d, m, Xtrain, Ytrain)
        
        distances = zeros(m,1);
        for j=1:m
            distances(j) = norm(x-Xtrain(j,:));
        end
        
        knnLabels = zeros(k,1);
        
        for j=1:k
            [~, minIndex] = min(distances);
            knnLabels(j) = Ytrain(minIndex);
            distances(minIndex) = realmax;
        end
            
        label = mode(knnLabels);
    end

    classifier = @(x) knn(x, k, d, m, Xtrain, Ytrain);
  
end