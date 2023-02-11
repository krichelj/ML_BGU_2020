function Ytestprediction = predictknn(classifier, n, Xtest)
  
    Ytestprediction = zeros(n,1);
    
    for i=1:n
        Ytestprediction(i) = classifier(Xtest(i,:));
    end
  
end