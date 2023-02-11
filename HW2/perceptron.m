function w = perceptron(m, d, Xtrain, Ytrain, maxupdates)
  
 t=1;
 w = zeros(1,d);
 index=1;
 numOfGoodExamples=0;
 
while numOfGoodExamples < m
   
   curr_x = Xtrain(index,:);
   curr_y = Ytrain(index);
   if curr_y*dot(w,curr_x) <= 0
     w=w+(curr_y*curr_x);
     t = t + 1;
     numOfGoodExamples=0;
   else
     numOfGoodExamples = numOfGoodExamples + 1;
   endif
   
   if (t-1) >= maxupdates
     break;
   end
   
   index = index + 1;
   if index > m
     index=1;
   end
   
   end
 
 w = w';
 
end
