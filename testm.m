function  testm( projectimg ,eigenfaces, testimg , folder ,m)

    
        figure;
        trainDataset = imageSet(folder);
        img = imread(testimg);
        [r c] = size(img); % get the size of image
        temp = reshape(img',r*c,1); %reshaping the image
        temp = double(temp)-m; % mean subtracted vector
        projtestimg = eigenfaces'*temp; %projecting test image
        
        euclide_dist = [ ];
        for k=1 : size(eigenfaces,2)
            temp = (norm(projtestimg-projectimg(:,k)))^2;
            euclide_dist = [euclide_dist temp];
        end
        [euclide_dist_min recognized_index] = min(euclide_dist);
        
        subplot(1,2,1);imshow((img));title('Test Face');
        subplot(1,2,2);imshow((read(trainDataset,recognized_index)));title('Recognised Face');
        disp(euclide_dist_min);
       
        
end


