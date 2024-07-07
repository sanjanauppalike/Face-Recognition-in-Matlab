function[projectimg,eigenfaces,folder,m] = trainm()

    folder ='G:\matlab_codes\eigenFacesmatlab\train1';
    trainDataset = imageSet(folder);


%%%%%%%%%%%%%%%%%%%%%%%%%%  creating the image matrix X  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    X=[];
    for i = 1:trainDataset.Count
        img = read(trainDataset,i);
        [r, c] = size(img);
        temp = reshape(img,r*c,1);% reshape image
        X = [X temp];  
    
    end
%% calculating mean image vector %%

    m = mean(X,2); % Computing the average face image m = (1/P)*sum(Xj's)    (j = 1 : P)
    imgcount = size(X,2);

%%  calculating A matrix, i.e. after subtraction of all image vectors from the mean image vector %%%%%%

    A = [];
    for i=1 : imgcount
        temp = double(X(:,i)) - m;
        A = [A temp];
    end

%% Calucating eigen vectors from A matrix %%

    L= A' * A;
    [V,D]=eig(L);  %% V : eigenvector matrix  D : eigenvalue matrix
%% Selecting k eigen vectors where k is less than n%%

    L_eig_vec = [];
    for i = 1 : size(V,2) 
        if( D(i,i) > 1 )
               L_eig_vec = [L_eig_vec V(:,i)];
        end
    end

%%% finally the eigenfaces %%%
    eigenfaces = A * L_eig_vec; 
    eigenfaces = A * L_eig_vec;
    face = eigenfaces(:,1);

    face2 = reshape(face,r,c);

    %figure;
    %imshow(face2);
%% finding the projection of each image vector on the facespace (where the eigenfaces are the co-ordinates or dimensions) %%%%%


    projectimg = [ ];  % projected image vector matrix
    for i = 1 : size(eigenfaces,2)
        temp = eigenfaces' * A(:,i); % by projectiong we get the weights which whn multiplied to respective eigen faces 
                                        %gives the image that is considered
        projectimg = [projectimg temp];
   
    end

    disp('Training completed');
end
