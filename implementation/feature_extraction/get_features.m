function feature_pixels = get_features(image, features, gparams, feat, layerInd)

if ~ iscell(features)
    features = {features};
end

[im_height, im_width, ~, num_images] = size(image);
% [im_height, im_width, num_im_chan, num_images] = size(image);

switch feat
    case 'fhog'
        fg_size = [floor(im_height/gparams.cell_size), floor(im_width/gparams.cell_size)];
        feature_pixels = zeros(fg_size(1),fg_size(2),features{1}.hog_params.nDim, num_images, 'single');
        feature_pixels(:,:,1:features{1}.hog_params.nDim,:) = features{1}.getFeature_fhog(image,features{1}.hog_params,gparams);
    case 'cn'
        fg_size = [floor(im_height/gparams.cell_size), floor(im_width/gparams.cell_size)];
        feature_pixels = zeros(fg_size(1),fg_size(2),features{1}.cn_params.nDim, num_images, 'single');
        feature_pixels(:,:,1:features{1}.cn_params.nDim,:) = features{1}.getFeature_cn(image,features{1}.cn_params,gparams);
    case 'handcrafted_assem'
        fg_size = [floor(im_height/gparams.cell_size), floor(im_width/gparams.cell_size)];
        feature_pixels = zeros(fg_size(1),fg_size(2),features{1}.handcrafted_params.nDim, num_images, 'single');
        feature_pixels(:,:,1:features{1}.handcrafted_params.nDim,:) = features{1}.getFeature_handcrafted(image,features{1}.handcrafted_params,gparams);
    case 'conv3'
        fg_size = [floor(im_height/gparams.cell_size), floor(im_width/gparams.cell_size)];
        feature_pixels = zeros(fg_size(1),fg_size(2),features{1}.deep_params.nDim(layerInd), num_images, 'single');
        feature_pixels(:,:,1:features{1}.deep_params.nDim(layerInd),:) = features{1}.getFeature_deep(image,features{1}.deep_params,gparams,layerInd);
end
end