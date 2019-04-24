function results = run_MKCT(seq)

params.visualization = 1;
% 'gaussian' 'polynomial' 'linear'
kernel_type{1} = 'gaussian';                   % gaussian kernel function for HC feature
kernel_type{2} = 'polynomial';                 % polynomial kernel function for deep feature
params.kernel_type = kernel_type;
params.tran_sigma = {0.5, 0.5};
params.polya = {1,7};
params.polyb = {1,2};
% multi-frames' restraints
params.mf_num  = 5;                            % if multi-frames' restraints available
params.r_time = 1.0;                           % regulation term for frames' restraints
params.mfl_interv = 2;                         % learning intervals for multi-frame /frame
% context similarity's restraints
params.context_l_interv = 5;                   % learning intervals for context similarity /frame
params.r_similar = 1.0;                        % regulation term for similarity
params.lambda1 = 1e-4;
params.lambda2 = 1/(16^2);

params.hog_cell_size = 4;
params.fixed_area = 200^2;   % 150^2           % standard area to which we resize the target
params.n_bins = 2^5;                           % number of bins for the color histograms (bg and fg models)
params.learning_rate_pwp = 0.02;               % bg and fg color models learning rate
params.lambda_scale = 0.1;                     % regularization weight
params.scale_sigma_factor = 1/16;
params.scale_sigma = 0.1;
params.merge_factor = 0.3;

params.hog_scale_cell_size = 4;
params.scale_model_factor = 1.0;

params.feature_type = 'fhog';
params.scale_adaptation = true;
params.grayscale_sequence = false;	          % suppose that sequence is colour
params.merge_method = 'const_factor';

params.img_files = seq.s_frames;
params.img_path = '';

s_frames = seq.s_frames;
params.s_frames = s_frames;
params.video_path = seq.video_path;
im = imread([s_frames{1}]);
if(size(im,3)==1)
    params.grayscale_sequence = true;
end

region = seq.init_rect;
if(numel(region)==8)
    % polygon format (VOT14, VOT15)
    [cx, cy, w, h] = getAxisAlignedBB(region);
else % rectangle format (WuCVPR13)
    x = region(1);
    y = region(2);
    w = region(3);
    h = region(4);
    cx = x+w/2;
    cy = y+h/2;
end

% init_pos is the centre of the initial bounding box
params.init_pos = [cy cx];                     
params.target_sz = round([h w]);
params.inner_padding = 0.2;
[params, bg_area, fg_area, area_resize_factor] = initializeAllAreas(im, params);

hog_params.nDim  = 31;                      % HOG feature parameters
cn_params.nDim = 11;                        % CN feature parameters
% Deep feature parameters
params.indLayers = [37, 28, 19];            % The CNN layers Conv3-4 in VGG Net
deep_params.nDim = [512, 512, 256];
deep_params.layers = params.indLayers;
%   handcrafted parameters
Feat1 = 'handcrafted_assem';
Feat2 = 'conv3';
params.layerInd{1} = 0;
params.layerInd{2} = 3;
params.feat_type = {Feat1, Feat2};

params.t_global.type_assem = 'fhog_cn';
switch params.t_global.type_assem
    case 'fhog_cn'
        handcrafted_params.nDim = hog_params.nDim + cn_params.nDim;
end

params.t_features = {struct('getFeature_fhog',@get_fhog,...
                    'getFeature_cn',@get_cn,...
                    'getFeature_deep',@get_deep,...
                    'getFeature_handcrafted',@get_handcrafted,...
                    'hog_params',hog_params,...
                    'cn_params',cn_params,...
                    'deep_params',deep_params,...
                    'handcrafted_params',handcrafted_params)};

params.t_global.w2c_mat = load('w2c.mat');
params.t_global.factor = 0.2;
params.t_global.cell_size = 4;
params.t_global.cell_selection_thresh = 0.75^2;
params.output_sigma_factor = {1/40, 1/16};    % label functionµÄÓ°Ïì,1/16
params.learning_rate_cf = 0.015;

% start the actual tracking
results = tracker(params, im, bg_area, fg_area, area_resize_factor);