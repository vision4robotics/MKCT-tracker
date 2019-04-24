function results = tracker(p, im, bg_area, fg_area, area_resize_factor)

%% INITIALIZATION
gamma_t = p.r_time * 0.5 .^ (1:p.mf_num);
feat_type = p.feat_type;
layerInd = p.layerInd;
lambda1 = p.lambda1;
lambda2 = p.lambda2;
features = p.t_features;
global_feat_params = p.t_global;
num_frames = numel(p.img_files);
s_frames = p.s_frames;
video_path = p.video_path;
% used for benchmark
rect_positions = zeros(num_frames, 4);
pos = p.init_pos;
target_sz = p.target_sz;
hann_window_cosine = single(hann(p.cf_response_size(1)) * hann(p.cf_response_size(2))');

% patch position
offset = [-target_sz(1) 0; 0 -target_sz(2); target_sz(1) 0; 0 target_sz(2)];

output_sigma{1} = sqrt(prod(p.norm_target_sz)) * p.output_sigma_factor{1} / p.hog_cell_size;
output_sigma{2} = sqrt(prod(p.norm_target_sz)) * p.output_sigma_factor{2} / p.hog_cell_size;
y{1} = gaussianResponse(p.cf_response_size, output_sigma{1});
yf{1} = fft2(y{1});
y{2} = gaussianResponse(p.cf_response_size, output_sigma{2});
yf{2} = fft2(y{2});

% variables initialization
model_x_f = cell(2,1);
model_w_f = cell(2,1);
model_1st_w_f = cell(2,1);
z  = cell(2,1);
z_f = cell(2,1);
kz_f = cell(2,1);
x = cell(2,1);
x_f = cell(2,1);
k_f = cell(2,1);
model_w_mfs = cell(2, p.mf_num);
r_w_mfs = cell(2, p.mf_num);

learning_rate_pwp = p.learning_rate_pwp;
patch_padded = getSubwindow(im, pos, p.norm_bg_area, bg_area);
new_pwp_model = true;
[bg_hist, fg_hist] = updateHistModel(new_pwp_model, patch_padded, bg_area, fg_area, target_sz, p.norm_bg_area, p.n_bins, p.grayscale_sequence);
new_pwp_model = false;

%% Scale filter
% from DSST ******************************************
p.num_scales = 33;
p.hog_scale_cell_size = 4;
p.learning_rate_scale = 0.025;
p.scale_sigma_factor = 1/2;
p.scale_model_factor = 1.0;
p.scale_step = 1.03;
p.scale_model_max_area = 32*16;
p.lambda = 1e-4;% 1e-4
scale_factor = 1;
base_target_sz = target_sz;
scale_sigma = sqrt(p.num_scales) * p.scale_sigma_factor;
ss = (1:p.num_scales) - ceil(p.num_scales/2);
ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
ysf = single(fft(ys));
if mod(p.num_scales,2) == 0
    scale_window = single(hann(p.num_scales+1));
    scale_window = scale_window(2:end);
else
    scale_window = single(hann(p.num_scales));
end
ss = 1:p.num_scales;
scale_factors = p.scale_step.^(ceil(p.num_scales/2) - ss);
if p.scale_model_factor^2 * prod(p.norm_target_sz) > p.scale_model_max_area
    p.scale_model_factor = sqrt(p.scale_model_max_area/prod(p.norm_target_sz));
end
scale_model_sz = floor(p.norm_target_sz * p.scale_model_factor);
% find maximum and minimum scales
min_scale_factor = p.scale_step ^ ceil(log(max(5 ./ bg_area)) / log(p.scale_step));
max_scale_factor = p.scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ target_sz)) / log(p.scale_step));

%% MAIN LOOP
t_imread = 0;
tic;
for frame = 1:num_frames
    if frame > 1
        tic_imread = tic;
        % Load the image at the current frame
        im = imread([s_frames{frame}]);
        t_imread = t_imread + toc(tic_imread);
        
        im_patch_cf = getSubwindow(im, pos, p.norm_bg_area, bg_area);
        likelihood_map = getColourMap(im_patch_cf, bg_hist, fg_hist, p.n_bins, p.grayscale_sequence);
        likelihood_map(isnan(likelihood_map)) = 0;
        likelihood_map = imResample(likelihood_map, p.cf_response_size);
        likelihood_map = (likelihood_map + min(likelihood_map(:)))/(max(likelihood_map(:)) + min(likelihood_map(:)));
        if (sum(likelihood_map(:))/prod(p.cf_response_size)<0.01)
            likelihood_map = 1;
        end
        likelihood_map = max(likelihood_map, 0.1);
        hann_window =  hann_window_cosine .* likelihood_map;
        for M = 1:2
            z{M} = bsxfun(@times, get_features(im_patch_cf, features, global_feat_params, feat_type{M}, layerInd{M}), hann_window);
            z_f{M} = double(fft2(z{M}));
            switch p.kernel_type{M}
                case 'gaussian'
                    kz_f{M} = gaussian_correlation(z_f{M}, model_x_f{M}, p.tran_sigma{M});
                case 'polynomial'
                    kz_f{M} = polynomial_correlation(z_f{M}, model_x_f{M}, p.polya{M}, p.polyb{M});
                case 'linear'
                    kz_f{M} = sum(z_f{M} .* conj(model_x_f{M}), 3) / numel(z_f{M});
            end
        end
        response_cf{1} = real(ifft2(model_w_f{1} .* kz_f{1}));
        response_cf{2} = real(ifft2(model_w_f{2} .* kz_f{2}));
        
        % Crop square search region (in feature pixels).
        response_cf{1} = cropFilterResponse(response_cf{1}, ...
            floor_odd(p.norm_delta_area / p.hog_cell_size));
        response_cf{2} = cropFilterResponse(response_cf{2}, ...
            floor_odd(p.norm_delta_area / p.hog_cell_size));
        
        if p.hog_cell_size > 1
            % Scale up to match center likelihood resolution.
            response_cf{1} = mexResize(response_cf{1}, p.norm_delta_area,'auto');
            response_cf{2} = mexResize(response_cf{2}, p.norm_delta_area,'auto');
        end
        p1 = adaptive_weight(response_cf{1});
        p2 = adaptive_weight(response_cf{2});
        sum_p = p1 + p2;
        p1 = p1/sum_p; p2 = p2/sum_p;
        response_cf_all = ...
            (p1.*response_cf{1}./max(response_cf{1}(:))) + ...
            (p2.*response_cf{2}./max(response_cf{2}(:)));
        center = (1+p.norm_delta_area) / 2;
        
        % resolution enhancement
        response = REO(response_cf_all, 3);
        
        [row, col] = find(response == max(response(:)));
        row = row(1);
        col = col(1);
        delta_row = row - center(1);
        delta_col = col - center(2);
        pos = pos + ([delta_row, delta_col]) / area_resize_factor;
        rect_position = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        %% SCALE SPACE SEARCH
        im_patch_scale = getScaleSubwindow(im, pos, base_target_sz, scale_factor * scale_factors, scale_window, scale_model_sz, p.hog_scale_cell_size);
        xsf = fft(im_patch_scale,[],2);
        scale_response = real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den + p.lambda) ));
        recovered_scale = ind2sub(size(scale_response),find(scale_response == max(scale_response(:)), 1));
        scale_factor = scale_factor * scale_factors(recovered_scale);
        
        if scale_factor < min_scale_factor
            scale_factor = min_scale_factor;
        elseif scale_factor > max_scale_factor
            scale_factor = max_scale_factor;
        end
        % use new scale to update bboxes for target, filter, bg and fg models
        target_sz = round(base_target_sz * scale_factor);
        p.avg_dim = sum(target_sz)/2;
        bg_area = round(target_sz + p.avg_dim * p.padding);
        if(bg_area(2)>size(im,2)),  bg_area(2)=size(im,2)-1;    end
        if(bg_area(1)>size(im,1)),  bg_area(1)=size(im,1)-1;    end
        
        bg_area = bg_area - mod(bg_area - target_sz, 2);
        fg_area = round(target_sz - p.avg_dim * p.inner_padding);
        fg_area = fg_area + mod(bg_area - fg_area, 2);
        % Compute the rectangle with (or close to) params.fixed_area and same aspect ratio as the target bboxgetScaleSubwindow
        area_resize_factor = sqrt(p.fixed_area/prod(bg_area));
    end
    
    %% Train and Update Model
    %% center patch
    obj = getSubwindow(im, pos, target_sz);
    im_patch_fg = getSubwindow(im, pos, p.norm_bg_area, bg_area);
    for M = 1:2
        x{M} = bsxfun(@times, get_features(im_patch_fg, features, global_feat_params, feat_type{M}, layerInd{M}), hann_window_cosine);
        x_f{M} = double(fft2(x{M}));
        switch p.kernel_type{M}
            case 'gaussian'
                k_f{M} = gaussian_correlation(x_f{M}, x_f{M}, p.tran_sigma{M});
            case 'polynomial'
                k_f{M} = polynomial_correlation(x_f{M}, x_f{M}, p.polya{M}, p.polyb{M});
            case 'linear'
                k_f{M} = sum(x_f{M} .* conj(x_f{M}), 3) / numel(x_f{M});
        end
    end
    
    %% dynamic weighted filters
    valid = floor((frame-3) / p.mfl_interv) + 1;
    if valid <= 0
        gamma_sum = 0;multi_frame_sum = 0;
    elseif valid > 0 && valid < p.mf_num
        gamma_sum = sumsqr(gamma_t(1:valid));
        if valid > 1
            for ft = 2:valid
                model_w_mfs{1,ft} = model_w_mfs{1,ft-1};
                r_w_mfs{1,ft} = gamma_t(ft)^2 * model_w_mfs{1,ft};
                model_w_mfs{2,ft} = model_w_mfs{2,ft-1};
                r_w_mfs{2,ft} = gamma_t(ft)^2 * model_w_mfs{2,ft};
            end
        end
        model_w_mfs{1} = model_w_f{1};
        r_w_mfs{1,1} = gamma_t(1)^2 * model_w_mfs{1,1};
        model_w_mfs{2} = model_w_f{2};
        r_w_mfs{2,1} = gamma_t(2)^2 * model_w_mfs{1,1};
        multi_frame_sum1 = sum(cat(3,r_w_mfs{1,:}),3);
        multi_frame_sum2 = sum(cat(3,r_w_mfs{2,:}),3);
    else
        gamma_sum = sumsqr(gamma_t(1:p.mf_num));
        for ft = 2:p.mf_num
            model_w_mfs{1,ft} = model_w_mfs{1,ft-1};
            r_w_mfs{1,ft} = gamma_t(ft)^2 * model_w_mfs{1,ft};
            model_w_mfs{2,ft} = model_w_mfs{2,ft-1};
            r_w_mfs{2,ft} = gamma_t(ft)^2 * model_w_mfs{2,ft};
        end
        model_w_mfs{1,1} = model_w_f{1};
        r_w_mfs{1,1} = gamma_t(1)^2 * model_w_mfs{1,1};
        model_w_mfs{2,1} = model_w_f{2};
        r_w_mfs{2,1} = gamma_t(2)^2 * model_w_mfs{2,1};
        multi_frame_sum1 = sum(cat(3,r_w_mfs{1,:}),3);
        multi_frame_sum2 = sum(cat(3,r_w_mfs{2,:}),3);
    end
    
    %% adaptive GMSD-based context learning
    if mod(frame - 1, p.context_l_interv) == 0
        x_bf{1} = zeros([size(x_f{1}) length(offset)]);
        for j = 1:length(offset)
            context = getSubwindow(im, pos+offset(j,:), target_sz);
            im_patch_bg = getSubwindow(im, pos+offset(j,:), p.norm_bg_area, bg_area);
            IQM = GMSD_factor(obj, context);
            factor(j) = p.r_similar * exp(1 - IQM);
            Factor = factor * sqrt(lambda2);
            x_b = bsxfun(@times, get_features(im_patch_bg, features, global_feat_params, feat_type{1}, layerInd{1}), hann_window_cosine);
            x_bf{1}(:,:,:,j) = fft2(x_b);
            switch p.kernel_type{1}
                case 'gaussian'
                    k_bf{1}(:,:,j) = gaussian_correlation(x_bf{1}(:,:,:,j), x_bf{1}(:,:,:,j), p.tran_sigma{1});
                case 'polynomial'
                    k_bf{1}(:,:,j) = polynomial_correlation(x_bf{1}(:,:,:,j), x_bf{1}(:,:,:,j), p.polya{1}, p.polyb{1});
                case 'linear'
                    k_bf{1} = sum(x_bf{1}(:,:,:,j) .* conj(x_bf{1}(:,:,:,j)), 3) / numel(x_bf{1}(:,:,:,j));
            end
            k_bf{1}(:,:,j) = Factor(j) * k_bf{1}(:,:,j);
        end
        if ~multi_frame_sum
            new_wf_num{1} = k_f{1} .* yf{1};
            new_wf_den{1} = k_f{1} .* conj(k_f{1}) + lambda1 + sum(k_bf{1} .* conj(k_bf{1}), 3);
            new_wf_num{2} = k_f{2} .* yf{2};
            new_wf_den{2} = k_f{2} .* conj(k_f{2}) + lambda1;
        else
            new_wf_num{1} = k_f{1} .* yf{1} + multi_frame_sum1;
            new_wf_den{1} = k_f{1} .* conj(k_f{1}) + lambda1 + gamma_sum + sum(k_bf{1} .* conj(k_bf{1}), 3);
            new_wf_num{2} = k_f{2} .* yf{2} + multi_frame_sum2;
            new_wf_den{2} = k_f{2} .* conj(k_f{2}) + lambda1 + gamma_sum;
        end
    else
        if ~multi_frame_sum
            new_wf_num{1} = k_f{1} .* yf{1};
            new_wf_den{1} = k_f{1} .* conj(k_f{1}) + lambda1;
            new_wf_num{2} = k_f{2} .* yf{2};
            new_wf_den{2} = k_f{2} .* conj(k_f{2}) + lambda1;
        else
            new_wf_num{1} = k_f{1} .* yf{1} + multi_frame_sum1;
            new_wf_den{1} = k_f{1} .* conj(k_f{1}) + lambda1 + gamma_sum;
            new_wf_num{2} = k_f{2} .* yf{2} + multi_frame_sum2;
            new_wf_den{2} = k_f{2} .* conj(k_f{2}) + lambda1 + gamma_sum;
        end
    end
    if ~multi_frame_sum
        new_wf_num{1} = k_f{1} .* yf{1};
        new_wf_den{1} = k_f{1} .* conj(k_f{1}) + lambda1;
        new_wf_num{2} = k_f{2} .* yf{2};
        new_wf_den{2} = k_f{2} .* conj(k_f{2}) + lambda1;
    else
        new_wf_num{1} = k_f{1} .* yf{1} + multi_frame_sum1;
        new_wf_den{1} = k_f{1} .* conj(k_f{1}) + lambda1 + gamma_sum;
        new_wf_num{2} = k_f{2} .* yf{2} + multi_frame_sum2;
        new_wf_den{2} = k_f{2} .* conj(k_f{2}) + lambda1 + gamma_sum;
    end
    w_f{1} = new_wf_num{1} ./ new_wf_den{1};
    w_f{2} = new_wf_num{2} ./ new_wf_den{2};
    
    %% Initialization for first frame
    if frame == 1
        model_x_f = x_f;
        model_w_f = w_f;
        model_1st_w_f = w_f;
    else
        % subsequent frames, update the model by linear interpolation
        model_x_f{1} = (1 - p.learning_rate_cf) * model_x_f{1} + p.learning_rate_cf * x_f{1};
        model_x_f{2} = (1 - p.learning_rate_cf) * model_x_f{2} + p.learning_rate_cf * x_f{2};
        
        model_w_f{1} = (1 - p.learning_rate_cf) * model_w_f{1} + p.learning_rate_cf * w_f{1};
        model_w_f{2} = (1 - p.learning_rate_cf) * model_w_f{2} + p.learning_rate_cf * w_f{2};
        
        % BG/FG MODEL UPDATE   patch of the target + padding
        im_patch_color = getSubwindow(im, pos, p.norm_bg_area, bg_area*(1-p.inner_padding));
        [bg_hist, fg_hist] = updateHistModel(new_pwp_model, im_patch_color, bg_area, fg_area, target_sz, p.norm_bg_area, p.n_bins, p.grayscale_sequence, bg_hist, fg_hist, learning_rate_pwp);
    end
    %% Upadate Scale
    im_patch_scale = getScaleSubwindow(im, pos, base_target_sz, scale_factor*scale_factors, scale_window, scale_model_sz, p.hog_scale_cell_size);
    xsf = fft(im_patch_scale,[],2);
    new_sf_num = bsxfun(@times, ysf, conj(xsf));
    new_sf_den = sum(xsf .* conj(xsf), 1);
    if frame == 1
        sf_den = new_sf_den;
        sf_num = new_sf_num;
    else
        sf_den = (1 - p.learning_rate_scale) * sf_den + p.learning_rate_scale * new_sf_den;
        sf_num = (1 - p.learning_rate_scale) * sf_num + p.learning_rate_scale * new_sf_num;
    end
    % update bbox position
    if (frame == 1)
        rect_position = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
    end
    rect_position_padded = [pos([2,1]) - bg_area([2,1])/2, bg_area([2,1])];
    rect_positions(frame,:) = rect_position;
    elapsed_time = toc;
    
    %% Visualization
    if p.visualization == 1
        if frame == 1   %first frame, create GUI
            figure('Name',['Tracker - ' video_path]);
            im_handle = imshow(uint8(im), 'Border','tight', 'InitialMag', 100 + 100 * (length(im) < 500));
            rect_handle = rectangle('Position',rect_position, 'EdgeColor','g', 'LineWidth',2);
            rect_handle2 = rectangle('Position',rect_position_padded, 'LineWidth',2, 'LineStyle','--', 'EdgeColor','b');
            text_handle = text(10, 10, int2str(frame));
            set(text_handle, 'color', [0 1 1]);
        else
            try  %subsequent frames, update GUI
                set(im_handle, 'CData', im);
                set(rect_handle, 'Position', rect_position);
                set(rect_handle2, 'Position', rect_position_padded);
                set(text_handle, 'string', int2str(frame));
            catch
                return
            end
        end
        
        drawnow
    end
end

%% save data for benchmark
results.type = 'rect';
results.res = rect_positions;
results.fps = num_frames/(elapsed_time - t_imread);

end