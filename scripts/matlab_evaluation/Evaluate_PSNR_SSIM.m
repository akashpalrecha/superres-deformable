function Evaluate_PSNR_SSIM(hr_folder, sr_folder, scale, suffix, output_file, tag, extension)
    % This function can be used to evaluate PSNR and SSIM metrics between
    % contents of 2 arbitrary folders
    % Parameters:
    % hr_folder: Folder containing high resolution images
    % sr_folder: Folder containing model output images (super-resolved)
    % scale: scale of super-resolution
    % suffix: suffix if the corresponding images in the sr_folder. For
    % example, if hr_folder contains baboon.png and sr_folder contains
    % baboon_x2_SR.png, then suffix='_x2_SR'
    % output_file: path of file to output evaluation results to
    % tag: identifier text to be added to the top of the results file
    % extension: extension of all images to be evaluated
    if nargin < 6
        tag = "No Tag Provided"
        extension = ".png";
    end
    if nargin < 7
        extension = ".png";
    end
        
    scale = str2num(scale);   

    hr_pattern = fullfile(hr_folder, "*" + extension);
    dirVar = dir(hr_pattern);
    hr_filenames = {dirVar.name};
    total = length(hr_filenames);
    [folder, name, ext] = fileparts(output_file);
    if ~exist(folder, 'dir')
        mkdir(folder)
    end
    display(output_file)
    results = fopen(output_file, 'wt');
    fprintf(results, "********* EVALUTAION OUTPUT *********\n");
    fprintf(results, tag+"\n");
    fprintf(results, "********* ————————————————— *********\n");
    
    
    PSNR_all = zeros(1, total);
    SSIM_all = zeros(1, total);
    
    for i=1:total
        hr_name = string(fullfile(hr_folder, hr_filenames(i)));
        [folder, name, ext] = fileparts(hr_name);
        sr_name = fullfile(sr_folder, name + suffix + extension);
        hr_y = get_y_channel(imread(hr_name));
        sr_y= get_y_channel(imread(sr_name));
        % calculate PSNR, SSIM
        [PSNR_all(i), SSIM_all(i)] = Cal_Y_PSNRSSIM(hr_y*255, sr_y*255, scale, scale);
        fprintf(results, 'x%d %d %s: PSNR= %f SSIM= %f\n', scale, i, sr_name, PSNR_all(i), SSIM_all(i));
    end
    
    fprintf(results, '--------Mean--------\n');
    fprintf('--------Mean--------\n');
    fprintf(results, 'x%d: PSNR= %f SSIM= %f\n', scale, mean(PSNR_all), mean(SSIM_all));
    fprintf('x%d: PSNR= %f SSIM= %f\n', scale, mean(PSNR_all), mean(SSIM_all));
    fclose(results);
end 
