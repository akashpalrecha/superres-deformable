function y_channel = get_y_channel(image)
    % Get the Y channel from an image after converting it to YCbCr space
    if 3 == size(image, 3)
        im_ycbcr = single(rgb2ycbcr(im2double(image)));
        y_channel = im_ycbcr(:, :, 1);
    else
        y_channel = single(im2double(image));
    end
end